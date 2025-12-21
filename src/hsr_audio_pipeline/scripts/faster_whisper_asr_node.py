#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
faster_whisper_asr_node.py (ROS1 / Noetic)

Sub:
  - ~audio_topic  (audio_common_msgs/AudioData) default: /audio/audio
  - ~vad_topic    (std_msgs/Bool)               default: /vad/is_speech

Pub:
  - ~raw_text_topic       (std_msgs/String) default: /gpsr/asr/raw_text
  - ~text_topic           (std_msgs/String) default: /gpsr/asr/text
  - ~utterance_end_topic  (std_msgs/Bool)   default: /gpsr/asr/utterance_end
  - ~confidence_topic     (std_msgs/Float32) default: /gpsr/asr/confidence (optional)

Key features:
  - External VAD drives segmentation
  - pre_roll + post_roll audio
  - Uses faster-whisper WhisperModel
  - Hotwords from MD vocab (/data/vocab/*.md) preferred; fallback to vocab.yaml
      * builds hotwords STRING (NOT list) -> avoids .strip() crash
  - Loads corrections YAML via ~corrections_yaml (default: /data/vocab/corrections.yaml)
      * merged with vocab.yaml's embedded corrections (corrections_yaml wins)
  - Model cache persistence:
      * ~model_cache_dir (HF cache root; writable) + auto fallback
      * ~torch_home (torch hub cache root; writable) + auto fallback
      * ~model_path (optional local model dir/file; if missing fallback to model_size)

Notes:
  - To persist across container restarts, mount host dir to /data (compose bind mount).
  - MD vocab files are expected under /data/vocab/:
      names.md, room_names.md, location_names.md, objects.md, (optional) test_objects.md
"""

import os
import re
import time
import yaml
import threading
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import rospy

from std_msgs.msg import String, Bool, Float32
from audio_common_msgs.msg import AudioData

from faster_whisper import WhisperModel


# ----------------------------
# helpers
# ----------------------------
def _normalize_ws(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s


def _lower(s: str) -> str:
    return _normalize_ws(s).lower()


def _unique_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _ensure_writable_dir(preferred: str, fallback: str, label: str) -> str:
    """
    Return a writable directory path.
    - Try preferred; if cannot mkdir/write -> fallback.
    """
    preferred = os.path.expanduser(preferred or "")
    fallback = os.path.expanduser(fallback or "")

    def _try(path: str) -> bool:
        if not path:
            return False
        try:
            os.makedirs(path, exist_ok=True)
            testfile = os.path.join(path, ".write_test")
            with open(testfile, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(testfile)
            return True
        except Exception as e:
            rospy.logwarn("Could not create/use %s dir: %s (%s)", label, path, e)
            return False

    if _try(preferred):
        return preferred
    if _try(fallback):
        rospy.logwarn("%s dir fallback -> %s", label, fallback)
        return fallback

    rospy.logwarn("%s dir fallback -> (.)", label)
    return "."


def _load_corrections_any(y) -> Dict[str, str]:
    """
    Accept:
      - dict: {from: to, ...}
      - list of dict: [{from: "a", to:"b"}, ...]
      - list of pairs: [["a","b"], ...] (optional)
      - list of strings: ["a->b", ...] (optional)
    Return normalized lower map.
    """
    corr_map: Dict[str, str] = {}

    if isinstance(y, dict):
        for k, v in y.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                corr_map[_lower(k)] = _lower(v)
        return corr_map

    if isinstance(y, list):
        for it in y:
            if isinstance(it, dict):
                f = it.get("from")
                t = it.get("to")
                if isinstance(f, str) and isinstance(t, str) and f.strip() and t.strip():
                    corr_map[_lower(f)] = _lower(t)
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                f, t = it[0], it[1]
                if isinstance(f, str) and isinstance(t, str) and f.strip() and t.strip():
                    corr_map[_lower(f)] = _lower(t)
            elif isinstance(it, str):
                # "foo -> bar" or "foo=>bar"
                m = re.split(r"\s*(?:->|=>)\s*", it.strip(), maxsplit=1)
                if len(m) == 2 and m[0].strip() and m[1].strip():
                    corr_map[_lower(m[0])] = _lower(m[1])
        return corr_map

    return corr_map


def _words_from_vocab_yaml(path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    vocab.yaml 想定（柔軟に拾う）:
      objects: [...]
      categories: [...]
      names: [...]
      locations:
        rooms: [...]
        placements: [{name: "...", placement: true/false}, ...] or ["...", ...]
      corrections: (optional)
    """
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    words: List[str] = []

    def add_list(key: str):
        v = y.get(key)
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str) and s.strip():
                    words.append(s.strip())

    add_list("objects")
    add_list("categories")
    add_list("names")

    loc = y.get("locations", {})
    if isinstance(loc, dict):
        rooms = loc.get("rooms")
        if isinstance(rooms, list):
            for s in rooms:
                if isinstance(s, str) and s.strip():
                    words.append(s.strip())
        pls = loc.get("placements")
        if isinstance(pls, list):
            for p in pls:
                if isinstance(p, dict):
                    n = p.get("name")
                    if isinstance(n, str) and n.strip():
                        words.append(n.strip())
                elif isinstance(p, str) and p.strip():
                    words.append(p.strip())

    # optional flat "placements"
    pls2 = y.get("placements")
    if isinstance(pls2, list):
        for p in pls2:
            if isinstance(p, dict):
                n = p.get("name")
                if isinstance(n, str) and n.strip():
                    words.append(n.strip())
            elif isinstance(p, str) and p.strip():
                words.append(p.strip())

    # embedded corrections (optional)
    corr_map = _load_corrections_any(y.get("corrections"))

    words = _unique_keep_order([w for w in words if w])
    return words, corr_map


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _words_from_md(
    names_md: str,
    rooms_md: str,
    locations_md: str,
    objects_md: str,
    test_objects_md: str = "",
) -> List[str]:
    """
    MD files under /data/vocab/ (robocupathome generator-style tables):
      names.md
      room_names.md
      location_names.md (with (p) suffix for placement locations)
      objects.md (with # Class plural singular ... sections)
      test_objects.md (optional; for hotwords only)
    """
    # names
    names = re.findall(r"\|\s*([A-Za-z]+)\s*\|", _safe_read(names_md))
    names = [x.strip() for x in names][1:] if names else []

    # rooms
    rooms = re.findall(r"\|\s*(\w+ \w*)\s*\|", _safe_read(rooms_md))
    rooms = [x.strip() for x in rooms][1:] if rooms else []

    # locations (+ placement)
    loc_pairs = re.findall(r"\|\s*([0-9]+)\s*\|\s*([A-Za-z,\s\(\)]+)\|", _safe_read(locations_md))
    locs = [b.strip() for (_, b) in loc_pairs]
    placement = [x.replace("(p)", "").strip() for x in locs if x.strip().endswith("(p)")]
    locs = [x.replace("(p)", "").strip() for x in locs]

    # objects + categories
    md_obj = _safe_read(objects_md)
    obj_names = re.findall(r"\|\s*(\w+)\s*\|", md_obj)
    obj_names = [o for o in obj_names if o != "Objectname"]
    obj_names = [o.replace("_", " ").strip() for o in obj_names if o.strip()]

    cats = re.findall(r"# Class \s*([\w,\s\(\)]+)\s*", md_obj)
    cats = [c.strip().replace("(", "").replace(")", "") for c in cats]
    cat_plur, cat_sing = [], []
    for c in cats:
        parts = c.split()
        if len(parts) >= 2:
            cat_plur.append(parts[0].replace("_", " "))
            cat_sing.append(parts[1].replace("_", " "))

    # test objects (hotwords増量用：objectsに加える)
    if test_objects_md and os.path.exists(test_objects_md):
        md_test = _safe_read(test_objects_md)
        test_objs = re.findall(r"\|\s*(\w+)\s*\|", md_test)
        test_objs = [o for o in test_objs if o != "Objectname"]
        test_objs = [o.replace("_", " ").strip() for o in test_objs if o.strip()]
        obj_names = list(set(obj_names + test_objs))

    words = list(set(names + rooms + locs + placement + obj_names + cat_plur + cat_sing))
    words = [w for w in words if isinstance(w, str) and len(w.strip()) >= 2]
    return words


def _build_hotwords_string(words: List[str], max_terms: int = 250) -> str:
    # faster-whisper の hotwords は “文字列” が安全（内部で .strip() される）
    w = []
    for s in words:
        s2 = _normalize_ws(s)
        if not s2:
            continue
        # 句読点など最小限除去（multiwordは保持）
        s2 = re.sub(r"[^\w\s\-]", " ", s2).strip()
        s2 = " ".join(s2.split())
        if s2:
            w.append(s2)
    w = _unique_keep_order(w)
    if max_terms > 0:
        w = w[:max_terms]
    return " ".join(w)


class CorrectionEngine:
    """
    超軽量の置換（例: foot->food, past->pantry など）
    強くしすぎると事故るので最低限。
    """
    def __init__(self, corr_map: Dict[str, str]):
        self.map = corr_map or {}

    def apply(self, text: str) -> str:
        if not text:
            return ""
        s = _lower(text)
        for f, t in self.map.items():
            if not f or not t:
                continue
            if " " in f:
                s = s.replace(f, t)
            else:
                s = re.sub(rf"\b{re.escape(f)}\b", t, s)
        s = _normalize_ws(s)
        return s


class FasterWhisperASRNode:
    def __init__(self):
        rospy.init_node("faster_whisper_asr_node")

        # ---- topics ----
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.raw_text_topic = rospy.get_param("~raw_text_topic", "/gpsr/asr/raw_text")
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")

        # ---- audio format ----
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.sample_width = int(rospy.get_param("~sample_width", 2))  # bytes (S16LE)
        self.channels = int(rospy.get_param("~channels", 1))

        # ---- VAD segmentation ----
        self.pre_roll = float(rospy.get_param("~pre_roll", 0.25))     # seconds
        self.post_roll = float(rospy.get_param("~post_roll", 0.35))   # seconds
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 18.0))

        # ---- faster-whisper ----
        self.model_size = rospy.get_param("~model_size", "small")
        self.model_path = rospy.get_param("~model_path", "")  # optional local dir/file
        self.device = rospy.get_param("~device", "cpu")
        self.compute_type = rospy.get_param("~compute_type", "float32")
        self.language = rospy.get_param("~language", "en")
        self.beam_size = int(rospy.get_param("~beam_size", 5))

        # ---- persistence / cache dirs ----
        preferred_cache = rospy.get_param("~model_cache_dir", "/data/models/hf")
        preferred_torch = rospy.get_param("~torch_home", "/data/models/torch")

        fallback_cache = os.path.expanduser("~/.cache/hsr_models/hf")
        fallback_torch = os.path.expanduser("~/.cache/hsr_models/torch")

        self.model_cache_dir = _ensure_writable_dir(preferred_cache, fallback_cache, "hf_cache")
        self.torch_home = _ensure_writable_dir(preferred_torch, fallback_torch, "torch_home")

        # Set env vars so HF + torch.hub reuse caches across runs (BEFORE model init)
        os.environ["HF_HOME"] = self.model_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.model_cache_dir, "hub")
        os.environ["XDG_CACHE_HOME"] = os.path.join(self.model_cache_dir, "xdg")
        os.environ["TORCH_HOME"] = self.torch_home

        for d in [os.environ["HUGGINGFACE_HUB_CACHE"], os.environ["XDG_CACHE_HOME"], self.torch_home]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                rospy.logwarn("Could not mkdir %s: %s", d, e)

        rospy.loginfo(
            "HF cache: HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s XDG_CACHE_HOME=%s TORCH_HOME=%s",
            os.environ["HF_HOME"], os.environ["HUGGINGFACE_HUB_CACHE"], os.environ["XDG_CACHE_HOME"], os.environ["TORCH_HOME"]
        )

        # ---- vocab / corrections paths (persist under /data/vocab) ----
        self.vocab_dir = rospy.get_param("~vocab_dir", "/data/vocab")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", os.path.join(self.vocab_dir, "vocab.yaml"))
        self.corrections_yaml = rospy.get_param("~corrections_yaml", os.path.join(self.vocab_dir, "corrections.yaml"))

        # ---- MD vocab paths (preferred for hotwords) ----
        self.names_md = rospy.get_param("~names_md", os.path.join(self.vocab_dir, "names.md"))
        self.rooms_md = rospy.get_param("~rooms_md", os.path.join(self.vocab_dir, "room_names.md"))
        self.locations_md = rospy.get_param("~locations_md", os.path.join(self.vocab_dir, "location_names.md"))
        self.objects_md = rospy.get_param("~objects_md", os.path.join(self.vocab_dir, "objects.md"))
        self.test_objects_md = rospy.get_param("~test_objects_md", os.path.join(self.vocab_dir, "test_objects.md"))

        # ---- prompts / hotwords ----
        self.use_hotwords = bool(rospy.get_param("~use_hotwords", True))
        self.max_hotwords_terms = int(rospy.get_param("~max_hotwords_terms", 250))

        self.initial_prompt = rospy.get_param(
            "~initial_prompt",
            "You are a home service robot. Transcribe spoken GPSR commands in English accurately."
        )

        self.publish_confidence = bool(rospy.get_param("~publish_confidence", True))
        self.enable_corrections = bool(rospy.get_param("~enable_corrections", True))

        # ---- vocab -> hotwords + corrections ----
        self.hotwords_str = ""
        merged_corr: Dict[str, str] = {}

        # 1) load MD vocab for hotwords (preferred)
        vocab_words: List[str] = []
        use_md = all([
            self.names_md and os.path.exists(self.names_md),
            self.rooms_md and os.path.exists(self.rooms_md),
            self.locations_md and os.path.exists(self.locations_md),
            self.objects_md and os.path.exists(self.objects_md),
        ])

        if use_md:
            try:
                vocab_words = _words_from_md(
                    self.names_md,
                    self.rooms_md,
                    self.locations_md,
                    self.objects_md,
                    self.test_objects_md,
                )
                if self.use_hotwords:
                    self.hotwords_str = _build_hotwords_string(vocab_words, max_terms=self.max_hotwords_terms)
                rospy.loginfo(
                    "Loaded MD vocab for hotwords: dir=%s (hotwords_terms=%d)",
                    self.vocab_dir, len(self.hotwords_str.split())
                )
            except Exception as e:
                rospy.logwarn("Failed to load MD vocab (fallback to YAML): %s", e)
                use_md = False

        # 2) load vocab.yaml (fallback for hotwords + embedded corrections)
        vocab_corr: Dict[str, str] = {}
        if not use_md:
            if self.vocab_yaml and os.path.exists(self.vocab_yaml):
                try:
                    vocab_words, vocab_corr = _words_from_vocab_yaml(self.vocab_yaml)
                    if self.use_hotwords:
                        self.hotwords_str = _build_hotwords_string(vocab_words, max_terms=self.max_hotwords_terms)
                    rospy.loginfo(
                        "Loaded vocab_yaml: %s (hotwords_terms=%d)",
                        self.vocab_yaml, len(self.hotwords_str.split())
                    )
                except Exception as e:
                    rospy.logwarn("Failed to load vocab_yaml=%s: %s", self.vocab_yaml, e)
            else:
                rospy.logwarn("vocab_yaml not found: %s", self.vocab_yaml)

        # embedded corrections (optional) from vocab.yaml (fallback)
        merged_corr.update(vocab_corr)

        # 3) load corrections.yaml (overrides vocab corrections)
        if self.corrections_yaml and os.path.exists(self.corrections_yaml):
            try:
                with open(self.corrections_yaml, "r", encoding="utf-8") as f:
                    cy = yaml.safe_load(f) or {}
                cy_map = _load_corrections_any(cy)
                merged_corr.update(cy_map)
                rospy.loginfo("Loaded corrections_yaml: %s (rules=%d)", self.corrections_yaml, len(cy_map))
            except Exception as e:
                rospy.logwarn("Failed to load corrections_yaml=%s: %s", self.corrections_yaml, e)
        else:
            rospy.loginfo("corrections_yaml not found (ok): %s", self.corrections_yaml)

        self.corr = CorrectionEngine(merged_corr if self.enable_corrections else {})
        rospy.loginfo("Corrections: %s (rules=%d)", "on" if self.enable_corrections else "off", len(self.corr.map))

        # ---- model ----
        model_id = self.model_size
        if self.model_path:
            mp = os.path.expanduser(self.model_path)
            if os.path.exists(mp):
                model_id = mp
                rospy.loginfo("Using local model_path: %s", mp)
            else:
                rospy.logwarn("model_path set but not found: %s (fallback to model_size=%s)", mp, self.model_size)

        rospy.loginfo(
            "Loading WhisperModel: model=%s device=%s compute_type=%s download_root=%s",
            model_id, self.device, self.compute_type, self.model_cache_dir
        )

        self.model = WhisperModel(
            model_id,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_cache_dir,
        )

        # ---- ROS pub/sub ----
        self.pub_raw = rospy.Publisher(self.raw_text_topic, String, queue_size=10)
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_utt_
