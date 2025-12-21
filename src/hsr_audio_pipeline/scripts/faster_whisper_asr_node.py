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
  - Loads GPSR vocab YAML via ~vocab_yaml:
      * builds hotwords STRING (NOT list) -> avoids .strip() crash
      * builds light correction dict
  - Model cache persistence:
      * ~model_cache_dir (HF cache root; writable) + auto fallback
      * ~torch_home (torch hub cache for silero etc.) + auto fallback
      * ~model_path (optional local model dir/file; if missing fallback to model_size)
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
    preferred = preferred or ""
    fallback = fallback or ""
    preferred = os.path.expanduser(preferred)
    fallback = os.path.expanduser(fallback)

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

    # last resort: current directory
    rospy.logwarn("%s dir fallback -> (.)", label)
    return "."


def _words_from_vocab_yaml(path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    vocab.yaml 想定:
      objects: [...]
      categories: [...]
      names: [...]
      locations:
        rooms: [...]
        placements: [...]
      corrections:
        - from: "foot"
          to: "food"
    などを柔軟に拾う
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

    # corrections: list[{from,to}] or dict
    corr_map: Dict[str, str] = {}
    corr = y.get("corrections")
    if isinstance(corr, dict):
        for k, v in corr.items():
            if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                corr_map[_lower(k)] = _lower(v)
    elif isinstance(corr, list):
        for it in corr:
            if not isinstance(it, dict):
                continue
            f = it.get("from")
            t = it.get("to")
            if isinstance(f, str) and isinstance(t, str) and f.strip() and t.strip():
                corr_map[_lower(f)] = _lower(t)

    words = _unique_keep_order([w for w in words if w])
    return words, corr_map


def _build_hotwords_string(words: List[str], max_terms: int = 250) -> str:
    # faster-whisper の hotwords は “文字列” が安全（内部で .strip() される）
    # multiword もそのまま通す
    w = []
    for s in words:
        s2 = _normalize_ws(s)
        if not s2:
            continue
        # 句読点など最小限除去（任意）
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
    ここは “強くしすぎると事故る” ので最低限。
    """
    def __init__(self, corr_map: Dict[str, str]):
        self.map = corr_map or {}

    def apply(self, text: str) -> str:
        if not text:
            return ""
        s = _lower(text)
        # 単語境界優先で置換
        for f, t in self.map.items():
            if not f or not t:
                continue
            # multiword は単純に包含で
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
        # Recommended: mount host to /data and use /data/models/...
        preferred_cache = rospy.get_param("~model_cache_dir", "/data/models/hf")
        preferred_torch = rospy.get_param("~torch_home", "/data/models/torch")

        fallback_cache = os.path.expanduser("~/.cache/hsr_models/hf")
        fallback_torch = os.path.expanduser("~/.cache/hsr_models/torch")

        self.model_cache_dir = _ensure_writable_dir(preferred_cache, fallback_cache, "hf_cache")
        self.torch_home = _ensure_writable_dir(preferred_torch, fallback_torch, "torch_home")

        # Set env vars so HF + torch.hub reuse caches across runs
        # (Do this BEFORE WhisperModel initialization)
        os.environ["HF_HOME"] = self.model_cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(self.model_cache_dir, "hub")
        os.environ["XDG_CACHE_HOME"] = os.path.join(self.model_cache_dir, "xdg")
        os.environ["TORCH_HOME"] = self.torch_home

        # Ensure subdirs exist
        for d in [os.environ["HUGGINGFACE_HUB_CACHE"], os.environ["XDG_CACHE_HOME"], self.torch_home]:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception as e:
                rospy.logwarn("Could not mkdir %s: %s", d, e)

        rospy.loginfo(
            "HF cache: HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s XDG_CACHE_HOME=%s TORCH_HOME=%s",
            os.environ["HF_HOME"], os.environ["HUGGINGFACE_HUB_CACHE"], os.environ["XDG_CACHE_HOME"], os.environ["TORCH_HOME"]
        )

        # ---- prompts / hotwords ----
        self.use_hotwords = bool(rospy.get_param("~use_hotwords", True))
        self.vocab_yaml = rospy.get_param("~vocab_yaml", "")
        self.max_hotwords_terms = int(rospy.get_param("~max_hotwords_terms", 250))

        self.initial_prompt = rospy.get_param(
            "~initial_prompt",
            "You are a home service robot. Transcribe spoken GPSR commands in English accurately."
        )

        self.publish_confidence = bool(rospy.get_param("~publish_confidence", True))
        self.enable_corrections = bool(rospy.get_param("~enable_corrections", True))

        # ---- vocab -> hotwords + corrections ----
        self.hotwords_str = ""
        self.corr = CorrectionEngine({})
        if self.vocab_yaml and os.path.exists(self.vocab_yaml):
            try:
                words, corr_map = _words_from_vocab_yaml(self.vocab_yaml)
                if self.use_hotwords:
                    self.hotwords_str = _build_hotwords_string(words, max_terms=self.max_hotwords_terms)
                self.corr = CorrectionEngine(corr_map if self.enable_corrections else {})
                rospy.loginfo(
                    "Loaded vocab_yaml: %s (hotwords_terms=%d, corrections=%s)",
                    self.vocab_yaml, len(self.hotwords_str.split()), "on" if self.enable_corrections else "off"
                )
            except Exception as e:
                rospy.logwarn("Failed to load vocab_yaml=%s: %s", self.vocab_yaml, e)
        else:
            if self.vocab_yaml:
                rospy.logwarn("vocab_yaml not found: %s (hotwords disabled)", self.vocab_yaml)

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

        # download_root makes faster-whisper reuse the cached snapshot under model_cache_dir
        self.model = WhisperModel(
            model_id,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_cache_dir,
        )

        # ---- ROS pub/sub ----
        self.pub_raw = rospy.Publisher(self.raw_text_topic, String, queue_size=10)
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_utt_end = rospy.Publisher(self.utt_end_topic, Bool, queue_size=10)

        self.pub_conf = None
        if self.publish_confidence:
            self.pub_conf = rospy.Publisher(self.conf_topic, Float32, queue_size=10)

        rospy.Subscriber(self.audio_topic, AudioData, self._on_audio, queue_size=200)
        rospy.Subscriber(self.vad_topic, Bool, self._on_vad, queue_size=50)

        # ---- audio buffers ----
        self._lock = threading.Lock()
        self._ring = deque()  # list of (t, bytes)
        self._ring_max_sec = max(3.0, self.pre_roll + self.max_segment_sec + self.post_roll + 1.0)

        self._in_speech = False
        self._seg_bytes = bytearray()
        self._seg_start_time = None

        self._pending_finalize = False
        self._finalize_at = 0.0  # wall time

        # timer
        self._timer = rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo(
            "FasterWhisperASRNode started. audio=%s vad=%s -> raw=%s text=%s end=%s",
            self.audio_topic, self.vad_topic, self.raw_text_topic, self.text_topic, self.utt_end_topic
        )

    # ---------------- audio callbacks ----------------
    def _on_audio(self, msg: AudioData):
        b = bytes(msg.data)
        now = time.time()
        with self._lock:
            self._ring.append((now, b))
            cutoff = now - self._ring_max_sec
            while self._ring and self._ring[0][0] < cutoff:
                self._ring.popleft()

            if self._in_speech:
                self._seg_bytes.extend(b)
                if self._seg_start_time is not None and (now - self._seg_start_time) > self.max_segment_sec:
                    rospy.logwarn("ASR: max_segment_sec reached -> force finalize")
                    self._schedule_finalize(post_roll=self.post_roll)

    def _on_vad(self, msg: Bool):
        is_speech = bool(msg.data)
        now = time.time()

        with self._lock:
            if is_speech and not self._in_speech:
                # speech start: add pre-roll from ring
                self._in_speech = True
                self._pending_finalize = False
                self._seg_bytes = bytearray()
                self._seg_start_time = now

                preroll_cut = now - self.pre_roll
                for t, b in self._ring:
                    if t >= preroll_cut:
                        self._seg_bytes.extend(b)

                rospy.loginfo("ASR: VAD ON -> start segment (pre_roll=%.2fs)", self.pre_roll)

            elif (not is_speech) and self._in_speech:
                self._in_speech = False
                rospy.loginfo("ASR: VAD OFF -> finalize pending (post_roll=%.2fs)", self.post_roll)
                self._schedule_finalize(post_roll=self.post_roll)

    def _schedule_finalize(self, post_roll: float):
        self._pending_finalize = True
        self._finalize_at = time.time() + max(0.0, post_roll)

    # ---------------- timer ----------------
    def _timer_cb(self, _evt):
        with self._lock:
            if not self._pending_finalize:
                return
            if time.time() < self._finalize_at:
                return

            end_now = time.time()
            post_cut = end_now - self.post_roll
            post_bytes = bytearray()
            for t, b in self._ring:
                if t >= post_cut:
                    post_bytes.extend(b)

            chunks = bytes(self._seg_bytes) + bytes(post_bytes)
            self._pending_finalize = False
            self._seg_bytes = bytearray()
            self._seg_start_time = None

        if not chunks:
            return

        raw_text, conf = self._transcribe_bytes_s16le(chunks)
        raw_text = (raw_text or "").strip()

        if raw_text:
            # publish raw first
            self.pub_raw.publish(String(data=raw_text))

            # apply corrections -> normalized lower
            text = raw_text
            if self.enable_corrections:
                text2 = self.corr.apply(raw_text)
                if text2:
                    text = text2

            self.pub_text.publish(String(data=text))
            if self.pub_conf is not None and conf is not None:
                self.pub_conf.publish(Float32(data=float(conf)))

            # utterance_end pulse (True)
            self.pub_utt_end.publish(Bool(data=True))

    # ---------------- ASR core ----------------
    def _transcribe_bytes_s16le(self, pcm_bytes: bytes) -> Tuple[str, Optional[float]]:
        if self.sample_width != 2 or self.channels != 1:
            rospy.logwarn("ASR: this node expects S16LE mono (sample_width=2, channels=1)")

        a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if a.size < int(self.sample_rate * 0.15):
            return "", None

        try:
            kwargs = dict(
                language=self.language,
                beam_size=self.beam_size,
                initial_prompt=self.initial_prompt if self.initial_prompt else None,
                vad_filter=False,  # external VAD already used
            )

            # IMPORTANT: hotwords must be a STRING, not list
            if self.use_hotwords and isinstance(self.hotwords_str, str) and self.hotwords_str.strip():
                kwargs["hotwords"] = self.hotwords_str

            segments, _info = self.model.transcribe(a, **kwargs)
            seg_list = list(segments)
            text = "".join([(s.text or "") for s in seg_list]).strip()

            conf = None
            try:
                if seg_list:
                    avg_lp = float(np.mean([s.avg_logprob for s in seg_list if s.avg_logprob is not None]))
                    conf = float(1.0 / (1.0 + np.exp(-avg_lp)))  # sigmoid
            except Exception:
                conf = None

            return text, conf

        except Exception as e:
            rospy.logerr("ASR transcribe error: %s", e)
            return "", None


def main():
    _ = FasterWhisperASRNode()
    rospy.spin()


if __name__ == "__main__":
    main()
