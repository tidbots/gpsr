#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 Noetic
faster_whisper_asr_node.py (GPSR ASR with VAD stabilization)

Inputs
------
- /audio (audio_common_msgs/AudioData)   [default: ~audio_topic:=/audio]
- /vad/is_speech (std_msgs/Bool)         [default: ~vad_topic:=/vad/is_speech]

Outputs
-------
- /gpsr/asr/raw_text        (std_msgs/String)  Whisper raw
- /gpsr/asr/text            (std_msgs/String)  corrected + optional projection
- /gpsr/asr/utterance_end   (std_msgs/Bool)    True per finalized utterance
- /gpsr/asr/confidence      (std_msgs/Float32) avg_logprob mean (can be negative)
- /gpsr/asr/debug           (std_msgs/String)  JSON for debugging

Key features for RoboCup GPSR
-----------------------------
- External VAD based segmentation (Silero VAD)
- Pre-roll / post-roll audio padding
- Hard cap max_segment_sec with *forced finalize*
- Finalize debounce (cooldown) to avoid duplicate triggers
- Minimum speech length gate to ignore accidental clicks
- Hotwords from /data/vocab/*.md or vocab.yaml
- corrections.yaml
- Projection to official commands list (/data/vocab/commands.txt)
"""

import os
import sys
import re
import time
import yaml
import json
import threading
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import rospy

from std_msgs.msg import String, Bool, Float32
from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel

# === Make sure scripts/ is importable (catkin wrapper exec() case) ===
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# === GPSR PROJECTION ===
from gpsr_projector import GpsrProjector


# =========================
# utility
# =========================
def _norm_ws(s: str) -> str:
    return " ".join((s or "").strip().split())


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _build_hotwords(words: List[str], max_terms: int) -> str:
    out = []
    for w in words:
        w = _norm_ws(w)
        w = re.sub(r"[^\w\s\-]", " ", w)
        w = _norm_ws(w)
        if len(w) >= 2:
            out.append(w)
    out = list(dict.fromkeys(out))
    return " ".join(out[:max_terms])


def _read_lines(path: str) -> List[str]:
    txt = _safe_read(path)
    if not txt:
        return []
    lines = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # allow "1. command" formats too
        ln = re.sub(r"^\s*\d+[\.\)]\s*", "", ln).strip()
        if ln:
            lines.append(ln)
    return lines


# =========================
# vocab loaders
# =========================
def load_words_from_md(vocab_dir: str) -> Tuple[List[str], Dict[str, List[str]]]:
    names = re.findall(r"\|\s*([A-Za-z]+)\s*\|", _safe_read(os.path.join(vocab_dir, "names.md")))
    rooms = re.findall(r"\|\s*(\w+ \w*)\s*\|", _safe_read(os.path.join(vocab_dir, "room_names.md")))

    loc_pairs = re.findall(
        r"\|\s*([0-9]+)\s*\|\s*([A-Za-z,\s\(\)]+)\|",
        _safe_read(os.path.join(vocab_dir, "location_names.md")),
    )
    locs = [b.replace("(p)", "").strip() for _, b in loc_pairs]

    md_obj = _safe_read(os.path.join(vocab_dir, "objects.md"))
    objs = re.findall(r"\|\s*(\w+)\s*\|", md_obj)
    objs = [o.replace("_", " ") for o in objs if o != "Objectname"]

    cats = re.findall(r"# Class \s*([\w,\s\(\)]+)", md_obj)
    for c in cats:
        for p in c.replace("(", "").replace(")", "").split():
            objs.append(p.replace("_", " "))

    words = []
    for src in (names, rooms, locs, objs):
        words += [w.strip() for w in src if w.strip()]

    vocab_dict = {
        "names": list(dict.fromkeys([w.strip() for w in names if w.strip()])),
        "locations": list(dict.fromkeys([w.strip() for w in (rooms + locs) if w.strip()])),
        "objects": list(dict.fromkeys([w.strip() for w in objs if w.strip()])),
        "object_categories": [],  # optional
    }
    return list(dict.fromkeys(words)), vocab_dict


def load_words_from_yaml(path: str) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    words: List[str] = []
    corr: Dict[str, str] = {}

    names: List[str] = []
    objects: List[str] = []
    categories: List[str] = []
    locations: List[str] = []

    for k in ("names", "objects", "categories"):
        for w in y.get(k, []) or []:
            if isinstance(w, str):
                words.append(w)
                if k == "names":
                    names.append(w)
                elif k == "objects":
                    objects.append(w)
                elif k == "categories":
                    categories.append(w)

    for loc in y.get("locations", []) or []:
        if isinstance(loc, dict) and "name" in loc:
            words.append(loc["name"])
            locations.append(loc["name"])
        elif isinstance(loc, str):
            words.append(loc)
            locations.append(loc)

    for k, v in (y.get("corrections") or {}).items():
        corr[str(k).lower()] = str(v).lower()

    vocab_dict = {
        "names": list(dict.fromkeys(names)),
        "locations": list(dict.fromkeys(locations)),
        "objects": list(dict.fromkeys(objects)),
        "object_categories": list(dict.fromkeys(categories)),
    }
    return list(dict.fromkeys(words)), corr, vocab_dict


class CorrectionEngine:
    def __init__(self, corr: Dict[str, str]):
        self.corr = corr or {}

    def apply(self, s: str) -> str:
        t = (s or "").lower()
        for k, v in self.corr.items():
            t = re.sub(rf"\b{re.escape(str(k))}\b", str(v), t)
        return _norm_ws(t)


# =========================
# ASR Node
# =========================
class FasterWhisperASRNode:
    def __init__(self):
        rospy.loginfo("Initializing faster_whisper_asr_node")

        # ---- topics ----
        self.audio_topic = rospy.get_param("~audio_topic", "/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.raw_text_topic = rospy.get_param("~raw_text_topic", "/gpsr/asr/raw_text")
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.debug_topic = rospy.get_param("~debug_topic", "/gpsr/asr/debug")

        # ---- audio segmentation ----
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.pre_roll = float(rospy.get_param("~pre_roll", 0.35))
        self.post_roll = float(rospy.get_param("~post_roll", 0.45))
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 22.0))

        # VAD stabilization (ASR-side)
        self.min_speech_sec = float(rospy.get_param("~min_speech_sec", 0.45))  # ignore micro-triggers
        self.finalize_cooldown_sec = float(rospy.get_param("~finalize_cooldown_sec", 0.8))  # debounce finalize
        self._last_finalize_time = 0.0

        # ---- model ----
        self.device = rospy.get_param("~device", "cpu")
        self.compute_type = rospy.get_param("~compute_type", "float32")
        self.model_size = rospy.get_param("~model_size", "small")
        self.language = rospy.get_param("~language", "en")

        # ---- vocab paths ----
        self.vocab_dir = rospy.get_param("~vocab_dir", "/data/vocab")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", os.path.join(self.vocab_dir, "vocab.yaml"))
        self.corrections_yaml = rospy.get_param("~corrections_yaml", os.path.join(self.vocab_dir, "corrections.yaml"))

        self.use_hotwords = bool(rospy.get_param("~use_hotwords", True))
        self.max_hotwords_terms = int(rospy.get_param("~max_hotwords_terms", 250))

        self.initial_prompt = rospy.get_param(
            "~initial_prompt",
            "Transcribe ONLY RoboCup@Home GPSR commands in English. Output only the command sentence.",
        )

        # ---- projection (official commands list) ----
        self.enable_projection = bool(rospy.get_param("~enable_projection", True))
        self.projection_threshold = float(rospy.get_param("~projection_threshold", 0.72))
        self.commands_path = rospy.get_param("~commands_path", os.path.join(self.vocab_dir, "commands.txt"))
        self.extra_commands_paths = rospy.get_param("~extra_commands_paths", "")  # comma-separated

        # ---- publishers ----
        self.pub_raw = rospy.Publisher(self.raw_text_topic, String, queue_size=10)
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_end = rospy.Publisher(self.utt_end_topic, Bool, queue_size=10)
        self.pub_conf = rospy.Publisher(self.conf_topic, Float32, queue_size=10)
        self.pub_debug = rospy.Publisher(self.debug_topic, String, queue_size=10)

        # ---- vocab / corrections / hotwords ----
        words: List[str] = []
        corr: Dict[str, str] = {}
        vocab_dict: Dict[str, List[str]] = {"names": [], "locations": [], "objects": [], "object_categories": []}

        try:
            words, vocab_dict = load_words_from_md(self.vocab_dir)
            rospy.loginfo("ASR: loaded MD vocab (%d words)", len(words))
        except Exception as e:
            rospy.logwarn("ASR: MD vocab failed (%s), fallback YAML", e)

        if not words and os.path.exists(self.vocab_yaml):
            words, corr, vocab_dict = load_words_from_yaml(self.vocab_yaml)
            rospy.loginfo("ASR: loaded YAML vocab (%d words)", len(words))

        self.hotwords = _build_hotwords(words, self.max_hotwords_terms) if self.use_hotwords else ""
        rospy.loginfo("ASR hotwords terms=%d", len(self.hotwords.split()))

        if os.path.exists(self.corrections_yaml):
            with open(self.corrections_yaml, "r", encoding="utf-8") as f:
                c2 = yaml.safe_load(f) or {}
                for k, v in (c2 or {}).items():
                    corr[str(k).lower()] = str(v).lower()

        self.corrector = CorrectionEngine(corr)

        # ---- projection: load official command list ----
        self.projector: Optional[GpsrProjector] = None
        if self.enable_projection:
            cmd_list: List[str] = []
            cmd_list += _read_lines(self.commands_path)
            extras = [p.strip() for p in str(self.extra_commands_paths).split(",") if p.strip()]
            for p in extras:
                cmd_list += _read_lines(p)

            cmd_list = list(dict.fromkeys([c.strip() for c in cmd_list if c.strip()]))

            if cmd_list:
                rospy.loginfo("ASR: loaded official commands for projection: %d lines", len(cmd_list))
                self.projector = GpsrProjector(cmd_list, vocab=vocab_dict)
            else:
                rospy.logwarn("ASR: projection enabled but command list is empty. Set ~commands_path.")
                self.enable_projection = False

        # ---- whisper model ----
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        # ---- buffers ----
        self._lock = threading.Lock()
        self._ring = deque()
        self._ring_max_sec = self.pre_roll + self.max_segment_sec + self.post_roll + 1.0

        self._in_speech = False
        self._seg_bytes = bytearray()
        self._seg_start: Optional[float] = None
        self._pending_finalize = False
        self._finalize_at = 0.0
        self._finalize_reason = ""

        # ---- subs/timer ----
        rospy.Subscriber(self.audio_topic, AudioData, self._on_audio, queue_size=200)
        rospy.Subscriber(self.vad_topic, Bool, self._on_vad, queue_size=50)
        rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo(
            "faster_whisper_asr_node READY (audio=%s vad=%s pre=%.2f post=%.2f max=%.1f min_speech=%.2f cooldown=%.2f)",
            self.audio_topic, self.vad_topic, self.pre_roll, self.post_roll, self.max_segment_sec,
            self.min_speech_sec, self.finalize_cooldown_sec
        )

    # ---------------- callbacks ----------------
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

                # Hard cap: force finalize if too long (prevents "never ending True")
                if self._seg_start is not None and (now - self._seg_start) > self.max_segment_sec:
                    # Force a finalize regardless of VAD. Keep post_roll tail.
                    self._in_speech = False
                    self._schedule_finalize(reason="max_segment_sec")

    def _on_vad(self, msg: Bool):
        now = time.time()
        with self._lock:
            if msg.data and not self._in_speech:
                # start speech
                self._in_speech = True
                self._seg_bytes.clear()
                self._seg_start = now
                self._pending_finalize = False

                # prepend pre-roll from ring
                for t, b in self._ring:
                    if t >= now - self.pre_roll:
                        self._seg_bytes.extend(b)

            elif (not msg.data) and self._in_speech:
                # end speech -> schedule finalize (with post-roll)
                self._in_speech = False
                self._schedule_finalize(reason="vad_fall")

    def _schedule_finalize(self, reason: str):
        self._pending_finalize = True
        self._finalize_reason = reason
        self._finalize_at = time.time() + self.post_roll

    def _timer_cb(self, _):
        # Wait until it's time to finalize
        with self._lock:
            if (not self._pending_finalize) or (time.time() < self._finalize_at):
                return

            # Debounce finalize (ASR-side safety)
            now = time.time()
            if now - self._last_finalize_time < self.finalize_cooldown_sec:
                # Drop this finalize to prevent duplicate triggers
                self._pending_finalize = False
                self._seg_bytes.clear()
                self._seg_start = None
                return

            # Minimum speech length gate
            if self._seg_start is not None:
                speech_len = now - self._seg_start
                if speech_len < self.min_speech_sec and self._finalize_reason != "max_segment_sec":
                    # Too short; ignore as noise
                    self._pending_finalize = False
                    self._seg_bytes.clear()
                    self._seg_start = None
                    return

            pcm = bytes(self._seg_bytes)
            reason = self._finalize_reason
            self._pending_finalize = False
            self._seg_bytes.clear()
            self._seg_start = None

            # Record finalize time NOW (debounce)
            self._last_finalize_time = now

        if not pcm:
            return

        text, conf = self._transcribe(pcm)
        if not text:
            return

        # raw
        self.pub_raw.publish(String(text))

        # corrections
        norm = self.corrector.apply(text)

        # projection
        final = norm
        proj_score = None
        proj_target = None
        proj_slots = None
        projected = False

        if self.enable_projection and self.projector is not None:
            res = self.projector.project(norm)
            if res is not None:
                proj_score = float(res.score)
                proj_target = res.projected_text
                proj_slots = res.slots
                if res.score >= self.projection_threshold:
                    final = res.projected_text
                    projected = True

        # Publish final text + utterance end
        self.pub_text.publish(String(final))
        self.pub_conf.publish(Float32(conf if conf is not None else 0.0))
        self.pub_end.publish(Bool(True))

        # debug
        self.pub_debug.publish(String(json.dumps({
            "raw": text,
            "norm": norm,
            "final": final,
            "projected": projected,
            "projection_threshold": self.projection_threshold,
            "projection_score": proj_score,
            "projection_target": proj_target,
            "slots": proj_slots,
            "commands_path": self.commands_path,
            "finalize_reason": reason,
            "cooldown_sec": self.finalize_cooldown_sec,
            "min_speech_sec": self.min_speech_sec,
            "pre_roll": self.pre_roll,
            "post_roll": self.post_roll,
            "max_segment_sec": self.max_segment_sec,
        }, ensure_ascii=False)))

    # ---------------- ASR core ----------------
    def _transcribe(self, pcm_bytes: bytes) -> Tuple[str, Optional[float]]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # too short -> ignore
        if len(audio) < int(self.sample_rate * 0.2):
            return "", None

        kwargs = dict(
            language=self.language,
            initial_prompt=self.initial_prompt,
            vad_filter=False,  # external VAD used
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=True,
        )
        if self.hotwords:
            kwargs["hotwords"] = self.hotwords

        segments, _ = self.model.transcribe(audio, **kwargs)
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()

        conf = None
        if segs:
            vals = [s.avg_logprob for s in segs if s.avg_logprob is not None]
            if vals:
                conf = float(np.mean(vals))

        return text, conf


# =========================
# main (ABSOLUTELY REQUIRED)
# =========================
def main():
    rospy.init_node("faster_whisper_asr_node")
    _ = FasterWhisperASRNode()
    rospy.loginfo("faster_whisper_asr_node spinning")
    rospy.spin()


if __name__ == "__main__":
    main()
