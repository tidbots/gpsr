#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 Noetic
faster_whisper_asr_node.py

- External VAD (/vad/is_speech)
- Audio input: audio_common_msgs/AudioData (/audio/audio)
- Publish:
    /gpsr/asr/raw_text
    /gpsr/asr/text
    /gpsr/asr/utterance_end
    /gpsr/asr/confidence
- MD vocab (/data/vocab/*.md) preferred for hotwords
- YAML vocab fallback
- corrections.yaml supported
- **IMPORTANT**: rospy.spin() is guaranteed (no clean exit)
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


# =========================
# utility
# =========================
def _norm(s: str) -> str:
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
        w = _norm(w)
        w = re.sub(r"[^\w\s\-]", " ", w)
        w = _norm(w)
        if len(w) >= 2:
            out.append(w)
    out = list(dict.fromkeys(out))
    return " ".join(out[:max_terms])


# =========================
# vocab loaders
# =========================
def load_words_from_md(vocab_dir: str) -> List[str]:
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

    return list(dict.fromkeys(words))


def load_words_from_yaml(path: str) -> Tuple[List[str], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    words = []
    corr = {}

    for k in ("names", "objects", "categories"):
        for w in y.get(k, []) or []:
            if isinstance(w, str):
                words.append(w)

    for loc in y.get("locations", []) or []:
        if isinstance(loc, dict) and "name" in loc:
            words.append(loc["name"])

    for k, v in (y.get("corrections") or {}).items():
        corr[k.lower()] = v.lower()

    return list(dict.fromkeys(words)), corr


class CorrectionEngine:
    def __init__(self, corr: Dict[str, str]):
        self.corr = corr or {}

    def apply(self, s: str) -> str:
        t = s.lower()
        for k, v in self.corr.items():
            t = re.sub(rf"\b{re.escape(k)}\b", v, t)
        return _norm(t)


# =========================
# ASR Node
# =========================
class FasterWhisperASRNode:
    def __init__(self):
        rospy.loginfo("Initializing faster_whisper_asr_node")

        # ---- params ----
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.raw_text_topic = rospy.get_param("~raw_text_topic", "/gpsr/asr/raw_text")
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.pre_roll = float(rospy.get_param("~pre_roll", 0.25))
        self.post_roll = float(rospy.get_param("~post_roll", 0.35))
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 18.0))

        self.device = rospy.get_param("~device", "cpu")
        self.compute_type = rospy.get_param("~compute_type", "float32")
        self.model_size = rospy.get_param("~model_size", "small")
        self.language = rospy.get_param("~language", "en")

        self.vocab_dir = rospy.get_param("~vocab_dir", "/data/vocab")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", os.path.join(self.vocab_dir, "vocab.yaml"))
        self.corrections_yaml = rospy.get_param("~corrections_yaml", os.path.join(self.vocab_dir, "corrections.yaml"))

        self.use_hotwords = bool(rospy.get_param("~use_hotwords", True))
        self.max_hotwords_terms = int(rospy.get_param("~max_hotwords_terms", 250))

        self.initial_prompt = rospy.get_param(
            "~initial_prompt",
            "You are a RoboCup@Home service robot. Transcribe GPSR commands in English.",
        )

        # ---- publishers ----
        self.pub_raw = rospy.Publisher(self.raw_text_topic, String, queue_size=10)
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_end = rospy.Publisher(self.utt_end_topic, Bool, queue_size=10)
        self.pub_conf = rospy.Publisher(self.conf_topic, Float32, queue_size=10)

        # ---- vocab / hotwords ----
        words = []
        corr = {}

        try:
            words = load_words_from_md(self.vocab_dir)
            rospy.loginfo("ASR: loaded MD vocab (%d words)", len(words))
        except Exception as e:
            rospy.logwarn("ASR: MD vocab failed (%s), fallback YAML", e)

        if not words and os.path.exists(self.vocab_yaml):
            words, corr = load_words_from_yaml(self.vocab_yaml)
            rospy.loginfo("ASR: loaded YAML vocab (%d words)", len(words))

        self.hotwords = _build_hotwords(words, self.max_hotwords_terms) if self.use_hotwords else ""
        rospy.loginfo("ASR hotwords terms=%d", len(self.hotwords.split()))

        if os.path.exists(self.corrections_yaml):
            with open(self.corrections_yaml, "r", encoding="utf-8") as f:
                corr.update(yaml.safe_load(f) or {})

        self.corrector = CorrectionEngine(corr)

        # ---- model ----
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        # ---- buffers ----
        self._lock = threading.Lock()
        self._ring = deque()
        self._ring_max_sec = self.pre_roll + self.max_segment_sec + self.post_roll + 1.0

        self._in_speech = False
        self._seg_bytes = bytearray()
        self._seg_start = None
        self._pending_finalize = False
        self._finalize_at = 0.0

        rospy.Subscriber(self.audio_topic, AudioData, self._on_audio, queue_size=200)
        rospy.Subscriber(self.vad_topic, Bool, self._on_vad, queue_size=50)

        rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo("faster_whisper_asr_node READY")

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
                if self._seg_start and now - self._seg_start > self.max_segment_sec:
                    self._schedule_finalize()

    def _on_vad(self, msg: Bool):
        now = time.time()
        with self._lock:
            if msg.data and not self._in_speech:
                self._in_speech = True
                self._seg_bytes.clear()
                self._seg_start = now
                for t, b in self._ring:
                    if t >= now - self.pre_roll:
                        self._seg_bytes.extend(b)

            elif not msg.data and self._in_speech:
                self._in_speech = False
                self._schedule_finalize()

    def _schedule_finalize(self):
        self._pending_finalize = True
        self._finalize_at = time.time() + self.post_roll

    def _timer_cb(self, _):
        with self._lock:
            if not self._pending_finalize or time.time() < self._finalize_at:
                return
            pcm = bytes(self._seg_bytes)
            self._pending_finalize = False
            self._seg_bytes.clear()

        if not pcm:
            return

        text, conf = self._transcribe(pcm)
        if not text:
            return

        self.pub_raw.publish(String(text))
        norm = self.corrector.apply(text)
        self.pub_text.publish(String(norm))
        self.pub_conf.publish(Float32(conf if conf is not None else 0.0))
        self.pub_end.publish(Bool(True))

    # ---------------- ASR core ----------------
    def _transcribe(self, pcm_bytes: bytes) -> Tuple[str, Optional[float]]:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < int(self.sample_rate * 0.2):
            return "", None

        kwargs = dict(
            language=self.language,
            initial_prompt=self.initial_prompt,
            vad_filter=False,
        )
        if self.hotwords:
            kwargs["hotwords"] = self.hotwords

        segments, _ = self.model.transcribe(audio, **kwargs)
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()

        conf = None
        if segs:
            conf = float(np.mean([s.avg_logprob for s in segs if s.avg_logprob is not None]))

        return text, conf


# =========================
# main (ABSOLUTELY REQUIRED)
# =========================
def main():
    rospy.init_node("faster_whisper_asr_node")
    node = FasterWhisperASRNode()
    rospy.loginfo("faster_whisper_asr_node spinning")
    rospy.spin()


if __name__ == "__main__":
    main()
