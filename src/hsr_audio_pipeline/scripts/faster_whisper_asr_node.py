#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
faster_whisper_asr_node.py (ROS1 / Noetic)

Sub:
  - ~audio_topic  (audio_common_msgs/AudioData) default: /audio/audio
  - ~vad_topic    (std_msgs/Bool)               default: /vad/is_speech

Pub:
  - ~text_topic           (std_msgs/String) default: /gpsr/asr/text
  - ~utterance_end_topic  (std_msgs/Bool)   default: /gpsr/asr/utterance_end
  - ~confidence_topic     (std_msgs/Float32) default: /gpsr/asr/confidence (optional)

Key features:
  - External VAD drives segmentation
  - post_roll audio appended after speech_end
  - Uses faster-whisper WhisperModel
  - loads GPSR vocab YAML via ~vocab_yaml
  - NEW (cache fix):
      * ~model_cache_dir to persist HF cache
      * sets HF_HOME / HUGGINGFACE_HUB_CACHE / XDG_CACHE_HOME
      * uses WhisperModel(..., download_root=...)
      * optional ~model_path to load local model directly if exists
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


def _ensure_dir(path: str) -> str:
    if not path:
        return path
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        rospy.logwarn("Could not create dir: %s (%s)", path, e)
    return path


def _words_from_vocab_yaml(path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
      words: list[str]  (for hotwords)
      corrections: dict[str,str] (lowercase mapping for post-correction)
    """
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    words: List[str] = []
    corrections: Dict[str, str] = {}

    # names
    for n in y.get("names", []) or []:
        if isinstance(n, str) and n.strip():
            words.append(n.strip())

    # rooms
    for r in y.get("rooms", []) or []:
        if isinstance(r, str) and r.strip():
            words.append(r.strip())
            rr = r.strip()
            if " " in rr:
                corrections[_lower(rr.replace(" ", ""))] = _lower(rr)

    # locations
    for loc in y.get("locations", []) or []:
        if not isinstance(loc, dict) or "name" not in loc:
            continue
        name = str(loc["name"]).strip()
        if not name:
            continue
        words.append(name)
        if " " in name:
            corrections[_lower(name.replace(" ", ""))] = _lower(name)
        if "-" in name:
            corrections[_lower(name.replace("-", " "))] = _lower(name.replace("-", " "))
            corrections[_lower(name.replace("-", ""))] = _lower(name.replace("-", " "))

    # categories + objects
    for c in y.get("categories", []) or []:
        if not isinstance(c, dict):
            continue
        singular = str(c.get("singular", "")).strip()
        plural = str(c.get("plural", "")).strip()
        if singular:
            words.append(singular)
        if plural:
            words.append(plural)
        if singular and plural:
            corrections[_lower(plural)] = _lower(plural)
            corrections[_lower(singular)] = _lower(singular)

        for o in c.get("objects", []) or []:
            if not isinstance(o, str):
                continue
            obj = o.replace("_", " ").strip()
            if not obj:
                continue
            words.append(obj)
            corrections[_lower(o)] = _lower(obj)
            if " " in obj:
                corrections[_lower(obj.replace(" ", ""))] = _lower(obj)

    corrections.setdefault("foot", "food")
    corrections.setdefault("past room", "bathroom")

    words = [w.strip() for w in words if isinstance(w, str) and w.strip()]
    words = _unique_keep_order(words)

    return words, corrections


def _build_hotwords_string(words: List[str], max_terms: int = 250) -> str:
    tmp = {}
    for w in words:
        ww = _normalize_ws(w)
        if not ww:
            continue
        tmp.setdefault(ww.lower(), ww)
    uniq = list(tmp.values())
    uniq.sort(key=lambda s: (-len(s), s.lower()))
    if len(uniq) > max_terms:
        uniq = uniq[:max_terms]
    return " ".join(uniq)


class CorrectionEngine:
    def __init__(self, mapping: Dict[str, str]):
        self.mapping = {_lower(k): _lower(v) for k, v in (mapping or {}).items() if _lower(k) and _lower(v)}
        self.keys = sorted(self.mapping.keys(), key=lambda s: -len(s))
        self.patterns: List[Tuple[re.Pattern, str]] = []
        for k in self.keys:
            pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)")
            self.patterns.append((pat, self.mapping[k]))

    def apply(self, text: str) -> str:
        t = _lower(text)
        if not t:
            return t
        for pat, rep in self.patterns:
            t = pat.sub(rep, t)
        return _normalize_ws(t)


# ----------------------------
# Node
# ----------------------------
class FasterWhisperASRNode:
    def __init__(self):
        rospy.init_node("faster_whisper_asr_node")

        # ---- params ----
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.sample_width = int(rospy.get_param("~sample_width", 2))  # bytes (S16LE)
        self.channels = int(rospy.get_param("~channels", 1))

        # VAD segmentation
        self.pre_roll = float(rospy.get_param("~pre_roll", 0.25))
        self.post_roll = float(rospy.get_param("~post_roll", 0.35))
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 12.0))

        # faster-whisper
        self.model_size = rospy.get_param("~model_size", "small")
        self.device = rospy.get_param("~device", "cpu")
        self.compute_type = rospy.get_param("~compute_type", "float32")
        self.language = rospy.get_param("~language", "en")
        self.beam_size = int(rospy.get_param("~beam_size", 5))

        # NEW: model cache / local path
        # - model_path: if exists, load directly (no download)
        # - model_cache_dir: persistent cache root for HF hub (use Docker volume!)
        self.model_path = rospy.get_param("~model_path", "").strip()
        self.model_cache_dir = rospy.get_param("~model_cache_dir", "/models/hf").strip()
        self.model_cache_dir = _ensure_dir(self.model_cache_dir)

        # Force HF caches to persistent dir (prevents re-download each boot)
        # NOTE: keep consistent across processes/users inside container.
        hf_home = self.model_cache_dir
        hub_cache = os.path.join(self.model_cache_dir, "hub")
        xdg_cache = os.path.join(self.model_cache_dir, "xdg")
        _ensure_dir(hub_cache)
        _ensure_dir(xdg_cache)

        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hub_cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(self.model_cache_dir, "transformers"))
        os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)

        # prompts / hotwords
        self.use_hotwords = bool(rospy.get_param("~use_hotwords", True))
        self.vocab_yaml = rospy.get_param("~vocab_yaml", "")
        self.max_hotwords_terms = int(rospy.get_param("~max_hotwords_terms", 250))

        self.initial_prompt = rospy.get_param(
            "~initial_prompt",
            "You are a home service robot. Transcribe spoken GPSR commands in English accurately."
        )

        self.publish_confidence = bool(rospy.get_param("~publish_confidence", True))

        # ---- vocab -> hotwords + corrections ----
        self.hotwords_str = ""
        self.corr = CorrectionEngine({})
        if self.vocab_yaml and os.path.exists(self.vocab_yaml):
            try:
                words, corr_map = _words_from_vocab_yaml(self.vocab_yaml)
                if self.use_hotwords:
                    self.hotwords_str = _build_hotwords_string(words, max_terms=self.max_hotwords_terms)
                self.corr = CorrectionEngine(corr_map)
                rospy.loginfo("Loaded vocab_yaml: %s (hotwords_terms=%d)", self.vocab_yaml, len(self.hotwords_str.split()))
            except Exception as e:
                rospy.logwarn("Failed to load vocab_yaml=%s: %s", self.vocab_yaml, e)
        else:
            if self.vocab_yaml:
                rospy.logwarn("vocab_yaml not found: %s (hotwords disabled)", self.vocab_yaml)

        # ---- model ----
        rospy.loginfo("HF cache: HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s XDG_CACHE_HOME=%s",
                      os.environ.get("HF_HOME"), os.environ.get("HUGGINGFACE_HUB_CACHE"), os.environ.get("XDG_CACHE_HOME"))

        model_id = self.model_path if (self.model_path and os.path.exists(self.model_path)) else self.model_size
        if self.model_path and not os.path.exists(self.model_path):
            rospy.logwarn("model_path set but not found: %s (fallback to model_size=%s)", self.model_path, self.model_size)

        rospy.loginfo("Loading WhisperModel: model=%s device=%s compute_type=%s download_root=%s",
                      model_id, self.device, self.compute_type, self.model_cache_dir)

        # download_root forces faster-whisper to place downloaded snapshots under this directory.
        # If already downloaded, it will reuse existing files (no re-download).
        self.model = WhisperModel(
            model_id,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_cache_dir,
        )

        # ---- ROS pub/sub ----
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

        self._timer = rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo("FasterWhisperASRNode started. audio=%s vad=%s -> text=%s utt_end=%s",
                      self.audio_topic, self.vad_topic, self.text_topic, self.utt_end_topic)

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
        text, conf = self._transcribe_bytes_s16le(chunks)

        if text:
            text2 = self.corr.apply(text)
            if text2:
                text = text2

            self.pub_text.publish(String(data=text))
            if self.pub_conf is not None and conf is not None:
                self.pub_conf.publish(Float32(data=float(conf)))

            self.pub_utt_end.publish(Bool(data=True))

    # ---------------- ASR core ----------------
    def _transcribe_bytes_s16le(self, pcm_bytes: bytes) -> Tuple[str, Optional[float]]:
        if self.sample_width != 2 or self.channels != 1:
            rospy.logwarn("ASR: only supports S16LE mono in this node (sample_width=2, channels=1)")
        a = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if a.size < int(self.sample_rate * 0.15):
            return "", None

        try:
            kwargs = dict(
                language=self.language,
                beam_size=self.beam_size,
                initial_prompt=self.initial_prompt if self.initial_prompt else None,
                vad_filter=False,
            )

            if self.use_hotwords and isinstance(self.hotwords_str, str) and self.hotwords_str.strip():
                kwargs["hotwords"] = self.hotwords_str

            segments, _info = self.model.transcribe(a, **kwargs)

            seg_list = list(segments)
            text = "".join([(s.text or "") for s in seg_list]).strip()

            conf = None
            try:
                if seg_list:
                    avg_lp = float(np.mean([s.avg_logprob for s in seg_list if s.avg_logprob is not None]))
                    conf = float(1.0 / (1.0 + np.exp(-avg_lp)))
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
