#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 (Noetic) Faster-Whisper ASR node with external VAD gating.

- Subscribe:
  - ~audio_topic (audio_common_msgs/AudioData)  default: /audio/audio
  - ~vad_topic   (std_msgs/Bool)               default: /vad/is_speech
- Publish:
  - ~text_topic  (std_msgs/String)             default: /asr/text

Behavior:
  - When VAD becomes True: start accumulating audio into a segment
  - While VAD True: keep buffering audio (and optionally post_roll)
  - When VAD becomes False: wait post_roll seconds then finalize segment and transcribe
  - Fix: hotwords param may be list/tuple or string; normalize to string for faster-whisper

Notes:
  - Assumes incoming AudioData is PCM bytes (commonly S16LE) with sample rate provided in ~sample_rate.
  - If your audio_capture is not S16LE PCM, you must ensure it is (e.g., audio_capture dst=appsink, sample_format=S16LE).
"""

import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple, Union, List

import numpy as np
import rospy
from std_msgs.msg import String, Bool
from audio_common_msgs.msg import AudioData

# faster-whisper
from faster_whisper import WhisperModel


def pcm16le_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert little-endian signed 16-bit PCM bytes -> float32 [-1, 1]."""
    if not pcm_bytes:
        return np.zeros((0,), dtype=np.float32)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if pcm.size == 0:
        return np.zeros((0,), dtype=np.float32)
    f32 = pcm.astype(np.float32) / 32768.0
    return f32


def compute_rms_peak_int16(pcm_bytes: bytes) -> Tuple[float, int, int]:
    """Return (rms, peak, n_samples) computed from int16 PCM bytes."""
    if not pcm_bytes:
        return 0.0, 0, 0
    x = np.frombuffer(pcm_bytes, dtype=np.int16)
    if x.size == 0:
        return 0.0, 0, 0
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
    peak = int(np.max(np.abs(x)))
    return rms, peak, int(x.size)


class FasterWhisperASRNode:
    def __init__(self):
        rospy.init_node("faster_whisper_asr_node")

        # ---- Params ----
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")
        self.text_topic = rospy.get_param("~text_topic", "/asr/text")

        # Audio format assumptions
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))

        # Segmenting
        self.pre_roll_sec = float(rospy.get_param("~pre_roll_sec", 0.20))
        self.post_roll_sec = float(rospy.get_param("~post_roll_sec", 0.35))
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 20.0))

        # Transcribe options
        self.device = rospy.get_param("~device", "auto")  # "cuda", "cpu", "auto"
        self.compute_type = rospy.get_param("~compute_type", "auto")  # "int8_float16", "float16", "int8", ...
        self.model_size = rospy.get_param("~model_size", "small")  # "base", "small", "medium", "large-v3", ...
        self.language = rospy.get_param("~language", "en")  # "en" or None/"" for auto
        self.task = rospy.get_param("~task", "transcribe")  # "transcribe" or "translate"

        self.beam_size = int(rospy.get_param("~beam_size", 5))
        self.best_of = int(rospy.get_param("~best_of", 5))
        self.temperature = float(rospy.get_param("~temperature", 0.0))
        self.no_speech_threshold = float(rospy.get_param("~no_speech_threshold", 0.6))
        self.log_prob_threshold = float(rospy.get_param("~log_prob_threshold", -1.0))
        self.compression_ratio_threshold = float(rospy.get_param("~compression_ratio_threshold", 2.4))
        self.condition_on_previous_text = bool(rospy.get_param("~condition_on_previous_text", False))

        # Hotwords (can be string or list in YAML)
        self.hotwords = rospy.get_param("~hotwords", None)

        # Logging audio stats
        self.log_audio_stats = bool(rospy.get_param("~log_audio_stats", True))
        self.audio_stats_interval_sec = float(rospy.get_param("~audio_stats_interval_sec", 0.5))

        # Publish empty if short / too quiet?
        self.min_segment_sec = float(rospy.get_param("~min_segment_sec", 0.30))

        # ---- State ----
        self._lock = threading.RLock()
        self._is_speech = False

        # ring buffer for pre-roll
        self._pre_roll_buf: Deque[bytes] = deque()
        self._pre_roll_samples = int(self.pre_roll_sec * self.sample_rate)

        # current segment buffer (bytes chunks)
        self._segment_chunks: List[bytes] = []
        self._segment_samples: int = 0
        self._segment_start_time: Optional[float] = None

        # post-roll finalization pending
        self._pending_finalize = False
        self._pending_finalize_time: Optional[float] = None

        # Audio stats timer
        self._last_stats_time = 0.0

        # ---- Model ----
        rospy.loginfo(f"Loading WhisperModel: size={self.model_size} device={self.device} compute_type={self.compute_type}")
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        # ---- ROS I/O ----
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)

        self.sub_audio = rospy.Subscriber(self.audio_topic, AudioData, self._on_audio, queue_size=50)
        self.sub_vad = rospy.Subscriber(self.vad_topic, Bool, self._on_vad, queue_size=50)

        # Timer to check finalize pending
        self.timer = rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo("FasterWhisperASRNode ready.")
        rospy.loginfo(f" audio_topic={self.audio_topic}")
        rospy.loginfo(f" vad_topic={self.vad_topic}")
        rospy.loginfo(f" text_topic={self.text_topic}")

    # ---------- Hotwords fix ----------
    @staticmethod
    def _normalize_hotwords(hotwords: Union[None, str, List, Tuple]) -> Optional[str]:
        """
        faster-whisper expects hotwords as a string (it calls .strip()).
        ROS yaml often turns it into list. Normalize robustly.
        """
        if hotwords is None:
            return None

        # If it's already bytes -> decode
        if isinstance(hotwords, (bytes, bytearray)):
            hotwords = hotwords.decode("utf-8", errors="ignore")

        if isinstance(hotwords, (list, tuple)):
            tokens = []
            for x in hotwords:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    tokens.append(s)
            s_join = " ".join(tokens).strip()
            return s_join if s_join else None

        # otherwise treat as string
        s = str(hotwords).strip()
        return s if s else None

    # ---------- Callbacks ----------
    def _on_vad(self, msg: Bool):
        now = time.time()
        with self._lock:
            if msg.data and (not self._is_speech):
                self._is_speech = True
                self._pending_finalize = False
                self._pending_finalize_time = None
                rospy.loginfo("SileroVADNode: speech start (prob=unknown)")  # probは別トピックなら拡張

                # start new segment: include pre-roll
                self._start_segment_locked(now)

            elif (not msg.data) and self._is_speech:
                self._is_speech = False
                rospy.loginfo("SileroVADNode: speech end (prob=unknown)")

                # schedule finalize after post-roll
                self._pending_finalize = True
                self._pending_finalize_time = now + self.post_roll_sec

    def _on_audio(self, msg: AudioData):
        pcm_bytes = msg.data
        if not pcm_bytes:
            return

        now = time.time()

        # Optional audio stats logging (RMS/PEAK)
        if self.log_audio_stats:
            if now - self._last_stats_time >= self.audio_stats_interval_sec:
                rms, peak, n = compute_rms_peak_int16(pcm_bytes)
                rospy.loginfo(f"Audio RMS: {rms:.2f}, PEAK: {peak} (n={n})")
                self._last_stats_time = now

        with self._lock:
            # Update pre-roll ring buffer always
            self._push_pre_roll_locked(pcm_bytes)

            # If currently in speech, append to segment
            if self._is_speech:
                self._append_segment_locked(pcm_bytes, now)

                # If segment too long, force finalize
                if self._segment_start_time is not None:
                    dur = now - self._segment_start_time
                    if dur >= self.max_segment_sec:
                        rospy.logwarn(f"ASR: max_segment_sec reached ({dur:.2f}s). Forcing finalize.")
                        self._pending_finalize = True
                        self._pending_finalize_time = now  # finalize asap

            # If pending finalize (post-roll), also keep collecting audio until finalize time
            if self._pending_finalize and self._segment_start_time is not None:
                # keep collecting post-roll audio bytes
                self._append_segment_locked(pcm_bytes, now)

    def _timer_cb(self, _evt):
        with self._lock:
            if not self._pending_finalize:
                return
            if self._pending_finalize_time is None:
                return
            if time.time() < self._pending_finalize_time:
                return

            # finalize
            chunks = list(self._segment_chunks)
            samples = int(self._segment_samples)
            self._pending_finalize = False
            self._pending_finalize_time = None

            # reset segment state before transcribe (avoid re-entrancy)
            self._segment_chunks = []
            self._segment_samples = 0
            self._segment_start_time = None

        # Transcribe outside lock
        self._finalize_segment(chunks, samples)

    # ---------- Segment management ----------
    def _push_pre_roll_locked(self, pcm_bytes: bytes):
        # Keep a ring buffer by samples
        self._pre_roll_buf.append(pcm_bytes)

        # crude accounting: each sample is 2 bytes * channels (assume S16LE)
        bytes_per_sample = 2 * max(1, self.channels)
        max_bytes = self._pre_roll_samples * bytes_per_sample

        total = 0
        # drop from left until within limit
        for b in reversed(self._pre_roll_buf):
            total += len(b)
            if total > max_bytes:
                break
        # Now total is bytes from tail; remove older ones if buffer too big
        # Simplify by popping until size approx
        while True:
            cur_total = sum(len(x) for x in self._pre_roll_buf)
            if cur_total <= max_bytes:
                break
            if len(self._pre_roll_buf) <= 1:
                break
            self._pre_roll_buf.popleft()

    def _start_segment_locked(self, now: float):
        rospy.loginfo("ASR: VAD ON -> start segment")
        self._segment_chunks = list(self._pre_roll_buf)  # include pre-roll
        self._segment_samples = self._estimate_samples(self._segment_chunks)
        self._segment_start_time = now

    def _append_segment_locked(self, pcm_bytes: bytes, now: float):
        if self._segment_start_time is None:
            # VAD may be noisy; start a segment anyway if audio arrives during speech
            self._start_segment_locked(now)
        self._segment_chunks.append(pcm_bytes)
        self._segment_samples += self._estimate_samples([pcm_bytes])

    def _estimate_samples(self, chunks: List[bytes]) -> int:
        if not chunks:
            return 0
        bytes_per_sample = 2 * max(1, self.channels)  # S16LE
        total_bytes = sum(len(c) for c in chunks)
        return int(total_bytes // bytes_per_sample)

    # ---------- Transcription ----------
    def _finalize_segment(self, chunks: List[bytes], samples: int):
        sec = samples / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        rospy.loginfo(f"ASR: VAD OFF -> finalize pending (post_roll={self.post_roll_sec:.2f}s, samples={samples})")

        if sec < self.min_segment_sec:
            rospy.logwarn(f"ASR: segment too short ({sec:.2f}s). Skip.")
            return

        pcm_bytes = b"".join(chunks)
        audio_f32 = pcm16le_bytes_to_float32(pcm_bytes)

        try:
            raw_text, conf = self._transcribe(audio_f32)
        except Exception as e:
            rospy.logerr(f"ASR: transcribe failed: {e}")
            return

        text = (raw_text or "").strip()
        if not text:
            rospy.loginfo("ASR: empty result.")
            return

        # publish
        self.pub_text.publish(String(data=text))
        rospy.loginfo(f"ASR: '{text}' (conf={conf:.3f})")

    def _transcribe(self, audio_f32: np.ndarray) -> Tuple[str, float]:
        # Normalize hotwords (THIS FIXES YOUR CRASH)
        hotwords = self._normalize_hotwords(self.hotwords)

        if hotwords is not None:
            rospy.loginfo(f"ASR hotwords type={type(hotwords)} value='{hotwords}'")
        else:
            rospy.loginfo("ASR hotwords=None")

        # language: allow "" or None for auto
        lang = self.language
        if lang is not None and isinstance(lang, str) and lang.strip() == "":
            lang = None

        # Run faster-whisper
        segments, info = self.model.transcribe(
            audio_f32,
            language=lang,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            vad_filter=False,  # external VAD is used
            hotwords=hotwords,  # must be str or None
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
        )

        # Collect text + a simple confidence proxy
        seg_list = list(segments)
        text = "".join([s.text for s in seg_list]).strip()

        # Confidence: use avg_logprob if available; map to [0,1] roughly
        # Note: faster-whisper's info may not have a direct "confidence".
        conf = 0.0
        avg_lp = getattr(info, "average_logprob", None)
        if avg_lp is None:
            # fallback: try from segments
            lps = [getattr(s, "avg_logprob", None) for s in seg_list]
            lps = [x for x in lps if isinstance(x, (float, int))]
            if lps:
                avg_lp = float(np.mean(lps))

        if isinstance(avg_lp, (float, int)):
            # crude squashing: logprob typically negative; -1 ~ decent
            conf = float(1.0 / (1.0 + np.exp(- (avg_lp + 1.0))))  # heuristic

        return text, conf


def main():
    node = FasterWhisperASRNode()
    rospy.spin()


if __name__ == "__main__":
    main()
