#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS1 (Noetic) Faster-Whisper ASR node with external VAD gating.

Sub:
  - ~audio_topic (audio_common_msgs/AudioData) default: /audio/audio
  - ~vad_topic   (std_msgs/Bool)              default: /vad/is_speech

Pub:
  - ~text_topic          (std_msgs/String)    default: /gpsr/asr/text
  - ~utterance_end_topic (std_msgs/Bool)      default: /gpsr/asr/utterance_end

Key behavior:
  - VAD True  -> start a segment (with pre-roll)
  - VAD False -> after post-roll, finalize & transcribe
  - On finalize: publish TEXT first, then utterance_end pulse: True then False (after 50ms)
  - hotwords param accepts string or list/tuple (YAML), normalized to string for faster-whisper
"""

import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple, Union, List

import numpy as np
import rospy
from std_msgs.msg import String, Bool
from audio_common_msgs.msg import AudioData

from faster_whisper import WhisperModel


def pcm16le_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert little-endian signed 16-bit PCM bytes -> float32 [-1, 1]."""
    if not pcm_bytes:
        return np.zeros((0,), dtype=np.float32)
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if pcm.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (pcm.astype(np.float32) / 32768.0).astype(np.float32)


def compute_rms_peak_int16(pcm_bytes: bytes) -> Tuple[float, int, int]:
    """Return (rms, peak, n_samples) from int16 PCM bytes."""
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

        # GPSR-friendly defaults (match your current pipeline)
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utterance_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")

        # Audio format assumptions
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 1))

        # Segmenting
        self.pre_roll_sec = float(rospy.get_param("~pre_roll_sec", 0.20))
        self.post_roll_sec = float(rospy.get_param("~post_roll_sec", 0.35))
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 20.0))
        self.min_segment_sec = float(rospy.get_param("~min_segment_sec", 0.30))

        # Transcribe options
        self.device = rospy.get_param("~device", "auto")           # "cuda", "cpu", "auto"
        self.compute_type = rospy.get_param("~compute_type", "auto")
        self.model_size = rospy.get_param("~model_size", "small")
        self.language = rospy.get_param("~language", "en")         # "" or None -> auto
        self.task = rospy.get_param("~task", "transcribe")         # "transcribe" or "translate"

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
        self._last_stats_time = 0.0

        # Utterance end pulse config (IMPORTANT)
        # GPSR parser often expects edge; default 50ms True->False pulse
        self.utterance_end_pulse_width_sec = float(rospy.get_param("~utterance_end_pulse_width_sec", 0.05))

        # ---- State ----
        self._lock = threading.RLock()
        self._is_speech = False

        # ring buffer for pre-roll
        self._pre_roll_buf: Deque[bytes] = deque()
        self._pre_roll_samples = int(self.pre_roll_sec * self.sample_rate)

        # current segment
        self._segment_chunks: List[bytes] = []
        self._segment_samples: int = 0
        self._segment_start_time: Optional[float] = None

        # finalize pending
        self._pending_finalize = False
        self._pending_finalize_time: Optional[float] = None

        # ---- Model ----
        rospy.loginfo(f"Loading WhisperModel: size={self.model_size} device={self.device} compute_type={self.compute_type}")
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        # ---- ROS I/O ----
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_utt_end = rospy.Publisher(self.utterance_end_topic, Bool, queue_size=10)

        self.sub_audio = rospy.Subscriber(self.audio_topic, AudioData, self._on_audio, queue_size=50)
        self.sub_vad = rospy.Subscriber(self.vad_topic, Bool, self._on_vad, queue_size=50)

        self.timer = rospy.Timer(rospy.Duration(0.05), self._timer_cb)

        rospy.loginfo("FasterWhisperASRNode ready.")
        rospy.loginfo(f" audio_topic={self.audio_topic}")
        rospy.loginfo(f" vad_topic={self.vad_topic}")
        rospy.loginfo(f" text_topic={self.text_topic}")
        rospy.loginfo(f" utterance_end_topic={self.utterance_end_topic}")

        # Initialize utterance_end to False once (optional but helps edge-based consumers)
        try:
            self.pub_utt_end.publish(Bool(data=False))
        except Exception:
            pass

    # ---------- Hotwords fix ----------
    @staticmethod
    def _normalize_hotwords(hotwords: Union[None, str, List, Tuple]) -> Optional[str]:
        """Normalize hotwords to a string (faster-whisper calls .strip())."""
        if hotwords is None:
            return None
        if isinstance(hotwords, (bytes, bytearray)):
            hotwords = hotwords.decode("utf-8", errors="ignore")
        if isinstance(hotwords, (list, tuple)):
            toks = []
            for x in hotwords:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    toks.append(s)
            s = " ".join(toks).strip()
            return s if s else None
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
                rospy.loginfo("SileroVADNode: speech start")
                self._start_segment_locked(now)

            elif (not msg.data) and self._is_speech:
                self._is_speech = False
                rospy.loginfo("SileroVADNode: speech end")
                self._pending_finalize = True
                self._pending_finalize_time = now + self.post_roll_sec

    def _on_audio(self, msg: AudioData):
        pcm_bytes = msg.data
        if not pcm_bytes:
            return

        now = time.time()

        if self.log_audio_stats and (now - self._last_stats_time >= self.audio_stats_interval_sec):
            rms, peak, n = compute_rms_peak_int16(pcm_bytes)
            rospy.loginfo(f"Audio RMS: {rms:.2f}, PEAK: {peak} (n={n})")
            self._last_stats_time = now

        with self._lock:
            self._push_pre_roll_locked(pcm_bytes)

            if self._is_speech:
                self._append_segment_locked(pcm_bytes, now)
                if self._segment_start_time is not None:
                    dur = now - self._segment_start_time
                    if dur >= self.max_segment_sec:
                        rospy.logwarn(f"ASR: max_segment_sec reached ({dur:.2f}s). Forcing finalize.")
                        self._pending_finalize = True
                        self._pending_finalize_time = now

            if self._pending_finalize and self._segment_start_time is not None:
                self._append_segment_locked(pcm_bytes, now)

    def _timer_cb(self, _evt):
        with self._lock:
            if not self._pending_finalize or self._pending_finalize_time is None:
                return
            if time.time() < self._pending_finalize_time:
                return

            chunks = list(self._segment_chunks)
            samples = int(self._segment_samples)

            # reset state first
            self._pending_finalize = False
            self._pending_finalize_time = None
            self._segment_chunks = []
            self._segment_samples = 0
            self._segment_start_time = None

        # finalize outside lock
        self._finalize_and_publish(chunks, samples)

    # ---------- Segment management ----------
    def _push_pre_roll_locked(self, pcm_bytes: bytes):
        self._pre_roll_buf.append(pcm_bytes)

        bytes_per_sample = 2 * max(1, self.channels)  # S16LE
        max_bytes = self._pre_roll_samples * bytes_per_sample

        while True:
            cur_total = sum(len(x) for x in self._pre_roll_buf)
            if cur_total <= max_bytes:
                break
            if len(self._pre_roll_buf) <= 1:
                break
            self._pre_roll_buf.popleft()

    def _start_segment_locked(self, now: float):
        rospy.loginfo("ASR: VAD ON -> start segment")
        self._segment_chunks = list(self._pre_roll_buf)
        self._segment_samples = self._estimate_samples(self._segment_chunks)
        self._segment_start_time = now

    def _append_segment_locked(self, pcm_bytes: bytes, now: float):
        if self._segment_start_time is None:
            self._start_segment_locked(now)
        self._segment_chunks.append(pcm_bytes)
        self._segment_samples += self._estimate_samples([pcm_bytes])

    def _estimate_samples(self, chunks: List[bytes]) -> int:
        if not chunks:
            return 0
        bytes_per_sample = 2 * max(1, self.channels)
        total_bytes = sum(len(c) for c in chunks)
        return int(total_bytes // bytes_per_sample)

    # ---------- Publish helpers ----------
    def _publish_utterance_end_pulse(self):
        """Publish utterance_end as a True->False pulse (edge-friendly)."""
        self.pub_utt_end.publish(Bool(data=True))
        width = float(self.utterance_end_pulse_width_sec)
        if width < 0.0:
            width = 0.0
        if width > 0.0:
            rospy.sleep(width)
        self.pub_utt_end.publish(Bool(data=False))

    # ---------- Finalize (IMPORTANT: order) ----------
    def _finalize_and_publish(self, chunks: List[bytes], samples: int):
        rospy.loginfo(f"ASR: finalize (post_roll={self.post_roll_sec:.2f}s, samples={samples})")

        sec = samples / float(self.sample_rate) if self.sample_rate > 0 else 0.0
        if sec < self.min_segment_sec:
            rospy.logwarn(f"ASR: segment too short ({sec:.2f}s). Skip transcription.")
            # Even if transcription skipped, still signal end-of-utterance
            self._publish_utterance_end_pulse()
            return

        pcm_bytes = b"".join(chunks)
        audio_f32 = pcm16le_bytes_to_float32(pcm_bytes)

        try:
            text, conf = self._transcribe(audio_f32)
        except Exception as e:
            rospy.logerr(f"ASR: transcribe failed: {e}")
            # Still signal end-of-utterance so downstream doesn't hang
            self._publish_utterance_end_pulse()
            return

        text = (text or "").strip()

        # ---- ORDER IS CRITICAL ----
        # 1) publish text first
        if text:
            self.pub_text.publish(String(data=text))
            rospy.loginfo(f"ASR: '{text}' (conf={conf:.3f})")
        else:
            rospy.loginfo("ASR: empty result.")

        # 2) then publish utterance_end True->False pulse
        self._publish_utterance_end_pulse()

    def _transcribe(self, audio_f32: np.ndarray) -> Tuple[str, float]:
        hotwords = self._normalize_hotwords(self.hotwords)
        if hotwords is not None:
            rospy.loginfo(f"ASR hotwords='{hotwords}'")
        else:
            rospy.loginfo("ASR hotwords=None")

        lang = self.language
        if lang is not None and isinstance(lang, str) and lang.strip() == "":
            lang = None

        segments, info = self.model.transcribe(
            audio_f32,
            language=lang,
            task=self.task,
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperature,
            vad_filter=False,  # external VAD used
            hotwords=hotwords,
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.log_prob_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
        )

        seg_list = list(segments)
        text = "".join([s.text for s in seg_list]).strip()

        # crude confidence proxy
        conf = 0.0
        avg_lp = getattr(info, "average_logprob", None)
        if avg_lp is None:
            lps = [getattr(s, "avg_logprob", None) for s in seg_list]
            lps = [x for x in lps if isinstance(x, (float, int))]
            if lps:
                avg_lp = float(np.mean(lps))
        if isinstance(avg_lp, (float, int)):
            conf = float(1.0 / (1.0 + np.exp(- (avg_lp + 1.0))))  # heuristic

        return text, conf


def main():
    _ = FasterWhisperASRNode()
    rospy.spin()


if __name__ == "__main__":
    main()
