#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPSR-ready Faster-Whisper ASR node (ROS1 / Noetic)

Inputs:
  - audio_common_msgs/AudioData (int16 PCM, mono, 16kHz assumed)
  - std_msgs/Bool /vad/is_speech  (from silero_vad_node.py)

Outputs:
  - std_msgs/String  /gpsr/asr/raw_text
  - std_msgs/String  /gpsr/asr/text          (corrected/normalized)
  - std_msgs/Float32 /gpsr/asr/confidence
  - std_msgs/Bool    /gpsr/asr/utterance_end (pulse True)

Design goals for RoboCup@Home GPSR:
  - Segment only on speech end (avoid partial/garbage intents)
  - Pre-roll & post-roll to avoid clipping
  - Correction dictionary always applied
  - Robust to faster-whisper hotwords support differences
"""

import time
import threading
from typing import Deque, List, Optional

import numpy as np
import rospy
from std_msgs.msg import String, Bool, Float32
from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel

# --- Auto-generated corrections dictionary (optional) ---
try:
    from corrections_candidates import CORRECTIONS  # dict: wrong -> right
except Exception:
    CORRECTIONS = {}


class RingBuffer:
    """Simple ring buffer for int16 audio (per-sample)."""

    def __init__(self, max_samples: int):
        self.max_samples = max(0, int(max_samples))
        self._buf = np.zeros((0,), dtype=np.int16)

    def append(self, x: np.ndarray):
        if self.max_samples <= 0:
            return
        if x.size == 0:
            return
        if x.dtype != np.int16:
            x = x.astype(np.int16, copy=False)
        self._buf = np.concatenate([self._buf, x])
        if self._buf.size > self.max_samples:
            self._buf = self._buf[-self.max_samples :]

    def get(self) -> np.ndarray:
        if self.max_samples <= 0 or self._buf.size == 0:
            return np.zeros((0,), dtype=np.int16)
        return self._buf.copy()


class FasterWhisperASRNode(object):
    def __init__(self):
        # --------------------------
        # ROS params
        # --------------------------
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        # Output topics (GPSR naming)
        self.raw_text_topic = rospy.get_param("~raw_text_topic", "/gpsr/asr/raw_text")
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.conf_topic = rospy.get_param("~conf_topic", "/gpsr/asr/confidence")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))

        # Segment control
        self.pre_roll_sec = float(rospy.get_param("~pre_roll_sec", 0.25))     # add before speech start
        self.post_roll_sec = float(rospy.get_param("~post_roll_sec", 0.35))   # add after speech end
        self.min_segment_sec = float(rospy.get_param("~min_segment_sec", 0.6))
        self.max_segment_sec = float(rospy.get_param("~max_segment_sec", 18.0))

        # Model options
        model_size = rospy.get_param("~model_size", "small")  # tiny/base/small/medium/large-v2...
        device = rospy.get_param("~device", "cpu")            # "cpu" or "cuda"
        compute_type = rospy.get_param("~compute_type", "float32")  # "int8_float16" etc.
        self.language = rospy.get_param("~language", "en")

        # Decoding options
        self.beam_size = int(rospy.get_param("~beam_size", 7))
        self.best_of = int(rospy.get_param("~best_of", 7))
        self.without_timestamps = bool(rospy.get_param("~without_timestamps", True))

        # Correction
        self.enable_corrections = bool(rospy.get_param("~enable_corrections", True))

        # --------------------------
        # GPSR vocabulary prompt (hotwords-like)
        # If you already have gpsr_vocab.py, you can later import it.
        # Keep it here to make this node standalone & robust.
        # --------------------------
        self.gpsr_names = [
            "Adel", "Angel", "Axel", "Charlie", "Jane",
            "Jules", "Morgan", "Paris", "Robin", "Simone",
        ]
        self.gpsr_locations = [
            "bed", "bedside table", "shelf", "trashbin", "dishwasher",
            "potted plant", "kitchen table", "chairs", "pantry",
            "refrigerator", "sink", "cabinet", "coatrack", "desk",
            "armchair", "desk lamp", "waste basket", "tv stand",
            "storage rack", "lamp", "side tables", "sofa", "bookshelf",
            "entrance", "exit",
        ]
        self.gpsr_rooms = ["bedroom", "kitchen", "office", "living room", "bathroom"]
        self.gpsr_objects = [
            "juice pack", "cola", "milk", "orange juice", "tropical juice",
            "red wine", "iced tea", "tennis ball", "rubiks cube", "baseball",
            "soccer ball", "dice", "orange", "pear", "peach", "strawberry",
            "apple", "lemon", "banana", "plum", "cornflakes", "pringles",
            "cheezit", "cup", "bowl", "fork", "plate", "knife", "spoon",
            "chocolate jello", "coffee grounds", "mustard", "tomato soup",
            "tuna", "strawberry jello", "spam", "sugar", "cleanser", "sponge",
        ]
        self.gpsr_categories = [
            "drink", "drinks", "toy", "toys", "fruit", "fruits",
            "snack", "snacks", "dish", "dishes", "food",
            "cleaning supply", "cleaning supplies",
        ]
        vocab_words = (
            self.gpsr_names
            + self.gpsr_locations
            + self.gpsr_rooms
            + self.gpsr_objects
            + self.gpsr_categories
        )
        # Whisper prompt for biasing
        self.initial_prompt = " ".join(vocab_words)
        self.hotwords = vocab_words

        # --------------------------
        # Internal state
        # --------------------------
        self._lock = threading.Lock()

        # VAD state
        self.current_vad = False
        self.prev_vad = False

        # Segment buffers
        self.segment_chunks: List[np.ndarray] = []
        self.segment_samples = 0

        # Pre-roll ring buffer (always collecting)
        self.pre_roll = RingBuffer(max_samples=int(self.pre_roll_sec * self.sample_rate))

        # Finalize control (post-roll)
        self.finalize_pending = False
        self.finalize_deadline: float = 0.0  # wall time (time.time)

        # Model
        rospy.loginfo(
            "FasterWhisperASRNode: loading model='%s' device=%s compute_type=%s",
            model_size, device, compute_type
        )
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )
        rospy.loginfo("FasterWhisperASRNode: model loaded.")

        # Publishers
        self.pub_raw = rospy.Publisher(self.raw_text_topic, String, queue_size=10)
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_conf = rospy.Publisher(self.conf_topic, Float32, queue_size=10)
        self.pub_end = rospy.Publisher(self.utt_end_topic, Bool, queue_size=10)

        # Subscribers
        rospy.Subscriber(self.audio_topic, AudioData, self.audio_callback, queue_size=200)
        rospy.Subscriber(self.vad_topic, Bool, self.vad_callback, queue_size=200)

        # Timer to finalize after post-roll (keeps logic simple & reliable)
        self.timer = rospy.Timer(rospy.Duration(0.02), self._timer_cb)  # 50Hz

        rospy.loginfo(
            "FasterWhisperASRNode ready: audio=%s vad=%s out(text)=%s",
            self.audio_topic, self.vad_topic, self.text_topic
        )

    # --------------------------
    # Callbacks
    # --------------------------
    def audio_callback(self, msg: AudioData):
        data = np.frombuffer(msg.data, dtype=np.int16)
        if data.size == 0:
            return

        with self._lock:
            # Always keep pre-roll
            self.pre_roll.append(data)

            # During speech OR while post-roll pending, collect into segment buffer
            collecting = self.current_vad or (self.finalize_pending and time.time() < self.finalize_deadline)
            if not collecting:
                return

            self.segment_chunks.append(data)
            self.segment_samples += data.size

    def vad_callback(self, msg: Bool):
        with self._lock:
            self.prev_vad = self.current_vad
            self.current_vad = bool(msg.data)

            # Speech start
            if (not self.prev_vad) and self.current_vad:
                rospy.loginfo("ASR: VAD ON -> start segment")
                self.finalize_pending = False
                self.segment_chunks = []
                self.segment_samples = 0

                # Seed with pre-roll so we don't clip the initial consonant
                pre = self.pre_roll.get()
                if pre.size > 0:
                    self.segment_chunks.append(pre)
                    self.segment_samples += pre.size

            # Speech end
            if self.prev_vad and (not self.current_vad):
                # Start post-roll window; finalization occurs in timer callback.
                self.finalize_pending = True
                self.finalize_deadline = time.time() + max(0.0, self.post_roll_sec)
                rospy.loginfo(
                    "ASR: VAD OFF -> finalize pending (post_roll=%.2fs, samples=%d)",
                    self.post_roll_sec, self.segment_samples
                )

    def _timer_cb(self, _evt):
        # finalize after post_roll time passes
        with self._lock:
            if not self.finalize_pending:
                return
            if time.time() < self.finalize_deadline:
                return
            # finalize now
            self.finalize_pending = False
            chunks = self.segment_chunks
            samples = self.segment_samples

            # Clear buffers early to be robust to re-entrance
            self.segment_chunks = []
            self.segment_samples = 0

        # Do ASR outside lock (heavy)
        self._finalize_segment(chunks, samples)

    # --------------------------
    # Segment -> Whisper
    # --------------------------
    def _finalize_segment(self, chunks: List[np.ndarray], samples: int):
        if samples <= 0 or not chunks:
            rospy.loginfo("ASR: empty segment -> skip")
            return

        duration = float(samples) / float(self.sample_rate)
        if duration < self.min_segment_sec:
            rospy.loginfo("ASR: segment too short (%.2fs) -> skip", duration)
            return
        if duration > self.max_segment_sec:
            rospy.logwarn("ASR: segment long (%.2fs). Check VAD/hangover.", duration)

        audio_int16 = np.concatenate(chunks, axis=0)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0

        # Whisper transcribe
        raw_text, conf = self._transcribe(audio_f32)

        # Publish raw
        raw_text_norm = " ".join(raw_text.strip().split())
        self.pub_raw.publish(String(data=raw_text_norm))

        # Apply corrections + normalize
        text = raw_text_norm
        if self.enable_corrections:
            text = self.apply_gpsr_corrections(text)
        text = " ".join(text.strip().split())

        rospy.loginfo("ASR: raw='%s' -> text='%s' conf=%.3f dur=%.2fs",
                      raw_text_norm, text, conf, duration)

        # Publish corrected
        self.pub_text.publish(String(data=text))
        self.pub_conf.publish(Float32(data=float(conf)))

        # Pulse utterance_end
        self.pub_end.publish(Bool(data=True))

    def _transcribe(self, audio_f32: np.ndarray) -> (str, float):
        # Some faster-whisper builds don't support hotwords; fallback.
        try:
            segments, info = self.model.transcribe(
                audio=audio_f32,
                language=self.language,
                task="transcribe",
                beam_size=self.beam_size,
                best_of=self.best_of,
                initial_prompt=self.initial_prompt,
                hotwords=self.hotwords,
                without_timestamps=self.without_timestamps,
            )
        except TypeError:
            rospy.logwarn("ASR: 'hotwords' not supported. Retry without hotwords.")
            segments, info = self.model.transcribe(
                audio=audio_f32,
                language=self.language,
                task="transcribe",
                beam_size=self.beam_size,
                best_of=self.best_of,
                initial_prompt=self.initial_prompt,
                without_timestamps=self.without_timestamps,
            )

        segments = list(segments)
        if not segments:
            return "", 0.0

        texts = []
        avg_logprobs = []
        for s in segments:
            if getattr(s, "text", ""):
                texts.append(s.text.strip())
            if hasattr(s, "avg_logprob"):
                avg_logprobs.append(float(s.avg_logprob))

        raw_text = " ".join(texts)

        # very rough confidence from avg_logprob
        if avg_logprobs:
            conf = float(np.exp(np.mean(avg_logprobs)))
            conf = float(np.clip(conf, 0.0, 1.0))
        else:
            conf = 0.5

        return raw_text, conf

    # --------------------------
    # Corrections
    # --------------------------
    def apply_gpsr_corrections(self, text: str) -> str:
        fixed = text

        # 1) auto dictionary (from corrections_candidates.py)
        for wrong, right in CORRECTIONS.items():
            if not wrong:
                continue
            fixed = fixed.replace(wrong, right)
            fixed = fixed.replace(wrong.capitalize(), right.capitalize())

        # 2) baseline common fixes
        base = {
            # rooms / furniture
            "livingroom": "living room",
            "livin room": "living room",
            "bath room": "bathroom",
            "bed side table": "bedside table",
            "trash bin": "trashbin",
            "book shelf": "bookshelf",
            "books shelve": "bookshelf",
            "refridgerator": "refrigerator",
            # objects
            "corn flakes": "cornflakes",
            "pringles chips": "pringles",
            "cheese it": "cheezit",
            "cheese itz": "cheezit",
            "rubik cube": "rubiks cube",
            "soccerball": "soccer ball",
            # categories
            "cleaning supplys": "cleaning supplies",
            "cleaning supllies": "cleaning supplies",
            # typical whisper slips
            "past room": "bathroom",   # (optional) you can remove if you dislike
        }
        for wrong, right in base.items():
            fixed = fixed.replace(wrong, right)
            fixed = fixed.replace(wrong.capitalize(), right.capitalize())

        return fixed


def main():
    rospy.init_node("faster_whisper_asr_node")
    node = FasterWhisperASRNode()
    rospy.loginfo("faster_whisper_asr_node started.")
    rospy.spin()


if __name__ == "__main__":
    main()
