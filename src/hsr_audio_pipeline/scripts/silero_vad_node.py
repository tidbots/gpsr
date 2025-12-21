#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
silero_vad_node.py (ROS1 / Noetic)

Sub:
  - ~audio_topic (audio_common_msgs/AudioData) default: /audio/audio

Pub:
  - ~vad_topic   (std_msgs/Bool)               default: /vad/is_speech
  - (optional) ~prob_topic (std_msgs/Float32)  default: /vad/prob

Features:
  - Uses Silero VAD via torch.hub
  - Respects TORCH_HOME (persistable) and allows ROS params:
      ~torch_home      (default: /data/models/torch)
      ~torch_hub_dir   (default: <torch_home>/hub)
    If not writable -> fallback to ~/.cache/hsr_models/torch
  - Simple state machine with thresholds + hangover
"""

import os
import time
from typing import Optional

import numpy as np
import rospy

from std_msgs.msg import Bool, Float32
from audio_common_msgs.msg import AudioData


def _ensure_writable_dir(preferred: str, fallback: str, label: str) -> str:
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


class SileroVADNode:
    def __init__(self):
        rospy.init_node("silero_vad")

        # ---- params ----
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")
        self.prob_topic = rospy.get_param("~prob_topic", "/vad/prob")
        self.publish_prob = bool(rospy.get_param("~publish_prob", False))

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))

        # chunk_size in samples (Silero works well with 256/512/1024 @16k)
        self.chunk_size = int(rospy.get_param("~chunk_size", 512))

        # thresholds
        self.speech_threshold = float(rospy.get_param("~speech_threshold", 0.6))
        self.silence_threshold = float(rospy.get_param("~silence_threshold", 0.3))

        # hangover in ms: after speech drops below silence_threshold, keep "speech" for a while
        self.hangover_ms = int(rospy.get_param("~hangover_ms", 400))

        # ---- persistent torch cache ----
        preferred_torch_home = rospy.get_param("~torch_home", "/data/models/torch")
        fallback_torch_home = os.path.expanduser("~/.cache/hsr_models/torch")
        self.torch_home = _ensure_writable_dir(preferred_torch_home, fallback_torch_home, "torch_home")

        preferred_hub_dir = rospy.get_param("~torch_hub_dir", os.path.join(self.torch_home, "hub"))
        fallback_hub_dir = os.path.join(self.torch_home, "hub")
        self.torch_hub_dir = _ensure_writable_dir(preferred_hub_dir, fallback_hub_dir, "torch_hub_dir")

        # Ensure env + torch hub dir
        os.environ["TORCH_HOME"] = self.torch_home

        import torch
        # torch.hub cache dir is separate; set explicitly
        try:
            torch.hub.set_dir(self.torch_hub_dir)
        except Exception as e:
            rospy.logwarn("torch.hub.set_dir failed (%s). Continuing.", e)

        rospy.loginfo("SileroVAD: TORCH_HOME=%s torch_hub_dir=%s", os.environ["TORCH_HOME"], self.torch_hub_dir)

        # ---- load model ----
        rospy.loginfo("SileroVADNode: loading Silero VAD model via torch.hub...")
        # NOTE: trust_repo=True avoids future prompt-like behavior
        self.model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            force_reload=False,
        )
        self.model.eval()
        rospy.loginfo("SileroVADNode: model loaded.")

        # ---- state ----
        self._buf = np.zeros((0,), dtype=np.float32)
        self._is_speech = False

        # how many chunks to keep speech after silence
        chunk_ms = (self.chunk_size / float(self.sample_rate)) * 1000.0
        self._hangover_chunks = max(0, int(round(self.hangover_ms / chunk_ms)))
        self._hang_counter = 0

        # ---- ROS pub/sub ----
        self.pub_vad = rospy.Publisher(self.vad_topic, Bool, queue_size=10)
        self.pub_prob = rospy.Publisher(self.prob_topic, Float32, queue_size=10) if self.publish_prob else None

        rospy.Subscriber(self.audio_topic, AudioData, self._on_audio, queue_size=200)

        rospy.loginfo(
            "SileroVADNode ready: audio_topic=%s vad_topic=%s sample_rate=%d chunk_size=%d thresholds=(speech=%.2f silence=%.2f) hangover_chunks=%d",
            self.audio_topic, self.vad_topic, self.sample_rate, self.chunk_size,
            self.speech_threshold, self.silence_threshold, self._hangover_chunks
        )

    def _publish_state(self, is_speech: bool, prob: Optional[float] = None):
        # publish every update (cheap and easier to debug)
        self.pub_vad.publish(Bool(data=bool(is_speech)))
        if self.pub_prob is not None and prob is not None:
            self.pub_prob.publish(Float32(data=float(prob)))

    def _on_audio(self, msg: AudioData):
        # Expect S16LE mono from audio_capture
        pcm = np.frombuffer(bytes(msg.data), dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return

        # append to buffer
        self._buf = np.concatenate([self._buf, pcm])

        import torch

        # process in fixed-size chunks
        while self._buf.size >= self.chunk_size:
            chunk = self._buf[: self.chunk_size]
            self._buf = self._buf[self.chunk_size :]

            with torch.no_grad():
                t = torch.from_numpy(chunk)
                # Silero expects 1D float tensor and sample_rate int
                prob = float(self.model(t, self.sample_rate).item())

            # state machine
            if not self._is_speech:
                if prob >= self.speech_threshold:
                    self._is_speech = True
                    self._hang_counter = self._hangover_chunks
                    rospy.loginfo("SileroVADNode: speech start (prob=%.3f)", prob)
                self._publish_state(self._is_speech, prob)
            else:
                # currently speech
                if prob >= self.speech_threshold:
                    self._hang_counter = self._hangover_chunks
                    self._publish_state(True, prob)
                elif prob <= self.silence_threshold:
                    if self._hang_counter > 0:
                        self._hang_counter -= 1
                        self._publish_state(True, prob)
                    else:
                        self._is_speech = False
                        rospy.loginfo("SileroVADNode: speech end (prob=%.3f)", prob)
                        self._publish_state(False, prob)
                else:
                    # between thresholds: keep previous state, decay hangover slowly
                    if self._hang_counter > 0:
                        self._hang_counter -= 1
                    self._publish_state(True, prob)


def main():
    _ = SileroVADNode()
    rospy.spin()


if __name__ == "__main__":
    main()
