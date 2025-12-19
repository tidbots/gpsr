#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import Bool
from audio_common_msgs.msg import AudioData

import torch


class SileroVADNode(object):
    """ROS1 node using Silero VAD.

    - subscribe: audio_common_msgs/AudioData (16 kHz, mono, int16 PCM)
    - publish : std_msgs/Bool on /vad/is_speech
    - processes 32 ms chunks (512 samples @ 16kHz, 256 @ 8kHz)
    """

    def __init__(self):
        # Parameters
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        # Silero VAD は 8kHz / 16kHz のみ
        if self.sample_rate not in (8000, 16000):
            rospy.logwarn(
                "SileroVADNode: sample_rate %d is not 8000 or 16000, "
                "forcing to 16000 for Silero VAD.",
                self.sample_rate,
            )
            self.sample_rate = 16000

        # 512 samples at 16kHz ≈ 32 ms
        # 256 samples at 8kHz  ≈ 32 ms
        self.chunk_size = 512 if self.sample_rate == 16000 else 256

        # しきい値（あとで rosparam で微調整できます）
        self.speech_threshold = float(rospy.get_param("~speech_threshold", 0.5))
        self.silence_threshold = float(rospy.get_param("~silence_threshold", 0.3))
        self.hangover_ms = int(rospy.get_param("~hangover_ms", 300))

        # hangover を「何チャンク分続けるか」に変換
        self.hangover_chunks = max(
            1,
            int(
                round(
                    (self.hangover_ms / 1000.0)
                    * (self.sample_rate / float(self.chunk_size))
                )
            ),
        )

        # 内部状態
        self.buffer = np.array([], dtype=np.int16)
        self.current_is_speech = False
        self.hangover_count = 0

        # Silero VAD モデル読み込み
        torch.set_num_threads(1)
        rospy.loginfo("SileroVADNode: loading Silero VAD model via torch.hub...")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model.eval()
        rospy.loginfo("SileroVADNode: model loaded.")

        # ROS pub/sub
        self.pub_vad = rospy.Publisher(self.vad_topic, Bool, queue_size=10)
        rospy.Subscriber(
            self.audio_topic, AudioData, self.audio_callback, queue_size=100
        )

        rospy.loginfo(
            "SileroVADNode: audio_topic=%s vad_topic=%s sample_rate=%d chunk_size=%d "
            "thresholds=(speech=%.2f, silence=%.2f) hangover_chunks=%d",
            self.audio_topic,
            self.vad_topic,
            self.sample_rate,
            self.chunk_size,
            self.speech_threshold,
            self.silence_threshold,
            self.hangover_chunks,
        )

    def audio_callback(self, msg: AudioData):
        # AudioData.data は bytes (int16 PCM を想定)
        data = np.frombuffer(msg.data, dtype=np.int16)
        if data.size == 0:
            return

        # バッファに溜める
        self.buffer = np.concatenate([self.buffer, data])

        # chunk_size 以上たまったら処理
        while self.buffer.size >= self.chunk_size:
            chunk_int16 = self.buffer[: self.chunk_size]
            self.buffer = self.buffer[self.chunk_size :]

            speech_prob = self._infer_chunk(chunk_int16)
            self._update_state_and_publish(speech_prob)

    def _infer_chunk(self, chunk_int16: np.ndarray) -> float:
        # int16 → float32[-1, 1] に正規化
        audio_float32 = chunk_int16.astype(np.float32)
        abs_max = np.max(np.abs(audio_float32))
        if abs_max > 0:
            audio_float32 = audio_float32 / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)

        # Silero VAD: model(waveform, sample_rate) → speech probability
        with torch.no_grad():
            prob = float(self.model(audio_tensor, self.sample_rate).item())

        return prob

    def _update_state_and_publish(self, speech_prob: float):
        """ヒステリシス + hangover で /vad/is_speech を安定させる."""

        if speech_prob >= self.speech_threshold:
            # 音声が "ある" と判断
            if not self.current_is_speech:
                rospy.loginfo(
                    "SileroVADNode: speech start (prob=%.3f)", speech_prob
                )
            self.current_is_speech = True
            self.hangover_count = self.hangover_chunks
        else:
            # 既に発話中なら hangover カウントダウン
            if self.current_is_speech:
                self.hangover_count -= 1
                if self.hangover_count <= 0 and speech_prob <= self.silence_threshold:
                    # 発話終了
                    self.current_is_speech = False
                    rospy.loginfo(
                        "SileroVADNode: speech end (prob=%.3f)", speech_prob
                    )
            else:
                # もともと無音
                self.current_is_speech = False
                self.hangover_count = 0

        # /vad/is_speech を publish
        self.pub_vad.publish(Bool(data=self.current_is_speech))

    def spin(self):
        rospy.spin()


def main():
    rospy.init_node("silero_vad_node")
    node = SileroVADNode()
    rospy.loginfo("silero_vad_node started.")
    node.spin()


if __name__ == "__main__":
    main()
