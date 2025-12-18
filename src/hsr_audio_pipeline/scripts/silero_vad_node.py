#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import torch
from std_msgs.msg import Bool, Float32
from audio_common_msgs.msg import AudioData


class SileroVADNode(object):
    def __init__(self):
        # === パラメータ ===
        audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        self.speech_threshold = rospy.get_param("~speech_threshold", 0.5)
        self.log_interval = rospy.get_param("~log_interval", 50)
        # 16kHz で 100ms = 1600 サンプル
        self.min_samples = rospy.get_param("~min_samples", 1600)

        self.frame_count = 0
        self.buffer = np.zeros(0, dtype=np.float32)

        rospy.loginfo("SileroVADNode: loading Silero VAD model...")
        torch.set_num_threads(1)

        # Silero VAD モデル読み込み
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        rospy.loginfo("SileroVADNode: model loaded.")

        # publisher
        self.pub_is_speech = rospy.Publisher("/vad/is_speech", Bool, queue_size=10)
        self.pub_prob = rospy.Publisher("/vad/probability", Float32, queue_size=10)

        # subscriber
        rospy.Subscriber(audio_topic, AudioData, self.audio_callback)
        rospy.loginfo("SileroVADNode: subscribing to %s", audio_topic)

    def audio_callback(self, msg):
        self.frame_count += 1

        if not msg.data:
            if self.frame_count % self.log_interval == 0:
                rospy.logwarn("SileroVADNode: empty audio frame.")
            return

        buf = bytes(msg.data)
        if len(buf) % 2 != 0:
            if self.frame_count % self.log_interval == 0:
                rospy.logwarn("SileroVADNode: odd buffer length: %d", len(buf))
            return

        # 16bit PCM → float32 [-1, 1]
        samples = np.frombuffer(buf, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return

        audio_float = samples / 32768.0

        # ---- バッファに追記 ----
        if self.buffer.size == 0:
            self.buffer = audio_float
        else:
            self.buffer = np.concatenate([self.buffer, audio_float])

        # 最低サンプル数に達していなければまだ推定しない
        if self.buffer.size < self.min_samples:
            return

        # 直近 min_samples 分だけ取り出して判定（100msくらい）
        chunk = self.buffer[-self.min_samples:]

        # バッファが大きくなりすぎないように、最後の 2 * min_samples だけ保持
        max_buffer = self.min_samples * 2
        if self.buffer.size > max_buffer:
            self.buffer = self.buffer[-max_buffer:]

        # Torch tensor へ
        audio_tensor = torch.from_numpy(chunk)

        # === Silero VAD 推論 ===
        try:
            with torch.no_grad():
                probs = self.model(audio_tensor, self.sample_rate).cpu().numpy()
        except Exception as e:
            # 安全のため例外を握りつぶしてログだけ
            if self.frame_count % self.log_interval == 0:
                rospy.logerr("SileroVADNode: model forward failed: %s", e)
            return

        # 出力 shape によって処理を分ける
        if probs.ndim == 1:
            speech_prob = float(probs[-1])
        elif probs.ndim == 2:
            speech_prob = float(probs[0, -1])
        else:
            speech_prob = float(probs.mean())

        is_speech = speech_prob >= self.speech_threshold

        # publish
        self.pub_is_speech.publish(Bool(data=is_speech))
        self.pub_prob.publish(Float32(data=speech_prob))

        if self.frame_count % self.log_interval == 0:
            rospy.loginfo(
                "SileroVADNode: prob=%.3f -> is_speech=%s (buffer=%d)",
                speech_prob, is_speech, self.buffer.size
            )


def main():
    rospy.init_node("silero_vad_node")
    node = SileroVADNode()
    rospy.loginfo("SileroVADNode started.")
    rospy.spin()


if __name__ == "__main__":
    main()
