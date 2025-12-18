#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import torch
from std_msgs.msg import Bool, Float32
from audio_common_msgs.msg import AudioData

"""
Silero VAD を使って /audio/audio から
・/vad/is_speech (Bool)
・/vad/probability (Float32)
を出すノード。

※ 注意
  - Silero VAD の詳細なAPIはバージョンで少し変わることがあります。
  - ここでは torch.hub を使う想定の「典型的な」使い方を書いています。
  - 実際に動かすときは、公式README (snakers4/silero-vad) に合わせて
    `get_speech_timestamps` や `VADIterator` の使い方を微調整してください。
"""

class SileroVADNode(object):
    def __init__(self):
        # === パラメータ ===
        audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        self.speech_threshold = rospy.get_param("~speech_threshold", 0.5)
        self.log_interval = rospy.get_param("~log_interval", 50)  # 何フレームごとにログを出すか

        self.frame_count = 0

        rospy.loginfo("SileroVADNode: loading Silero VAD model...")

        # ★ Silero VAD モデル読み込み
        #   初回実行時はインターネット経由でモデルを取得する必要があります
        #   （オフライン運用する場合は、事前にイメージ内にキャッシュしておくなどの工夫が必要）
        torch.set_num_threads(1)
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

        # int16 PCM → [-1, 1] の float32 に正規化
        samples = np.frombuffer(buf, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return

        audio_float = samples / 32768.0

        # PyTorch Tensor に変換（バッチ1, 時系列長）
        audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

        # === Silero VAD で確率を計算 ===
        # Silero VAD モデルは、入力 audio_tensor と sample_rate から
        # フレームごとの "speech probability" を返す仕様になっています。
        #
        # 実際の戻り値の shape / 使い方はバージョンに依存するので、
        # 必要に応じて print で確認しながら調整してください。
        with torch.no_grad():
            probs = self.model(audio_tensor, self.sample_rate).cpu().numpy()

        # ここでは簡単のため、「最後のフレームの確率」を代表値として利用
        # probs.shape は (1, N) 想定 → probs[0, -1]
        if probs.ndim == 2:
            speech_prob = float(probs[0, -1])
        else:
            # 何か想定外の shape の場合は平均値を使う
            speech_prob = float(probs.mean())

        is_speech = speech_prob >= self.speech_threshold

        # publish
        self.pub_is_speech.publish(Bool(data=is_speech))
        self.pub_prob.publish(Float32(data=speech_prob))

        if self.frame_count % self.log_interval == 0:
            rospy.loginfo(
                "SileroVADNode: prob=%.3f -> is_speech=%s",
                speech_prob, is_speech
            )


def main():
    rospy.init_node("silero_vad_node")
    node = SileroVADNode()
    rospy.loginfo("SileroVADNode started.")
    rospy.spin()


if __name__ == "__main__":
    main()
