#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from audio_common_msgs.msg import AudioData

class AudioRMSMonitor(object):
    def __init__(self):
        # パラメータ
        topic = rospy.get_param("~topic", "/audio/audio")
        self.sample_format = rospy.get_param("~sample_format", "S16LE")
        self.log_interval = rospy.get_param("~log_interval", 10)  # 何フレームごとに出力するか

        self.frame_count = 0

        rospy.loginfo("AudioRMSMonitor: subscribing to %s", topic)
        rospy.Subscriber(topic, AudioData, self.audio_callback)

    def audio_callback(self, msg):
        self.frame_count += 1

        if not msg.data:
            # データが空
            if self.frame_count % self.log_interval == 0:
                rospy.logwarn("AudioRMSMonitor: empty audio frame.")
            return

        # ここでは 16bit PCM (S16LE) を前提とする
        # audio_capture 側の設定と合わせること
        if self.sample_format.upper() != "S16LE":
            # 将来的に他フォーマットを扱いたい場合はここを拡張
            if self.frame_count % self.log_interval == 0:
                rospy.logwarn("AudioRMSMonitor: unsupported sample_format: %s",
                              self.sample_format)
            return

        # uint8[] → int16配列へ変換
        buf = bytes(msg.data)
        if len(buf) % 2 != 0:
            # 16bitで割り切れない場合は無視
            if self.frame_count % self.log_interval == 0:
                rospy.logwarn("AudioRMSMonitor: odd buffer length: %d", len(buf))
            return

        samples = np.frombuffer(buf, dtype="<i2")  # little-endian 16bit signed

        if samples.size == 0:
            return

        # RMS & Peak 計算
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        peak = np.max(np.abs(samples))

        # 一定フレームごとにログ出力
        if self.frame_count % self.log_interval == 0:
            rospy.loginfo("Audio RMS: %.2f, PEAK: %d (n=%d)",
                          rms, int(peak), samples.size)


def main():
    rospy.init_node("audio_rms_monitor")
    monitor = AudioRMSMonitor()
    rospy.loginfo("AudioRMSMonitor started.")
    rospy.spin()


if __name__ == "__main__":
    main()
