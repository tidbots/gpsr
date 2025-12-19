#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import String, Float32
from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel


class FasterWhisperASRSimpleNode(object):
    def __init__(self):
        audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        self.segment_sec = rospy.get_param("~segment_sec", 2.0)  # 2秒ごとに切る
        model_size = rospy.get_param("~model_size", "small")
        device = rospy.get_param("~device", "cpu")
        compute_type = rospy.get_param("~compute_type", "float32")
        self.language = rospy.get_param("~language", "en")

        self.segment_samples_target = int(self.sample_rate * self.segment_sec)
        self.buffer = np.zeros(0, dtype=np.float32)

        rospy.loginfo("FasterWhisperASRSimpleNode: loading model '%s' on %s (%s)",
                      model_size, device, compute_type)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        rospy.loginfo("FasterWhisperASRSimpleNode: model loaded")

        self.pub_text = rospy.Publisher("/asr/text", String, queue_size=10)
        self.pub_conf = rospy.Publisher("/asr/confidence", Float32, queue_size=10)

        rospy.Subscriber(audio_topic, AudioData, self.audio_callback)
        rospy.loginfo("FasterWhisperASRSimpleNode: subscribing audio=%s", audio_topic)

    def audio_callback(self, msg: AudioData):
        if not msg.data:
            return

        buf = bytes(msg.data)
        if len(buf) % 2 != 0:
            return

        samples = np.frombuffer(buf, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return

        audio_float = samples / 32768.0

        # バッファに追記
        if self.buffer.size == 0:
            self.buffer = audio_float
        else:
            self.buffer = np.concatenate([self.buffer, audio_float])

        # 目標サンプル数を超えたら、一回 ASR にかけてバッファを空にする
        if self.buffer.size >= self.segment_samples_target:
            self.transcribe_current_buffer()
            self.buffer = np.zeros(0, dtype=np.float32)

    def transcribe_current_buffer(self):
        if self.buffer.size == 0:
            return

        duration_sec = self.buffer.size / float(self.sample_rate)
        rospy.loginfo("FasterWhisperASRSimpleNode: transcribing buffer (%.2f sec, %d samples)",
                      duration_sec, self.buffer.size)

        try:
            segments, info = self.model.transcribe(
                self.buffer,
                language=self.language,
                beam_size=5,
            )
        except Exception as e:
            rospy.logerr("FasterWhisperASRSimpleNode: transcription failed: %s", e)
            return

        texts = []
        try:
            for seg in segments:
                if seg.text:
                    texts.append(seg.text)
        except Exception as e:
            rospy.logwarn("FasterWhisperASRSimpleNode: error iterating segments: %s", e)

        if not texts:
            rospy.loginfo("FasterWhisperASRSimpleNode: empty transcription result")
            return

        text = "".join(texts).strip()
        rospy.loginfo("FasterWhisperASRSimpleNode: text='%s'", text)

        # 仮で 1.0 を confidence に
        self.pub_text.publish(String(data=text))
        self.pub_conf.publish(Float32(data=1.0))


def main():
    rospy.init_node("faster_whisper_asr_simple_node")
    node = FasterWhisperASRSimpleNode()
    rospy.loginfo("FasterWhisperASRSimpleNode started.")
    rospy.spin()


if __name__ == "__main__":
    main()
