#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import String, Bool, Float32
from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel   # ★ Whisper -> WhisperModel


class FasterWhisperASRNode(object):
    def __init__(self):
        # === パラメータ ===
        audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        self.min_segment_sec = rospy.get_param("~min_segment_sec", 0.3)
        self.max_segment_sec = rospy.get_param("~max_segment_sec", 10.0)

        model_size = rospy.get_param("~model_size", "small")   # tiny, base, small, ...
        device = rospy.get_param("~device", "cpu")
        compute_type = rospy.get_param("~compute_type", "float32")

        self.language = rospy.get_param("~language", "ja")

        self.frame_count = 0
        self.current_vad = False
        self.prev_vad = False

        # 発話区間バッファ
        self.segment_buffer = []   # list[np.ndarray]
        self.segment_samples = 0   # current segment length [samples]

        rospy.loginfo("FasterWhisperASRNode: loading model '%s' on %s (%s)",
                      model_size, device, compute_type)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        rospy.loginfo("FasterWhisperASRNode: model loaded")

        # Publisher
        self.pub_text = rospy.Publisher("/asr/text", String, queue_size=10)
        self.pub_conf = rospy.Publisher("/asr/confidence", Float32, queue_size=10)

        # Subscriber
        rospy.Subscriber(audio_topic, AudioData, self.audio_callback)
        rospy.Subscriber(vad_topic, Bool, self.vad_callback)

        rospy.loginfo("FasterWhisperASRNode: subscribing audio=%s, vad=%s",
                      audio_topic, vad_topic)

    def vad_callback(self, msg: Bool):
        self.prev_vad = self.current_vad
        self.current_vad = msg.data
        # 実処理は audio_callback 側でやる

    def audio_callback(self, msg: AudioData):
        self.frame_count += 1

        if not msg.data:
            return

        buf = bytes(msg.data)
        if len(buf) % 2 != 0:
            return

        # 16bit PCM → float32 [-1, 1]
        samples = np.frombuffer(buf, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return

        audio_float = samples / 32768.0
        frame_len = samples.size

        # --- VAD に応じてバッファリング ---
        if self.current_vad:
            # 話している間はひたすら貯める
            self.segment_buffer.append(audio_float)
            self.segment_samples += frame_len

            # 長すぎる発話は強制確定
            if (self.segment_samples / float(self.sample_rate)) >= self.max_segment_sec:
                rospy.loginfo("FasterWhisperASRNode: max_segment_sec reached, finalize segment")
                self.finalize_segment()
                self.reset_segment()
        else:
            # いまは無音だけど、直前まで話していた場合 → 確定
            if self.prev_vad and self.segment_samples > 0:
                rospy.loginfo("FasterWhisperASRNode: VAD off, finalize segment")
                self.finalize_segment()
                self.reset_segment()

        # prev_vad 更新
        self.prev_vad = self.current_vad

    def reset_segment(self):
        self.segment_buffer = []
        self.segment_samples = 0

    def finalize_segment(self):
        if not self.segment_buffer or self.segment_samples == 0:
            return

        duration_sec = self.segment_samples / float(self.sample_rate)
        if duration_sec < self.min_segment_sec:
            rospy.loginfo("FasterWhisperASRNode: segment too short (%.2f sec), ignore",
                          duration_sec)
            return

        audio_np = np.concatenate(self.segment_buffer, axis=0)

        rospy.loginfo("FasterWhisperASRNode: transcribing segment (%.2f sec, %d samples)",
                      duration_sec, audio_np.size)

        try:
            segments, info = self.model.transcribe(
                audio_np,
                language=self.language,
                beam_size=5,
            )
        except Exception as e:
            rospy.logerr("FasterWhisperASRNode: transcription failed: %s", e)
            return

        texts = []
        conf_scores = []

        try:
            for seg in segments:
                if seg.text:
                    texts.append(seg.text)
                if hasattr(seg, "no_speech_prob") and seg.no_speech_prob is not None:
                    conf_scores.append(1.0 - float_
