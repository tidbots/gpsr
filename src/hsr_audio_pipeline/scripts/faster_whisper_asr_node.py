#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from std_msgs.msg import String, Bool, Float32
from audio_common_msgs.msg import AudioData
from faster_whisper import WhisperModel

# 自動生成した辞書をここから import する想定
# 同じパッケージ / ディレクトリに corrections_candidates.py を置いてください
try:
    from corrections_candidates import CORRECTIONS
except ImportError:
    # まだ用意していない場合でも動くように、空 dict でフォールバック
    CORRECTIONS = {}
    rospy.logwarn("faster_whisper_asr_node: corrections_candidates.py not found. "
                  "CORRECTIONS is empty.")


class FasterWhisperASRNode(object):
    def __init__(self):
        # === パラメータ ===
        audio_topic = rospy.get_param("~audio_topic", "/audio/audio")
        vad_topic = rospy.get_param("~vad_topic", "/vad/is_speech")

        self.text_topic = rospy.get_param("~text_topic", "/asr/text")
        self.conf_topic = rospy.get_param("~conf_topic", "/asr/confidence")

        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))

        model_size = rospy.get_param("~model_size", "small")   # tiny, base, small, ...
        device = rospy.get_param("~device", "cpu")
        compute_type = rospy.get_param("~compute_type", "float32")

        # GPSR は英語前提なのでデフォルトを en に
        self.language = rospy.get_param("~language", "en")

        # === GPSR 用の語彙 ===
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
        self.gpsr_rooms = [
            "bedroom", "kitchen", "office", "living room", "bathroom",
        ]
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
            "drink", "drinks",
            "toy", "toys",
            "fruit", "fruits",
            "snack", "snacks",
            "dish", "dishes",
            "food",
            "cleaning supply", "cleaning supplies",
        ]

        vocab_words = (
            self.gpsr_names
            + self.gpsr_locations
            + self.gpsr_rooms
            + self.gpsr_objects
            + self.gpsr_categories
        )

        # Whisper に「こういう単語が出やすい」と教えるための文脈
        self.initial_prompt = " ".join(vocab_words)

        # hotwords（対応していない faster-whisper もあるので try/except で扱う）
        self.hotwords = vocab_words

        # --- セグメント長の制御 ---
        self.min_segment_duration = float(
            rospy.get_param("~min_segment_duration", 0.5)
        )  # [sec] これより短い区間は無視
        self.max_segment_duration = float(
            rospy.get_param("~max_segment_duration", 15.0)
        )  # [sec] 長すぎる場合は警告のみ

        # --- VAD 状態 & バッファ ---
        self.current_vad = False
        self.prev_vad = False
        self.segment_buffer = []
        self.segment_samples = 0

        # === モデル読み込み ===
        rospy.loginfo(
            "FasterWhisperASRNode: loading model '%s' on %s (%s)",
            model_size, device, compute_type,
        )
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
        )
        rospy.loginfo("FasterWhisperASRNode: model loaded.")

        # === Publisher / Subscriber ===
        self.pub_text = rospy.Publisher(self.text_topic, String, queue_size=10)
        self.pub_conf = rospy.Publisher(self.conf_topic, Float32, queue_size=10)

        rospy.Subscriber(audio_topic, AudioData, self.audio_callback, queue_size=100)
        rospy.Subscriber(vad_topic, Bool, self.vad_callback, queue_size=100)

        rospy.loginfo(
            "FasterWhisperASRNode: audio_topic=%s vad_topic=%s language=%s",
            audio_topic, vad_topic, self.language,
        )

    # =========================================================
    # コールバック
    # =========================================================

    def audio_callback(self, msg: AudioData):
        """VAD が True の間だけ音声をバッファに貯める。"""
        if not self.current_vad:
            return

        data = np.frombuffer(msg.data, dtype=np.int16)
        if data.size == 0:
            return

        self.segment_buffer.append(data)
        self.segment_samples += data.shape[0]

    def vad_callback(self, msg: Bool):
        """
        VAD の ON/OFF で区間を切り、True→False の立ち下がりで ASR を走らせる。
        """
        self.prev_vad = self.current_vad
        self.current_vad = bool(msg.data)

        # 立ち上がり: 新しい区間を開始
        if (not self.prev_vad) and self.current_vad:
            rospy.loginfo("FasterWhisperASRNode: VAD ON, start new segment.")
            self.segment_buffer = []
            self.segment_samples = 0

        # 立ち下がり: 発話区間の終了 → ASR 実行
        if self.prev_vad and (not self.current_vad):
            rospy.loginfo(
                "FasterWhisperASRNode: VAD OFF, finalize segment (%d samples).",
                self.segment_samples,
            )
            self.finalize_segment()

    # =========================================================
    # セグメント処理
    # =========================================================

    def finalize_segment(self):
        """現在のバッファを 1 発話として ASR にかける。"""
        if self.segment_samples <= 0 or len(self.segment_buffer) == 0:
            rospy.loginfo("FasterWhisperASRNode: empty segment, skip.")
            self.segment_buffer = []
            self.segment_samples = 0
            return

        duration_sec = float(self.segment_samples) / float(self.sample_rate)
        if duration_sec < self.min_segment_duration:
            rospy.loginfo(
                "FasterWhisperASRNode: segment too short (%.2f sec), skip.",
                duration_sec,
            )
            self.segment_buffer = []
            self.segment_samples = 0
            return

        if duration_sec > self.max_segment_duration:
            rospy.logwarn(
                "FasterWhisperASRNode: long segment (%.2f sec). "
                "Check your VAD parameters.",
                duration_sec,
            )

        audio_int16 = np.concatenate(self.segment_buffer, axis=0)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # === Whisper 呼び出し ===
        try:
            segments, info = self.model.transcribe(
                audio=audio_float32,
                language=self.language,
                task="transcribe",
                beam_size=7,
                best_of=7,
                patience=1.0,
                length_penalty=1.0,
                initial_prompt=self.initial_prompt,
                hotwords=self.hotwords,          # 対応していない環境だと TypeError
                without_timestamps=True,
            )
        except TypeError:
            rospy.logwarn(
                "FasterWhisperASRNode: 'hotwords' not supported, "
                "retrying without it."
            )
            segments, info = self.model.transcribe(
                audio=audio_float32,
                language=self.language,
                task="transcribe",
                beam_size=7,
                best_of=7,
                patience=1.0,
                length_penalty=1.0,
                initial_prompt=self.initial_prompt,
                without_timestamps=True,
            )

        segments = list(segments)
        if not segments:
            rospy.loginfo("FasterWhisperASRNode: no text recognized.")
            self.segment_buffer = []
            self.segment_samples = 0
            return

        # --- テキスト結合（スペース区切り & 正規化） ---
        texts = []
        for seg in segments:
            if seg.text:
                texts.append(seg.text.strip())

        text = " ".join(texts)
        text = " ".join(text.split())  # 余分なスペースを 1 個に

        # --- GPSR 用の誤認識補正 ---
        text = self.apply_gpsr_corrections(text)

        # --- confidence のざっくり計算 ---
        avg_logprobs = [s.avg_logprob for s in segments if hasattr(s, "avg_logprob")]
        if avg_logprobs:
            conf = float(np.exp(np.mean(avg_logprobs)))
        else:
            conf = 0.5

        rospy.loginfo(
            "FasterWhisperASRNode: text='%s' (conf=%.3f, %.2fs, %d segments)",
            text, conf, duration_sec, len(segments),
        )

        # publish
        self.pub_text.publish(String(data=text))
        self.pub_conf.publish(Float32(data=conf))

        # バッファクリア
        self.segment_buffer = []
        self.segment_samples = 0

    # =========================================================
    # 認識結果の補正
    # =========================================================

    def apply_gpsr_corrections(self, text: str) -> str:
        """
        GPSR コマンド向けの誤認識補正。
        - 自動生成辞書 CORRECTIONS をまず適用
        - それでも足りない典型的なミスは手書き辞書で補完
        """
        fixed = text

        # 1) 自動生成辞書（corrections_candidates.py）を適用
        for wrong, right in CORRECTIONS.items():
            if not wrong:
                continue
            fixed = fixed.replace(wrong, right)
            # 先頭だけ大文字のパターンも一応補正
            fixed = fixed.replace(wrong.capitalize(), right.capitalize())

        # 2) ベースラインの手書き補正（必要に応じて拡張）
        base_corrections = {
            # 部屋・家具
            "livingroom": "living room",
            "livin room": "living room",
            "bath room": "bathroom",
            "bed side table": "bedside table",
            "trash bin": "trashbin",
            "book shelf": "bookshelf",
            "books shelve": "bookshelf",
            "kitchen tablee": "kitchen table",
            "refridgerator": "refrigerator",
            # 物体
            "corn flakes": "cornflakes",
            "pringles chips": "pringles",
            "cheese it": "cheezit",
            "cheese itz": "cheezit",
            "rubik cube": "rubiks cube",
            "soccerball": "soccer ball",
            # カテゴリ
            "cleaning supplys": "cleaning supplies",
            "cleaning supllies": "cleaning supplies",
        }

        for wrong, right in base_corrections.items():
            fixed = fixed.replace(wrong, right)
            fixed = fixed.replace(wrong.capitalize(), right.capitalize())

        return fixed


def main():
    rospy.init_node("faster_whisper_asr_node")
    node = FasterWhisperASRNode()
    rospy.loginfo("FasterWhisperASRNode started.")
    rospy.spin()


if __name__ == "__main__":
    main()
