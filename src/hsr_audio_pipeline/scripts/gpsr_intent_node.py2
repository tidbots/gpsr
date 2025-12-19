#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
from std_msgs.msg import String


class GpsrIntentNode(object):
    """
    /asr/text に流れてくる音声認識結果（日本語）を受け取り、
    /gpsr/intent に「意図っぽい情報」を JSON 文字列で publish する簡易ノード。

    まずは最小限のルールベース：
      - コマンド種別（bring / guide / answer / other）
      - 生テキスト
      - スロット（object / source / destination / person）はとりあえず空 or 簡易抽出
    """

    def __init__(self):
        asr_topic = rospy.get_param("~asr_topic", "/asr/text")
        intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.pub_intent = rospy.Publisher(intent_topic, String, queue_size=10)
        rospy.Subscriber(asr_topic, String, self.asr_callback)

        rospy.loginfo("GpsrIntentNode: subscribing %s, publishing %s",
                      asr_topic, intent_topic)

    def asr_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        rospy.loginfo("GpsrIntentNode: received ASR text: '%s'", text)

        intent_type = self.detect_intent_type(text)
        slots = self.extract_slots_simple(text, intent_type)

        intent = {
            "raw_text": text,
            "intent_type": intent_type,   # "bring", "guide", "answer", "other"
            "slots": slots,               # dict
        }

        intent_json = json.dumps(intent, ensure_ascii=False)
        rospy.loginfo("GpsrIntentNode: intent=%s", intent_json)

        self.pub_intent.publish(String(data=intent_json))

    # --- 簡易ルール群 ---

    def detect_intent_type(self, text: str) -> str:
        """
        超簡易ルールで GPSR のコマンド種別っぽいものを推定。
        必要に応じてルールを足していく想定。
        """
        # 典型的な持ってきて系
        if ("持ってきて" in text) or ("運んで" in text) or ("取ってきて" in text):
            return "bring"

        # 案内系
        if ("案内して" in text) or ("連れて行って" in text) or ("連れていって" in text):
            return "guide"

        # 回答系（質問に答える）
        if ("教えて" in text) or ("何ですか" in text) or ("答えて" in text):
            return "answer"

        return "other"

    def extract_slots_simple(self, text: str, intent_type: str):
        """
        とりあえず「オブジェクト」を雑に拾うくらいの簡易実装。
        - 「〜を持ってきて」の「〜」部分を object に入れる程度。
        本番用には、ここをちゃんと文法 or LLM ベースの NLU に差し替える想定。
        """
        slots = {
            "object": "",
            "source": "",
            "destination": "",
            "person": "",
        }

        if intent_type == "bring":
            # ざっくり「〜を持ってきて」の「〜を」部分を拾う
            # ex) "テーブルの上のペットボトルを持ってきて"
            try:
                if "を持ってきて" in text:
                    before = text.split("を持ってきて")[0]
                    # 最後の "は/が/を" を削るくらいの簡易処理
                    slots["object"] = before
                elif "を取ってきて" in text:
                    slots["object"] = text.split("を取ってきて")[0]
                elif "を運んで" in text:
                    slots["object"] = text.split("を運んで")[0]
            except Exception as e:
                rospy.logwarn("GpsrIntentNode: slot extraction failed: %s", e)

        # 必要に応じて guide/answer 用の簡易抽出も追加していく

        return slots


def main():
    rospy.init_node("gpsr_intent_node")
    node = GpsrIntentNode()
    rospy.loginfo("GpsrIntentNode started.")
    rospy.spin()


if __name__ == "__main__":
    main()
