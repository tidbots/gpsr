#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.dirname(__file__))

import json
import rospy
from std_msgs.msg import String, Bool, Float32

from gpsr_parser import GpsrParser

class GpsrParserNode:
    def __init__(self):
        # ---- Params ----
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")

        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        # utterance_end を受けた時に使う “直近の確定テキスト”
        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf = None
        self._latest_conf_stamp = rospy.Time(0)

        # 何秒以内の text を “同一発話” とみなすか（ASRとイベントの微小なズレ吸収）
        self.max_text_age_sec = float(rospy.get_param("~max_text_age_sec", 1.0))

        # 低信頼度なら parse せず need_confirm を返す（0〜1, 無効化は <0）
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # ---- Vocabulary (現状維持：既存コードと同じ) ----
        person_names = [
            "Adel", "Angel", "Axel", "Charlie", "Jane",
            "Jules", "Morgan", "Paris", "Robin", "Simone",
        ]
        locations = [
            "bed", "bedside table", "shelf", "trashbin", "dishwasher",
            "potted plant", "kitchen table", "chairs", "pantry",
            "refrigerator", "sink", "cabinet", "coatrack", "desk",
            "armchair", "desk lamp", "waste basket", "tv stand",
            "storage rack", "lamp", "side tables", "sofa", "bookshelf",
            "entrance", "exit",
        ]
        placements = [
            "bed", "bedside table", "shelf", "dishwasher",
            "kitchen table", "pantry", "refrigerator", "sink",
            "cabinet", "desk", "tv stand", "storage rack",
            "side tables", "sofa", "bookshelf",
        ]
        rooms = ["bedroom", "kitchen", "office", "living room", "bathroom"]
        objects = [
            "juice pack", "cola", "milk", "orange juice", "tropical juice",
            "red wine", "iced tea", "tennis ball", "rubiks cube", "baseball",
            "soccer ball", "dice", "orange", "pear", "peach", "strawberry",
            "apple", "lemon", "banana", "plum", "cornflakes", "pringles",
            "cheezit", "cup", "bowl", "fork", "plate", "knife", "spoon",
            "chocolate jello", "coffee grounds", "mustard", "tomato soup",
            "tuna", "strawberry jello", "spam", "sugar", "cleanser", "sponge",
        ]
        categories = [
            ("drink", "drinks"),
            ("toy", "toys"),
            ("fruit", "fruits"),
            ("snack", "snacks"),
            ("dish", "dishes"),
            ("food", "food"),
            ("cleaning supply", "cleaning supplies"),
        ]
        cat_sing = [c[0] for c in categories]
        cat_plur = [c[1] for c in categories]

        self.parser = GpsrParser(
            person_names=person_names,
            location_names=locations,
            placement_location_names=placements,
            room_names=rooms,
            object_names=objects,
            object_categories_plural=cat_plur,
            object_categories_singular=cat_sing,
        )

        # ---- ROS pub/sub ----
        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)

        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)

        # confidence はあれば使う（無くても動く）
        rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)

        rospy.loginfo(
            "gpsr_parser_node: text=%s utterance_end=%s conf=%s -> intent=%s",
            self.text_topic, self.utt_end_topic, self.conf_topic, self.intent_topic
        )

    def _on_text(self, msg: String):
        self._latest_text = msg.data or ""
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)
        self._latest_conf_stamp = rospy.Time.now()

    def _on_utt_end(self, msg: Bool):
        # pulse True を期待
        if not bool(msg.data):
            return

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec()
        text = self._latest_text.strip()

        if (not text) or (age > self.max_text_age_sec):
            rospy.logwarn(
                "gpsr_parser_node: utterance_end but no fresh text (age=%.3f, text='%s')",
                age, text
            )
            return

        # optional: confidence gate
        if self.min_confidence >= 0.0 and self._latest_conf is not None:
            if self._latest_conf < self.min_confidence:
                payload = {
                    "ok": False,
                    "error": "need_confirm",
                    "raw_text": text,
                    "confidence": self._latest_conf,
                }
                self._publish(payload)
                rospy.logwarn(
                    "gpsr_parser_node: low confidence %.3f < %.3f -> need_confirm",
                    self._latest_conf, self.min_confidence
                )
                return

        rospy.loginfo("gpsr_parser_node: finalize parse text='%s'", text)

        command = self.parser.parse(text)
        if command is None:
            payload = {
                "ok": False,
                "error": "parse_failed",
                "raw_text": text,
                "confidence": self._latest_conf,
            }
        else:
            payload = {
                "ok": True,
                "command": json.loads(command.to_json()),
                "raw_text": text,
                "confidence": self._latest_conf,
            }

        self._publish(payload)

    def _publish(self, payload: dict):
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(out)


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.loginfo("gpsr_parser_node started (utterance_end driven).")
    rospy.spin()


if __name__ == "__main__":
    main()
