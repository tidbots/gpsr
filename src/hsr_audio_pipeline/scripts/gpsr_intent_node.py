#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import rospy
from std_msgs.msg import String, Float32


class GPSRIntentNode:
    """
    Lightweight intent+slot extractor that publishes unified schema gpsr_intent_v1.

    Inputs:
      - /gpsr/asr/text (String)
      - /gpsr/asr/confidence (Float32) optional

    Outputs:
      - /gpsr/intent (String JSON): gpsr_intent_v1

    Notes:
      - This node is useful when you want a backup/alternative to gpsr_parser_node.py
      - For best GPSR stability, run parsing on utterance_end (parser node). This node can still be used.
    """

    def __init__(self):
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        # If confidence is available and >=0: low confidence -> need_confirm
        self.confirm_threshold = float(rospy.get_param("~confirm_threshold", -1.0))

        # Simple vocab (should match your GPSR vocab)
        self.rooms = ["bedroom", "kitchen", "office", "living room", "bathroom"]
        self.names = ["adel", "angel", "axel", "charlie", "jane", "jules", "morgan", "paris", "robin", "simone"]

        self.objects = [
            "juice pack", "cola", "milk", "orange juice", "tropical juice",
            "red wine", "iced tea", "tennis ball", "rubiks cube", "baseball",
            "soccer ball", "dice", "orange", "pear", "peach", "strawberry",
            "apple", "lemon", "banana", "plum", "cornflakes", "pringles",
            "cheezit", "cup", "bowl", "fork", "plate", "knife", "spoon",
            "chocolate jello", "coffee grounds", "mustard", "tomato soup",
            "tuna", "strawberry jello", "spam", "sugar", "cleanser", "sponge",
        ]

        self.latest_conf = None

        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)
        rospy.Subscriber(self.text_topic, String, self.on_text, queue_size=20)
        rospy.Subscriber(self.conf_topic, Float32, self.on_conf, queue_size=20)

        rospy.loginfo("gpsr_intent_node: text=%s conf=%s -> intent=%s", self.text_topic, self.conf_topic, self.intent_topic)

    def on_conf(self, msg: Float32):
        self.latest_conf = float(msg.data)

    def coarse_intent_type(self, intent_name: str) -> str:
        s = (intent_name or "").lower()
        if "guide" in s or "escort" in s or "lead" in s:
            return "guide"
        if any(k in s for k in ["bring", "fetch", "deliver", "take", "give"]):
            return "bring"
        if any(k in s for k in ["count", "answer", "tell", "describe", "how many", "what", "who"]):
            return "answer"
        return "other"

    def extract_slots_simple(self, text: str) -> dict:
        """
        A simple slot extractor:
          - person: first match in known names
          - destination: first match in rooms
          - object: longest matching object phrase
        """
        t = text.lower()

        # person
        person = ""
        for n in self.names:
            if re.search(r"\b" + re.escape(n) + r"\b", t):
                person = n.capitalize()
                break

        # destination
        destination = ""
        for r in self.rooms:
            if r in t:
                destination = r
                break

        # object (prefer longest match)
        obj = ""
        for o in sorted(self.objects, key=len, reverse=True):
            if o in t:
                obj = o
                break

        return {
            "person": person,
            "destination": destination,
            "object": obj,
        }

    def infer_intent_name(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["guide", "escort", "lead"]):
            return "guide_person"
        if any(k in t for k in ["bring", "fetch", "deliver", "take", "give"]):
            return "bring_object"
        if any(k in t for k in ["how many", "count", "tell me how many", "describe", "what", "who"]):
            return "answer_question"
        return "unknown"

    def publish(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)

    def on_text(self, msg: String):
        snapped = " ".join((msg.data or "").strip().split())
        if not snapped:
            return

        conf = self.latest_conf

        intent_name = self.infer_intent_name(snapped)
        intent_type = self.coarse_intent_type(intent_name)
        slots = self.extract_slots_simple(snapped)

        # Confidence gating (optional)
        if self.confirm_threshold >= 0.0 and conf is not None and conf < self.confirm_threshold:
            payload = {
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "slots": {},
                "raw_text": snapped,
                "confidence": float(conf),
                "source": "intent",
                "error": "need_confirm_low_confidence",
            }
            self.publish(payload)
            rospy.logwarn("gpsr_intent_node: need_confirm conf=%.3f < %.3f text='%s'", conf, self.confirm_threshold, snapped)
            return

        # Normal publish
        payload = {
            "schema": "gpsr_intent_v1",
            "ok": True if intent_name != "unknown" else False,
            "need_confirm": False if intent_name != "unknown" else True,
            "intent_type": intent_type if intent_name != "unknown" else "other",
            "slots": slots if intent_name != "unknown" else {},
            "raw_text": snapped,
            "confidence": float(conf) if conf is not None else None,
            "source": "intent",
            "intent_name": intent_name,
        }
        if intent_name == "unknown":
            payload["error"] = "unknown_intent"

        self.publish(payload)
        rospy.loginfo("gpsr_intent_node: intent=%s type=%s slots=%s", intent_name, payload["intent_type"], payload["slots"])


def main():
    rospy.init_node("gpsr_intent_node")
    _ = GPSRIntentNode()
    rospy.loginfo("gpsr_intent_node started (unified schema).")
    rospy.spin()


if __name__ == "__main__":
    main()
