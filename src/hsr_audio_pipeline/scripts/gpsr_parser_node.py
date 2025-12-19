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
    """
    GPSR parser node (ROS1 / Noetic)

    - Subscribes to /gpsr/asr/text and /gpsr/asr/utterance_end
    - Parses ONLY when utterance_end(True) arrives
    - Publishes unified intent schema: gpsr_intent_v1 as std_msgs/String(JSON)

    Output JSON schema (gpsr_intent_v1):
      {
        "schema": "gpsr_intent_v1",
        "ok": true/false,
        "need_confirm": true/false,
        "intent_type": "bring|guide|answer|other",
        "slots": {...},
        "raw_text": "...",
        "confidence": 0.0..1.0 or null,
        "source": "parser",
        "error": "...",            # if ok=false
        "command_kind": "...",     # optional
        "steps": [...]             # optional
      }
    """

    def __init__(self):
        # ---- Params ----
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")

        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        # If utterance_end arrives but /gpsr/asr/text timestamp is older than this, ignore.
        self.max_text_age_sec = float(rospy.get_param("~max_text_age_sec", 1.0))

        # Optional confidence gating: if >=0 and confidence is available, low confidence -> need_confirm
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # ---- Latest ASR ----
        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)

        self._latest_conf = None
        self._latest_conf_stamp = rospy.Time(0)

        # ---- Vocabulary (keep consistent with your project) ----
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
        rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)  # optional

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

    # ----------------------------
    # Unified schema helpers
    # ----------------------------
    def _coarse_intent_type_from_command(self, cmd_dict: dict) -> str:
        """Map detailed parser output into coarse intent_type used by SMACH."""
        kind = (cmd_dict.get("kind") or "").lower()
        steps = cmd_dict.get("steps") or []
        actions = " ".join([(s.get("action", "") or "").lower() for s in steps])
        key = (kind + " " + actions).strip()

        if any(k in key for k in ["guide", "escort", "lead"]):
            return "guide"
        if any(k in key for k in ["bring", "fetch", "deliver", "take", "give"]):
            return "bring"
        if any(k in key for k in ["count", "tell", "answer", "describe", "how many", "what", "who"]):
            return "answer"
        return "other"

    def _extract_common_slots(self, cmd_dict: dict) -> dict:
        """
        Merge 'fields' from steps into a flat dict.
        Also ensures SMACH-friendly keys exist: object, destination, person.
        """
        slots = {}
        for s in (cmd_dict.get("steps") or []):
            f = s.get("fields") or {}
            if isinstance(f, dict):
                slots.update(f)

        out = dict(slots)
        out.setdefault("object", out.get("obj") or out.get("item") or out.get("thing") or "")
        out.setdefault("destination", out.get("to") or out.get("dest") or out.get("room") or out.get("location") or "")
        out.setdefault("person", out.get("name") or out.get("target") or out.get("who") or "")
        return out

    def _publish(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)

    # ----------------------------
    # Utterance-end driven parse
    # ----------------------------
    def _on_utt_end(self, msg: Bool):
        # utterance_end pulse True expected
        if not bool(msg.data):
            return

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec()
        text = (self._latest_text or "").strip()

        if (not text) or (age > self.max_text_age_sec):
            rospy.logwarn(
                "gpsr_parser_node: utterance_end but no fresh text (age=%.3f, text='%s')",
                age, text
            )
            return

        # Optional confidence gating
        if self.min_confidence >= 0.0 and self._latest_conf is not None:
            if self._latest_conf < self.min_confidence:
                payload = {
                    "schema": "gpsr_intent_v1",
                    "ok": False,
                    "need_confirm": True,
                    "intent_type": "other",
                    "slots": {},
                    "raw_text": text,
                    "confidence": self._latest_conf,
                    "source": "parser",
                    "error": "need_confirm_low_confidence",
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
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": False,
                "intent_type": "other",
                "slots": {},
                "raw_text": text,
                "confidence": self._latest_conf,
                "source": "parser",
                "error": "parse_failed",
            }
            self._publish(payload)
            return

        # Parser -> unified schema
        cmd_dict = json.loads(command.to_json())
        payload = {
            "schema": "gpsr_intent_v1",
            "ok": True,
            "need_confirm": False,
            "intent_type": self._coarse_intent_type_from_command(cmd_dict),
            "slots": self._extract_common_slots(cmd_dict),
            "raw_text": text,
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": cmd_dict.get("kind"),
            "steps": cmd_dict.get("steps", []),
        }
        self._publish(payload)


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.loginfo("gpsr_parser_node started (utterance_end driven, unified schema).")
    rospy.spin()


if __name__ == "__main__":
    main()
