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
    - Parses when utterance_end(True) arrives
    - Publishes unified intent schema: gpsr_intent_v1 as std_msgs/String(JSON)

    Key guarantees (v1):
      - steps: [{action, args}] (args unified; accepts old 'fields' internally)
      - slots: fixed keys (source_room/source_place/destination_room/destination_place/object/object_category/person...)
      - command_kind preserved (from parser 'kind')
    """

    # ----------------------------
    # Fixed schema keys
    # ----------------------------
    FIXED_SLOT_KEYS = [
        "object",
        "object_category",
        "quantity",

        "source_room",
        "source_place",
        "destination_room",
        "destination_place",

        "person",
        "person_at_source",
        "person_at_destination",

        "attribute",
        "question_type",
        "gesture",
    ]

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
            "gpsr_parser_node(v1): text=%s utterance_end=%s conf=%s -> intent=%s",
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
    @staticmethod
    def _normalize_text(s: str) -> str:
        s = (s or "").strip()
        # minimal normalization (keep it conservative)
        s = " ".join(s.split())
        return s.lower()

    @staticmethod
    def _ensure_steps_args(steps):
        """
        Convert old step format {action, fields} into v1 format {action, args}.
        Keep original keys (like fields) out of published steps to stabilize schema.
        """
        out = []
        for s in (steps or []):
            if not isinstance(s, dict):
                continue
            action = s.get("action", "")
            if "args" in s and isinstance(s.get("args"), dict):
                args = dict(s["args"])
            else:
                f = s.get("fields") or {}
                args = dict(f) if isinstance(f, dict) else {}
            out.append({"action": action, "args": args})
        return out

    def _coarse_intent_type(self, command_kind: str, steps_v1: list) -> str:
        """
        Coarse intent type for SMACH branching.
        - If multiple steps -> composite (recommended)
        - Else infer from kind/actions
        """
        if len(steps_v1) >= 2:
            return "composite"

        kind = (command_kind or "").lower()
        actions = " ".join([(s.get("action", "") or "").lower() for s in steps_v1])
        key = (kind + " " + actions).strip()

        if any(k in key for k in ["guide", "escort", "lead"]):
            return "guide"
        if any(k in key for k in ["bring", "fetch", "deliver", "take", "give"]):
            return "bring"
        if any(k in key for k in ["count", "tell", "answer", "describe", "how many", "what", "who"]):
            return "answer"
        if any(k in key for k in ["find"]):
            return "find"
        if any(k in key for k in ["place", "put"]):
            return "place"
        return "other"

    def _empty_fixed_slots(self) -> dict:
        return {k: None for k in self.FIXED_SLOT_KEYS}

    def _extract_fixed_slots_from_steps(self, steps_v1: list) -> dict:
        """
        Build fixed slots from v1 steps.
        Principle: steps are the source of truth; slots are a stable summary.
        """
        slots = self._empty_fixed_slots()

        # helper: first-non-null set
        def set_if_empty(key, val):
            if val is None:
                return
            if isinstance(val, str) and val.strip() == "":
                return
            if slots.get(key) is None:
                slots[key] = val

        for st in steps_v1:
            action = (st.get("action") or "").lower()
            args = st.get("args") or {}

            # Common keys (object/category/person/place/room)
            # We keep these conservative and action-aware.

            # find object in room
            if action in ["find_object_in_room", "findobjinroom", "find_object_room"]:
                set_if_empty("source_room", args.get("room"))
                set_if_empty("object", args.get("object"))
                set_if_empty("object_category", args.get("object_category") or args.get("object_or_category"))

            # bring me object from placement
            if action in ["bring_object_to_operator", "bring_to_operator", "bring_object"]:
                set_if_empty("object", args.get("object"))
                set_if_empty("source_place", args.get("source_place") or args.get("place"))

            # place object on place
            if action in ["place_object_on_place", "put_object_on_place", "place_object"]:
                set_if_empty("destination_place", args.get("place"))

            # go to room/place (sometimes parser emits these)
            if action in ["goto_room", "go_to_room"]:
                set_if_empty("source_room", args.get("room"))
            if action in ["goto_place", "go_to_place", "goto_location"]:
                # if we don't know whether source/destination, prefer destination_place when action comes late
                # but keep conservative: only set destination_place if empty
                set_if_empty("destination_place", args.get("place") or args.get("location"))

            # give/deliver to person
            if action in ["give_object_to_person", "deliver_object_to_person", "give_object"]:
                set_if_empty("person", args.get("person") or args.get("name"))
                set_if_empty("object", args.get("object"))

            # answer/question type
            if action in ["answer_question", "answer", "ask_question", "ask"]:
                set_if_empty("question_type", args.get("question_type"))
                set_if_empty("attribute", args.get("attribute"))

        # Some legacy compatibility (optional)
        # If object_category empty but object_or_category exists somewhere else, we already handled it above.

        return slots

    def _publish(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)

    # ----------------------------
    # Utterance-end driven parse
    # ----------------------------
    def _on_utt_end(self, msg: Bool):
        # Expect utterance_end pulse True
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
                    "slots": self._empty_fixed_slots(),
                    "raw_text": text,
                    "normalized_text": self._normalize_text(text),
                    "confidence": self._latest_conf,
                    "source": "parser",
                    "error": "need_confirm_low_confidence",
                    "command_kind": None,
                    "steps": [],
                    "context": {"lang": "en", "source": "parser"},
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
                "slots": self._empty_fixed_slots(),
                "raw_text": text,
                "normalized_text": self._normalize_text(text),
                "confidence": self._latest_conf,
                "source": "parser",
                "error": "parse_failed",
                "command_kind": None,
                "steps": [],
                "context": {"lang": "en", "source": "parser"},
            }
            self._publish(payload)
            return

        # Parser -> v1 unified schema
        # NOTE: your parser currently provides command.to_json() with keys:
        #   kind, steps[{action, fields}], ...
        try:
            cmd_dict = json.loads(command.to_json())
        except Exception as e:
            rospy.logerr("gpsr_parser_node: command.to_json() decode failed: %s", str(e))
            payload = {
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": False,
                "intent_type": "other",
                "slots": self._empty_fixed_slots(),
                "raw_text": text,
                "normalized_text": self._normalize_text(text),
                "confidence": self._latest_conf,
                "source": "parser",
                "error": "command_json_decode_failed",
                "command_kind": None,
                "steps": [],
                "context": {"lang": "en", "source": "parser"},
            }
            self._publish(payload)
            return

        command_kind = cmd_dict.get("kind")
        steps_v1 = self._ensure_steps_args(cmd_dict.get("steps", []))
        slots_v1 = self._extract_fixed_slots_from_steps(steps_v1)

        payload = {
            "schema": "gpsr_intent_v1",
            "ok": True,
            "need_confirm": False,
            "intent_type": self._coarse_intent_type(command_kind, steps_v1),
            "slots": slots_v1,
            "raw_text": text,
            "normalized_text": self._normalize_text(text),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": command_kind,
            "steps": steps_v1,
            "context": {"lang": "en", "source": "parser"},
        }
        self._publish(payload)


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.loginfo("gpsr_parser_node started (utterance_end driven, gpsr_intent_v1 fixed schema).")
    rospy.spin()


if __name__ == "__main__":
    main()
