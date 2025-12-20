#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_parser_node.py (patched)

Goal:
- Publish a *fixed* gpsr_intent_v1 JSON on /gpsr/intent.
- Keep your existing gpsr_parser.py logic intact (it may output legacy keys/steps.fields).
- Convert legacy output -> v1:
    - steps: {action, args}  (convert from fields if needed)
    - slots: fixed keys (object/object_category/source_room/source_place/destination_room/destination_place/person...)
    - keep extra legacy fields under "extras" so nothing is lost.

Trigger:
- Parse only when /gpsr/asr/utterance_end == True (pulse recommended).

"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

import json
import rospy
from std_msgs.msg import String, Bool, Float32

from gpsr_parser import GpsrParser


class GpsrParserNode:
    # ---- fixed slot schema (stable across all templates) ----
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
        # Topics
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        # Freshness window: utterance_end must be close to last text
        self.max_text_age_sec = float(rospy.get_param("~max_text_age_sec", 2.0))

        # Optional confidence gating
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # Latest ASR cache
        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf = None
        self._latest_conf_stamp = rospy.Time(0)

        # Vocabulary (keep same as your current node – adjust here if you maintain elsewhere)
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

        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)

        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)
        rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)

        rospy.loginfo("gpsr_parser_node(patched_v1): %s + %s -> %s",
                      self.text_topic, self.utt_end_topic, self.intent_topic)

    def _on_text(self, msg: String):
        self._latest_text = msg.data or ""
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)
        self._latest_conf_stamp = rospy.Time.now()

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = (s or "").strip()
        s = " ".join(s.split())
        return s.lower()

    def _empty_fixed_slots(self) -> dict:
        return {k: None for k in self.FIXED_SLOT_KEYS}

    @staticmethod
    def _ensure_steps_args(steps):
        """Legacy {action, fields} -> v1 {action, args}"""
        out = []
        for st in steps or []:
            if not isinstance(st, dict):
                continue
            action = st.get("action", "")
            args = st.get("args")
            if not isinstance(args, dict):
                fields = st.get("fields")
                args = fields if isinstance(fields, dict) else {}
            out.append({"action": action, "args": dict(args)})
        return out

    def _extract_slots_from_steps(self, steps_v1: list) -> dict:
        """
        Build fixed slots from v1 steps.
        Conservative mapping: only fill when confident.
        """
        slots = self._empty_fixed_slots()

        def put(key, val):
            if val is None:
                return
            if isinstance(val, str) and not val.strip():
                return
            if slots.get(key) is None:
                slots[key] = val

        for st in steps_v1:
            action = (st.get("action") or "").lower()
            a = st.get("args") or {}

            # find / go
            if action in ["find_object_in_room", "findobjinroom"]:
                put("source_room", a.get("room"))
                put("object", a.get("object"))
                put("object_category", a.get("object_category") or a.get("object_or_category"))

            if action in ["go_to_location", "goto_room", "go_to_room"]:
                # may contain room or place; try room first
                put("source_room", a.get("room"))

            if action in ["go_to_place", "goto_place", "goto_location"]:
                # ambiguous; treat as destination_place if not set
                put("destination_place", a.get("place") or a.get("location"))

            # bring
            if action in ["bring_object_to_operator", "bring_to_operator", "bring_object"]:
                put("object", a.get("object"))
                put("source_place", a.get("source_place") or a.get("place"))

            # take
            if action in ["take_object", "pick_object", "pickup_object"]:
                # could optionally set object if provided
                put("object", a.get("object") or a.get("target_object"))

            # place
            if action in ["place_object_on_place", "put_object_on_place", "place_object"]:
                put("destination_place", a.get("place"))

            # deliver / give
            if action in ["deliver_object_to_person_in_room", "deliver_object_to_person", "give_object_to_person"]:
                put("destination_room", a.get("room"))
                put("person", a.get("person") or a.get("name"))
                put("person_at_destination", a.get("person") or a.get("name"))
                # Some templates use person_filter
                put("attribute", a.get("person_filter"))

            # guide
            if action in ["guide_named_person_from_place_to_place", "guide_person_from_place_to_place"]:
                put("person", a.get("name"))
                put("source_place", a.get("from_place"))
                put("destination_place", a.get("to_place"))

            # tell/answer
            if action in ["tell_object_property_on_place", "tell_obj_prop_on_place"]:
                put("destination_place", a.get("place"))  # question refers to place
                # comparison/prop is not in fixed slots; store in extras later

        return slots

    def _coarse_intent_type(self, command_kind: str, steps_v1: list) -> str:
        """Prefer composite when multi-step; otherwise infer from kind/actions."""
        if len(steps_v1) >= 2:
            return "composite"

        k = (command_kind or "").lower()
        actions = " ".join([(s.get("action", "") or "").lower() for s in steps_v1])
        key = (k + " " + actions).strip()

        if any(w in key for w in ["guide"]):
            return "guide"
        if any(w in key for w in ["tell", "answer", "describe", "count", "question"]):
            return "answer"
        if any(w in key for w in ["bring", "deliver", "give", "fetch"]):
            return "bring"
        if any(w in key for w in ["find"]):
            return "find"
        if any(w in key for w in ["place", "put"]):
            return "place"
        return "other"

    def _publish(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)

    def _on_utt_end(self, msg: Bool):
        # Trigger on True pulse
        if not bool(msg.data):
            return

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec()
        text = (self._latest_text or "").strip()

        if not text or age > self.max_text_age_sec:
            rospy.logwarn("utterance_end but no fresh text (age=%.3f)", age)
            return

        # confidence gating
        if self.min_confidence >= 0.0 and self._latest_conf is not None and self._latest_conf < self.min_confidence:
            payload = {
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "raw_text": text,
                "normalized_text": self._normalize_text(text),
                "confidence": {"asr": self._latest_conf, "parser": None},
                "source": "parser",
                "command_kind": None,
                "slots": self._empty_fixed_slots(),
                "steps": [],
                "extras": {"error": "low_confidence"},
            }
            self._publish(payload)
            return

        rospy.loginfo("parse: %s", text)
        cmd = self.parser.parse(text)
        if cmd is None:
            payload = {
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "raw_text": text,
                "normalized_text": self._normalize_text(text),
                "confidence": {"asr": self._latest_conf, "parser": None},
                "source": "parser",
                "command_kind": None,
                "slots": self._empty_fixed_slots(),
                "steps": [],
                "extras": {"error": "parse_failed"},
            }
            self._publish(payload)
            return

        # If parser already has v1 method, use it
        if hasattr(cmd, "to_intent_v1"):
            payload = cmd.to_intent_v1()
            # Ensure minimal fields exist
            payload.setdefault("schema", "gpsr_intent_v1")
            payload.setdefault("raw_text", text)
            payload.setdefault("normalized_text", self._normalize_text(text))
            payload.setdefault("confidence", {"asr": self._latest_conf, "parser": None})
            self._publish(payload)
            return

        # Legacy parser json
        try:
            legacy = json.loads(cmd.to_json())
        except Exception as e:
            payload = {
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "raw_text": text,
                "normalized_text": self._normalize_text(text),
                "confidence": {"asr": self._latest_conf, "parser": None},
                "source": "parser",
                "command_kind": None,
                "slots": self._empty_fixed_slots(),
                "steps": [],
                "extras": {"error": "legacy_json_decode_failed", "detail": str(e)},
            }
            self._publish(payload)
            return

        command_kind = legacy.get("kind") or legacy.get("command_kind")
        steps_v1 = self._ensure_steps_args(legacy.get("steps", []))
        slots_v1 = self._extract_slots_from_steps(steps_v1)

        # Preserve legacy slots/fields that are not in the fixed set
        extras = {}
        if isinstance(legacy.get("slots"), dict):
            extras["legacy_slots"] = legacy["slots"]
        extras["legacy"] = {k: legacy.get(k) for k in ["kind", "command_kind"] if k in legacy}

        payload = {
            "schema": "gpsr_intent_v1",
            "ok": True,
            "need_confirm": False,
            "intent_type": self._coarse_intent_type(command_kind, steps_v1),
            "raw_text": text,
            "normalized_text": self._normalize_text(text),
            "confidence": {"asr": self._latest_conf, "parser": None},
            "source": "parser",
            "command_kind": command_kind,
            "slots": slots_v1,
            "steps": steps_v1,
            "extras": extras,
        }
        self._publish(payload)


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.loginfo("gpsr_parser_node(patched_v1) started.")
    rospy.spin()


if __name__ == "__main__":
    main()
