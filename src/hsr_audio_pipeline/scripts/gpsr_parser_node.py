#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_parser_node.py (ROS1 / Noetic)

Sub:
  - ~text_topic          (std_msgs/String)  default: /gpsr/asr/text
  - ~utterance_end_topic (std_msgs/Bool)    default: /gpsr/asr/utterance_end
  - ~confidence_topic    (std_msgs/Float32) default: /gpsr/asr/confidence (optional)

Pub:
  - ~intent_topic        (std_msgs/String)  default: /gpsr/intent   (JSON)

This node:
  - Triggers parse on utterance_end(True)
  - Calls gpsr_parser.GpsrParser (requires 7 vocab args)
  - Publishes fixed schema gpsr_intent_v1:
      steps: [{action, args}] (legacy fields -> args)
      slots: fixed keys with null when absent
      extras.legacy_slots preserves original variable slots
  - Robust: parser may return dict / JSON str / object; always normalizes to dict.

Additions (2025-12):
  - Map greet_person_with_clothes_in_room -> slots.source_room, slots.attribute
  - Map guide_person_to_dest (args empty) -> infer destination_place from raw_text
"""

import os
import sys
import json

# ---- PATH FIX (CRITICAL) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Add "<pkg>/scripts" even when run via catkin devel wrapper
try:
    import rospkg
    _rp = rospkg.RosPack()
    _pkg_dir = _rp.get_path("hsr_audio_pipeline")
    _scripts_dir = os.path.join(_pkg_dir, "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
except Exception:
    pass
# -----------------------------

import rospy
from std_msgs.msg import String, Bool, Float32

from gpsr_parser import GpsrParser


class GpsrParserNode:
    # fixed slots for gpsr_intent_v1
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
        "question_type",
        "attribute",
        "comparison",
        "gesture",
    ]

    def __init__(self):
        # ---- params ----
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.max_text_age_sec = float(rospy.get_param("~max_text_age_sec", 2.0))
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))  # <0 disables gating
        self.lang = rospy.get_param("~lang", "en")

        # ---- latest ASR caches ----
        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf = None
        self._latest_conf_stamp = rospy.Time(0)

        # ---- vocab (defaults; can be overridden by ROS params) ----
        default_person_names = [
            "Adel", "Angel", "Axel", "Charlie", "Jane",
            "Jules", "Morgan", "Paris", "Robin", "Simone",
        ]

        default_room_names = [
            "bedroom", "kitchen", "living room", "office", "bathroom"
        ]

        default_location_names = [
            "bed", "bedside table", "shelf", "trashbin", "dishwasher", "potted plant",
            "kitchen table", "chairs", "pantry", "refrigerator", "sink", "cabinet",
            "coatrack", "desk", "armchair", "desk lamp", "waste basket", "tv stand",
            "storage rack", "lamp", "side tables", "sofa", "bookshelf", "entrance", "exit",
        ]

        default_placement_location_names = [
            "bed", "bedside table", "shelf", "dishwasher",
            "kitchen table", "pantry", "refrigerator", "sink",
            "cabinet", "desk", "tv stand", "storage rack",
            "side tables", "sofa", "bookshelf",
        ]

        default_object_names = [
            "juice pack", "cola", "milk", "orange juice", "tropical juice",
            "red wine", "iced tea",
            "tennis ball", "rubiks cube", "baseball", "soccer ball", "dice",
            "orange", "pear", "peach", "strawberry", "apple", "lemon", "banana", "plum",
            "cornflakes", "pringles", "cheezit",
            "cup", "bowl", "fork", "plate", "knife", "spoon",
            "chocolate jello", "coffee grounds", "mustard", "tomato soup",
            "tuna", "strawberry jello", "spam", "sugar",
            "cleanser", "sponge",
        ]

        default_cat_singular = [
            "drink", "toy", "fruit", "snack", "dish", "food", "cleaning supply"
        ]
        default_cat_plural = [
            "drinks", "toys", "fruits", "snacks", "dishes", "food", "cleaning supplies"
        ]

        # read ROS params if provided
        person_names = rospy.get_param("~person_names", default_person_names)
        location_names = rospy.get_param("~location_names", default_location_names)
        placement_location_names = rospy.get_param("~placement_location_names", default_placement_location_names)
        room_names = rospy.get_param("~room_names", default_room_names)
        object_names = rospy.get_param("~object_names", default_object_names)
        object_categories_plural = rospy.get_param("~object_categories_plural", default_cat_plural)
        object_categories_singular = rospy.get_param("~object_categories_singular", default_cat_singular)

        # Keep vocab for slot補完（raw_textからplace推定などに使う）
        self._location_names = location_names
        self._placement_location_names = placement_location_names
        # place候補（lower/strip済み、重複除去）
        self._all_places = list(
            set([s.lower().strip() for s in (location_names + placement_location_names) if isinstance(s, str)])
        )

        # ---- build parser (REQUIRED ARGS) ----
        self.parser = GpsrParser(
            person_names=person_names,
            location_names=location_names,
            placement_location_names=placement_location_names,
            room_names=room_names,
            object_names=object_names,
            object_categories_plural=object_categories_plural,
            object_categories_singular=object_categories_singular,
        )

        # ---- ROS pub/sub ----
        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)
        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utterance_end, queue_size=50)

        # confidence is optional
        try:
            rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)
        except Exception:
            pass

        rospy.loginfo("gpsr_parser_node started: text=%s utt_end=%s -> intent=%s",
                      self.text_topic, self.utt_end_topic, self.intent_topic)

    # ---------------- callbacks ----------------
    def _on_text(self, msg: String):
        self._latest_text = (msg.data or "")
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)
        self._latest_conf_stamp = rospy.Time.now()

    def _on_utterance_end(self, msg: Bool):
        if not bool(msg.data):
            return

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec()
        raw_text = (self._latest_text or "").strip()

        if not raw_text:
            rospy.logwarn("utterance_end received but no text cached.")
            return
        if age > self.max_text_age_sec:
            rospy.logwarn("utterance_end received but text is stale (age=%.3fs): %r", age, raw_text)
            return

        # optional confidence gating
        if self.min_confidence >= 0.0 and self._latest_conf is not None:
            if self._latest_conf < self.min_confidence:
                payload = self._make_base_v1(raw_text)
                payload["ok"] = False
                payload["need_confirm"] = True
                payload["error"] = "low_confidence"
                payload["confidence"] = self._latest_conf
                self._publish(payload)
                return

        rospy.loginfo("parse: %s", raw_text)

        try:
            parsed = self.parser.parse(raw_text)
        except Exception as e:
            payload = self._make_base_v1(raw_text)
            payload["ok"] = False
            payload["need_confirm"] = True
            payload["error"] = f"parser_exception: {e}"
            payload["confidence"] = self._latest_conf
            self._publish(payload)
            return

        payload_any = self._ensure_dict(parsed, raw_text)
        payload_v1 = self._coerce_to_v1(payload_any, raw_text)
        self._publish(payload_v1)

    # ---------------- schema helpers ----------------
    def _ensure_dict(self, maybe, raw_text: str) -> dict:
        """
        Accept:
          - dict
          - JSON string
          - object with to_dict()/to_json_str()/to_json()
          - generic object (fallback to __dict__)
        """
        if isinstance(maybe, dict):
            return maybe

        if isinstance(maybe, str):
            s = maybe.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    return json.loads(s)
                except Exception:
                    pass

        for meth in ("to_dict", "to_json_str", "to_json"):
            if hasattr(maybe, meth):
                try:
                    out = getattr(maybe, meth)()
                    if isinstance(out, dict):
                        return out
                    if isinstance(out, str):
                        return json.loads(out)
                except Exception:
                    pass

        if hasattr(maybe, "__dict__"):
            d = dict(maybe.__dict__)
            if d:
                return d

        return {
            "schema": "gpsr_intent_v1",
            "ok": False,
            "need_confirm": True,
            "intent_type": "other",
            "raw_text": raw_text,
            "error": "parsed_value_not_convertible_to_dict",
            "source": "parser",
            "steps": [],
            "slots": {},
        }

    def _make_base_v1(self, raw_text: str) -> dict:
        return {
            "schema": "gpsr_intent_v1",
            "ok": True,
            "need_confirm": False,
            "intent_type": "other",
            "raw_text": raw_text,
            "normalized_text": self._normalize_text(raw_text),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": None,
            "slots": {k: None for k in self.FIXED_SLOT_KEYS},
            "steps": [],
            "extras": {},
            "context": {"lang": self.lang, "source": "parser"},
        }

    def _normalize_steps(self, steps_legacy) -> list:
        """legacy: {action, fields} -> v1: {action, args}"""
        out = []
        if not isinstance(steps_legacy, list):
            return out
        for st in steps_legacy:
            if not isinstance(st, dict):
                continue
            action = st.get("action") or ""
            if not action:
                continue
            if isinstance(st.get("args"), dict):
                args = dict(st["args"])
            elif isinstance(st.get("fields"), dict):
                args = dict(st["fields"])
            else:
                args = {}
            out.append({"action": action, "args": args})
        return out

    def _infer_intent_type(self, kind, steps_v1) -> str:
        # composite if multiple steps
        if isinstance(steps_v1, list) and len(steps_v1) >= 2:
            return "composite"
        k = (kind or "").lower()
        actions = " ".join([(s.get("action", "") or "").lower() for s in (steps_v1 or [])])
        s = (k + " " + actions).strip()
        if any(w in s for w in ["guide", "escort", "lead"]):
            return "guide"
        if any(w in s for w in ["bring", "fetch", "deliver", "give", "takeobj", "bringme"]):
            return "bring"
        if any(w in s for w in ["tell", "answer", "question", "count", "describe", "prop"]):
            return "answer"
        if "find" in s:
            return "find"
        if any(w in s for w in ["place", "put"]):
            return "place"
        return "other"

    def _best_place_from_text(self, raw_text: str):
        """
        raw_textから既知place候補を最長一致で推定する（ガイド先などの補完用）
        - "waste basket" vs "waistbasket" のような表記ゆれを拾うため、
          空白除去版でも照合する。
        """
        if not raw_text:
            return None
        t = self._normalize_text(raw_text)
        t_compact = t.replace(" ", "")

        best = None
        best_len = -1

        # 1) 通常の包含一致（語彙が閉じている前提）
        for p in self._all_places:
            if not p:
                continue
            if p in t:
                if len(p) > best_len:
                    best = p
                    best_len = len(p)

        # 2) 空白除去版でも照合（waistbasket 対策）
        for p in self._all_places:
            if not p:
                continue
            p_compact = p.replace(" ", "")
            if p_compact and p_compact in t_compact:
                if len(p) > best_len:
                    best = p
                    best_len = len(p)

        return best

    def _extract_fixed_slots(self, steps_v1: list, legacy_slots: dict, raw_text: str = "") -> dict:
        """
        Fill fixed slots primarily from steps (truth), then fallback from legacy_slots.
        raw_text は guide_person_to_dest など args空の補完に使用する。
        """
        slots = {k: None for k in self.FIXED_SLOT_KEYS}

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

            # object-property question on place
            if action in ("tell_object_property_on_place", "tellobjproponplcmt", "tell_object_property"):
                set_if_empty("comparison", args.get("comparison"))
                set_if_empty("source_place", args.get("place"))

            # NEW: greet_person_with_clothes_in_room -> source_room, attribute
            if action in ("greet_person_with_clothes_in_room", "greetclothdscinrm"):
                set_if_empty("source_room", args.get("room"))
                set_if_empty("attribute", args.get("clothes_description") or args.get("clothes") or args.get("attribute"))

            # go to location/room
            if action in ("go_to_location", "goto_location", "go_to_room", "goto_room"):
                set_if_empty("source_room", args.get("room") or args.get("location"))

            # find object in room
            if action in ("find_object_in_room", "findobjinroom"):
                set_if_empty("source_room", args.get("room"))
                set_if_empty("object", args.get("object"))
                set_if_empty("object_category", args.get("object_category") or args.get("object_or_category"))

            # bring to operator
            if action in ("bring_object_to_operator", "bringmeobjfromplcmt", "bring_to_operator"):
                set_if_empty("object", args.get("object"))
                set_if_empty("source_place", args.get("source_place") or args.get("place"))

            # deliver to person in room
            if action in ("deliver_object_to_person_in_room", "deliver_object_to_person", "give_object_to_person"):
                set_if_empty("destination_room", args.get("room"))
                set_if_empty("person", args.get("person") or args.get("name") or args.get("person_filter"))
                set_if_empty("object", args.get("object"))

            # guide named person with explicit from/to
            if action in ("guide_named_person_from_place_to_place", "guidenamefrombeactobeac"):
                set_if_empty("person", args.get("name") or args.get("person"))
                set_if_empty("source_place", args.get("from_place"))
                set_if_empty("destination_place", args.get("to_place"))

            # NEW: guide_person_to_dest (args empty) -> infer destination_place from raw_text
            if action in ("guide_person_to_dest", "guide_person_to_destination"):
                set_if_empty("destination_place", args.get("to_place") or args.get("destination_place") or args.get("place"))
                if slots.get("destination_place") is None:
                    inferred = self._best_place_from_text(raw_text)
                    set_if_empty("destination_place", inferred)

            # place object
            if action in ("place_object_on_place", "place_object", "put_object_on_place"):
                set_if_empty("destination_place", args.get("place"))

        # ---- fallback mappings from legacy slots ----
        if isinstance(legacy_slots, dict):
            set_if_empty("object", legacy_slots.get("object"))
            set_if_empty("object_category", legacy_slots.get("object_category") or legacy_slots.get("object_or_category"))
            set_if_empty("source_room", legacy_slots.get("source_room") or legacy_slots.get("room"))
            set_if_empty("source_place", legacy_slots.get("source_place") or legacy_slots.get("place") or legacy_slots.get("from_place"))
            set_if_empty("destination_room", legacy_slots.get("destination_room") or legacy_slots.get("destination"))
            set_if_empty("destination_place", legacy_slots.get("destination_place") or legacy_slots.get("to_place"))
            set_if_empty("person", legacy_slots.get("person") or legacy_slots.get("name") or legacy_slots.get("person_filter"))
            set_if_empty("comparison", legacy_slots.get("comparison"))
            set_if_empty("attribute", legacy_slots.get("attribute"))
            set_if_empty("question_type", legacy_slots.get("question_type"))

        return slots

    def _coerce_to_v1(self, payload: dict, raw_text: str) -> dict:
        v1 = self._make_base_v1(raw_text)

        v1["ok"] = bool(payload.get("ok", True))
        v1["need_confirm"] = bool(payload.get("need_confirm", False))
        v1["source"] = payload.get("source", "parser")
        v1["confidence"] = payload.get("confidence", self._latest_conf)

        # command_kind (legacy: kind)
        v1["command_kind"] = payload.get("command_kind", payload.get("kind"))

        # steps normalize
        steps_v1 = self._normalize_steps(payload.get("steps", []) or [])
        v1["steps"] = steps_v1

        # intent_type
        given_intent = payload.get("intent_type")
        if isinstance(given_intent, str) and given_intent.strip():
            v1["intent_type"] = given_intent
        else:
            v1["intent_type"] = self._infer_intent_type(v1["command_kind"], steps_v1)

        # keep parser raw if exists (do this before slots補完に渡す raw_text を確定)
        v1["raw_text"] = payload.get("raw_text", raw_text)
        v1["normalized_text"] = payload.get("normalized_text", self._normalize_text(v1["raw_text"]))

        # slots fixed + extras legacy
        legacy_slots = payload.get("slots", {}) if isinstance(payload.get("slots"), dict) else {}
        v1["slots"] = self._extract_fixed_slots(steps_v1, legacy_slots, raw_text=v1["raw_text"])
        v1["extras"] = {"legacy_slots": legacy_slots}

        v1["schema"] = "gpsr_intent_v1"
        return v1

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = (s or "").strip()
        s = " ".join(s.split())
        return s.lower()

    def _publish(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.loginfo("gpsr_parser_node running (gpsr_intent_v1 fixed schema).")
    rospy.spin()


if __name__ == "__main__":
    main()
