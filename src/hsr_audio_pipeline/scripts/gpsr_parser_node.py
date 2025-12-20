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

Goal:
  - Publish a consistent "gpsr_intent_v1" JSON schema.
  - Be robust: parser may return dict / JSON str / object (e.g., GpsrIntentV1).
  - Convert legacy steps {action, fields} -> v1 steps {action, args}.
  - Convert legacy/variable slots into fixed-key slots, and keep extras.legacy_*.

Notes:
  - This node assumes /gpsr/asr/utterance_end is a pulse. It triggers parsing on True.
  - If your utterance_end is True-only, edge-driven parsers may not trigger; you already fixed ASR to True->False.
"""

import json
import rospy
from std_msgs.msg import String, Bool, Float32

# Your parser module
from gpsr_parser import GpsrParser


class GpsrParserNode:
    FIXED_SLOT_KEYS = [
        # object / category
        "object",
        "object_category",
        "quantity",

        # locations
        "source_room",
        "source_place",
        "destination_room",
        "destination_place",

        # people
        "person",
        "person_at_source",
        "person_at_destination",

        # question / attributes
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

        # ---- build parser ----
        # If your gpsr_parser.py defines __init__ with vocab args, keep it.
        # If it doesn't, this still works because python will throw early, making it obvious.
        self.parser = GpsrParser()

        # ---- pubs/subs ----
        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)

        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utterance_end, queue_size=50)

        # confidence is optional; subscribe but tolerate missing publisher
        try:
            rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)
        except Exception:
            pass

        rospy.loginfo("gpsr_parser_node ready: text=%s utt_end=%s conf=%s -> intent=%s",
                      self.text_topic, self.utt_end_topic, self.conf_topic, self.intent_topic)

    # -------------------------------
    # callbacks
    # -------------------------------
    def _on_text(self, msg: String):
        self._latest_text = (msg.data or "")
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)
        self._latest_conf_stamp = rospy.Time.now()

    def _on_utterance_end(self, msg: Bool):
        # Trigger only on True pulse
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

        # Optional confidence gating
        if self.min_confidence >= 0.0 and self._latest_conf is not None:
            if self._latest_conf < self.min_confidence:
                payload = self._make_base_v1(raw_text)
                payload["ok"] = False
                payload["need_confirm"] = True
                payload["error"] = "low_confidence"
                payload["confidence"] = self._latest_conf
                self._publish(payload)
                rospy.logwarn("low confidence %.3f < %.3f -> need_confirm", self._latest_conf, self.min_confidence)
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
            rospy.logerr("parser exception: %s", str(e))
            return

        # Normalize whatever 'parsed' is into a dict
        payload = self._ensure_dict(parsed, raw_text)

        # Force schema to gpsr_intent_v1, and normalize into fixed schema
        payload = self._coerce_to_v1(payload, raw_text)

        self._publish(payload)

    # -------------------------------
    # dict coercion / schema normalize
    # -------------------------------
    def _ensure_dict(self, maybe, raw_text: str) -> dict:
        """
        Accept:
          - dict
          - JSON string
          - object with to_dict(), to_json_str(), to_json()
          - arbitrary object (fallback to __dict__)
        """
        if isinstance(maybe, dict):
            return maybe

        # JSON string?
        if isinstance(maybe, str):
            s = maybe.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    return json.loads(s)
                except Exception:
                    pass

        # object conversions
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

        # last resort
        if hasattr(maybe, "__dict__"):
            d = dict(maybe.__dict__)
            if d:
                return d

        # fallback error dict
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
            "slots": self._empty_fixed_slots(),
            "steps": [],
            "extras": {},
            "context": {"lang": self.lang, "source": "parser"},
        }

    def _coerce_to_v1(self, payload: dict, raw_text: str) -> dict:
        """
        Convert legacy outputs into stable v1:
          - steps: [{action, fields}] -> [{action, args}]
          - slots: fixed keys; keep legacy slots under extras.legacy_slots
          - command_kind: from payload['command_kind'] or payload['kind']
        """
        v1 = self._make_base_v1(raw_text)

        # Basic flags
        v1["ok"] = bool(payload.get("ok", True))
        v1["need_confirm"] = bool(payload.get("need_confirm", False))
        v1["source"] = payload.get("source", "parser")

        # Confidence may be in payload or cached
        v1["confidence"] = payload.get("confidence", self._latest_conf)

        # command_kind (legacy: 'kind')
        v1["command_kind"] = payload.get("command_kind", payload.get("kind"))

        # intent_type:
        # - prefer payload's intent_type if present
        # - else infer coarsely from steps/kind
        given_intent = payload.get("intent_type")
        # Steps normalization
        steps_legacy = payload.get("steps", []) or []
        steps_v1 = self._normalize_steps(steps_legacy)
        v1["steps"] = steps_v1

        if isinstance(given_intent, str) and given_intent.strip():
            v1["intent_type"] = given_intent
        else:
            v1["intent_type"] = self._infer_intent_type(v1["command_kind"], steps_v1)

        # slots:
        legacy_slots = payload.get("slots", {}) if isinstance(payload.get("slots"), dict) else {}
        fixed_slots = self._extract_fixed_slots(steps_v1, legacy_slots)
        v1["slots"] = fixed_slots

        # extras: keep legacy stuff
        v1["extras"] = {
            "legacy_slots": legacy_slots,
        }

        # Preserve any parser-provided raw_text/normalized_text if present
        v1["raw_text"] = payload.get("raw_text", raw_text)
        v1["normalized_text"] = payload.get("normalized_text", self._normalize_text(v1["raw_text"]))

        # Some parsers already include schema; enforce ours
        v1["schema"] = "gpsr_intent_v1"
        return v1

    def _normalize_steps(self, steps_legacy) -> list:
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

    def _empty_fixed_slots(self) -> dict:
        return {k: None for k in self.FIXED_SLOT_KEYS}

    def _extract_fixed_slots(self, steps_v1: list, legacy_slots: dict) -> dict:
        """
        Fill stable slots from steps first (source of truth), then fall back to legacy_slots.
        """
        slots = self._empty_fixed_slots()

        def set_if_empty(key, val):
            if val is None:
                return
            if isinstance(val, str) and val.strip() == "":
                return
            if slots.get(key) is None:
                slots[key] = val

        # --- derive from steps ---
        for st in steps_v1:
            action = (st.get("action") or "").lower()
            args = st.get("args") or {}

            # Find in room
            if action in ("find_object_in_room", "find_object_room", "findobjinroom"):
                set_if_empty("source_room", args.get("room"))
                set_if_empty("object", args.get("object"))
                set_if_empty("object_category", args.get("object_category") or args.get("object_or_category"))

            # Go to location/room/place
            if action in ("go_to_location", "goto_location", "go_to_room", "goto_room"):
                # Many implementations put 'room' here
                set_if_empty("source_room", args.get("room") or args.get("location"))
            if action in ("go_to_place", "goto_place"):
                set_if_empty("source_place", args.get("place") or args.get("location"))

            # Bring from place
            if action in ("bring_object_to_operator", "bring_to_operator"):
                set_if_empty("object", args.get("object"))
                set_if_empty("source_place", args.get("source_place") or args.get("place"))

            # Deliver to person in room
            if action in ("deliver_object_to_person_in_room", "deliver_object_to_person", "give_object_to_person"):
                set_if_empty("destination_room", args.get("room"))
                set_if_empty("person", args.get("person") or args.get("name") or args.get("person_filter"))
                set_if_empty("object", args.get("object"))

            # Guide
            if action in ("guide_named_person_from_place_to_place", "guide_person_from_place_to_place"):
                set_if_empty("person", args.get("name") or args.get("person"))
                set_if_empty("source_place", args.get("from_place"))
                set_if_empty("destination_place", args.get("to_place"))

            # Place object
            if action in ("place_object_on_place", "place_object", "put_object_on_place"):
                set_if_empty("destination_place", args.get("place"))
                set_if_empty("object", args.get("object"))

            # Q/A about object property on place
            if action in ("tell_object_property_on_place", "tell_object_property", "answer_question"):
                set_if_empty("comparison", args.get("comparison"))
                set_if_empty("source_place", args.get("place"))
                set_if_empty("question_type", args.get("question_type"))

        # --- fall back from legacy slots (non-fixed info is kept in extras.legacy_slots) ---
        # common legacy key mappings:
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

    def _infer_intent_type(self, kind, steps_v1) -> str:
        # If multiple steps, call it composite for easy SMACH routing
        if isinstance(steps_v1, list) and len(steps_v1) >= 2:
            return "composite"

        k = (kind or "").lower()
        actions = " ".join([(s.get("action", "") or "").lower() for s in (steps_v1 or [])])

        s = (k + " " + actions).strip()
        if any(w in s for w in ["guide", "escort", "lead"]):
            return "guide"
        if any(w in s for w in ["bring", "fetch", "deliver", "give"]):
            return "bring"
        if any(w in s for w in ["tell", "answer", "question", "count", "describe"]):
            return "answer"
        if any(w in s for w in ["find"]):
            return "find"
        if any(w in s for w in ["place", "put"]):
            return "place"
        return "other"

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
    rospy.loginfo("gpsr_parser_node started (robust v1 publisher).")
    rospy.spin()


if __name__ == "__main__":
    main()
