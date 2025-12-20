#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_parser_node.py (ROS1 / Noetic)

- Subscribes:
    ~text_topic (std_msgs/String)          default: /gpsr/asr/text
    ~utterance_end_topic (std_msgs/Bool)  default: /gpsr/asr/utterance_end
    ~confidence_topic (std_msgs/Float32)  default: /gpsr/asr/confidence (optional)

- Publishes:
    ~intent_topic (std_msgs/String)       default: /gpsr/intent

Features:
  * Supports external vocabulary via ~vocab_yaml (gpsr_vocab_v1)
  * Accepts parser results as dict/json OR object (GpsrCommand / dataclass)
  * Fixes race: utterance_end may arrive before text -> retry wait
  * Produces stable gpsr_intent_v1 schema with fixed slots/steps
"""

import os
import sys
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

import rospy
import yaml
from std_msgs.msg import String, Bool, Float32

# ---- PATH FIX (so scripts/ imports work in devel wrapper exec) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    import rospkg
    _rp = rospkg.RosPack()
    _pkg_dir = _rp.get_path("hsr_audio_pipeline")
    _scripts_dir = os.path.join(_pkg_dir, "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
except Exception:
    pass

from gpsr_parser import GpsrParser  # noqa


def _norm_ws(s: str) -> str:
    return " ".join((s or "").strip().split())


def _to_dict_any(obj: Any) -> Dict[str, Any]:
    """Convert parser return into a dict safely."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {"raw": obj}
    if is_dataclass(obj):
        return asdict(obj)
    # common "command object" patterns
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        # filter private / callables
        d = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            d[k] = v
        return d
    return {}


def _value_to_str_place(v: Any) -> Optional[str]:
    """Normalize place-like value into a string name."""
    if v is None:
        return None
    if isinstance(v, str):
        s = _norm_ws(v)
        return s if s else None
    if isinstance(v, dict):
        # YAML locations are dicts: {name, placement, ...}
        if "name" in v:
            return _norm_ws(str(v["name"]))
        return _norm_ws(str(v))
    return _norm_ws(str(v))


class GpsrParserNode:

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
        self.lang = rospy.get_param("~lang", "en")

        self.vocab_yaml = rospy.get_param("~vocab_yaml", "")
        self.max_text_age_sec = float(rospy.get_param("~max_text_age_sec", 1.0))
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # race fix knobs
        self.utt_end_retry_count = int(rospy.get_param("~utt_end_retry_count", 8))
        self.utt_end_retry_sleep = float(rospy.get_param("~utt_end_retry_sleep", 0.02))

        # ---- runtime state ----
        self._latest_text = ""
        self._latest_text_time = 0.0
        self._latest_conf: Optional[float] = None

        vocab = self._load_vocab(self.vocab_yaml)
        self.parser = GpsrParser(
            person_names=vocab["person_names"],
            location_names=vocab["location_names"],
            placement_location_names=vocab["placement_location_names"],
            room_names=vocab["room_names"],
            object_names=vocab["object_names"],
            object_categories_plural=vocab["object_categories_plural"],
            object_categories_singular=vocab["object_categories_singular"],
        )

        # pubs/subs
        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)
        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)
        try:
            rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)
        except Exception:
            pass

        rospy.loginfo("gpsr_parser_node ready (vocab=%s)", self.vocab_yaml if self.vocab_yaml else "defaults")

    # ---------------- vocab ----------------
    def _load_vocab(self, yaml_path: str) -> Dict[str, Any]:
        if yaml_path and os.path.exists(yaml_path):
            rospy.loginfo("Loading GPSR vocab from YAML: %s", yaml_path)
            with open(yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}

            names = [str(n).strip() for n in (y.get("names") or []) if str(n).strip()]
            rooms = [str(r).strip() for r in (y.get("rooms") or []) if str(r).strip()]

            location_names = []
            placement_location_names = []
            for loc in (y.get("locations") or []):
                if isinstance(loc, dict):
                    name = str(loc.get("name", "")).strip()
                    if not name:
                        continue
                    location_names.append(name)
                    if bool(loc.get("placement", False)):
                        placement_location_names.append(name)
                elif isinstance(loc, str):
                    name = loc.strip()
                    if not name:
                        continue
                    location_names.append(name)

            obj_names = []
            cats_s = []
            cats_p = []
            for c in (y.get("categories") or []):
                if not isinstance(c, dict):
                    continue
                s = str(c.get("singular", "")).strip()
                p = str(c.get("plural", "")).strip()
                if s:
                    cats_s.append(s)
                if p:
                    cats_p.append(p)
                for o in (c.get("objects") or []):
                    if not isinstance(o, str):
                        continue
                    obj_names.append(o.replace("_", " ").strip())

            # de-dup
            location_names = sorted(set(location_names), key=lambda x: x.lower())
            placement_location_names = sorted(set(placement_location_names), key=lambda x: x.lower())
            obj_names = sorted(set([o for o in obj_names if o]), key=lambda x: x.lower())
            cats_s = sorted(set(cats_s), key=lambda x: x.lower())
            cats_p = sorted(set(cats_p), key=lambda x: x.lower())

            return dict(
                person_names=names,
                room_names=rooms,
                location_names=location_names,
                placement_location_names=placement_location_names,
                object_names=obj_names,
                object_categories_singular=cats_s,
                object_categories_plural=cats_p,
            )

        rospy.logwarn("vocab_yaml not set or not found -> using built-in defaults")
        return dict(
            person_names=["Adel", "Angel", "Axel", "Charlie", "Jane", "Jules", "Morgan", "Paris", "Robin", "Simone"],
            room_names=["bedroom", "kitchen", "living room", "office", "bathroom"],
            location_names=["bed", "bedside table", "shelf", "trashbin", "dishwasher",
                            "kitchen table", "pantry", "refrigerator", "sink", "cabinet",
                            "desk", "tv stand", "storage rack", "side tables", "sofa", "bookshelf",
                            "armchair", "desk lamp"],
            placement_location_names=["bed", "bedside table", "shelf", "dishwasher",
                                      "kitchen table", "pantry", "refrigerator", "sink",
                                      "cabinet", "desk", "tv stand", "storage rack",
                                      "side tables", "sofa", "bookshelf"],
            object_names=["red wine", "milk", "cola", "juice", "sponge", "cleanser", "mustard"],
            object_categories_singular=["drink", "food", "cleaning supply"],
            object_categories_plural=["drinks", "food", "cleaning supplies"],
        )

    # ---------------- callbacks ----------------
    def _on_text(self, msg: String):
        self._latest_text = (msg.data or "").strip()
        self._latest_text_time = time.time()

    def _on_conf(self, msg: Float32):
        try:
            self._latest_conf = float(msg.data)
        except Exception:
            self._latest_conf = None

    def _get_fresh_text(self) -> Tuple[str, float]:
        """Return (text, age_sec)."""
        now = time.time()
        age = now - (self._latest_text_time or 0.0)
        return self._latest_text, age

    def _on_utt_end(self, msg: Bool):
        if not bool(msg.data):
            return

        # retry: utterance_end may arrive before the last text message
        text, age = self._get_fresh_text()
        if (not text) or (age > self.max_text_age_sec):
            for _ in range(max(0, self.utt_end_retry_count)):
                rospy.sleep(self.utt_end_retry_sleep)
                text, age = self._get_fresh_text()
                if text and age <= self.max_text_age_sec:
                    break

        if not text or age > self.max_text_age_sec:
            rospy.logwarn("gpsr_parser_node: utterance_end but no fresh text (age=%.3f, text='%s')", age, text)
            return

        # confidence gate (optional)
        if self.min_confidence >= 0.0 and (self._latest_conf is None or self._latest_conf < self.min_confidence):
            rospy.logwarn("gpsr_parser_node: confidence gate (conf=%s < %.2f) -> drop", self._latest_conf, self.min_confidence)
            return

        raw = text
        rospy.loginfo("parse: %s", raw)

        try:
            parsed_obj = self.parser.parse(raw)
        except Exception as e:
            rospy.logerr("parse failed: %s", e)
            return

        parsed = _to_dict_any(parsed_obj)
        payload = self._coerce_to_v1(parsed, raw, parsed_obj)
        self._publish(payload)

    # ---------------- shaping ----------------
    def _coerce_to_v1(self, parsed: Dict[str, Any], raw_text: str, parsed_obj: Any) -> Dict[str, Any]:
        payload = {
            "schema": "gpsr_intent_v1",
            "ok": bool(parsed.get("ok", True)),
            "need_confirm": bool(parsed.get("need_confirm", False)),
            "intent_type": parsed.get("intent_type", "other"),
            "raw_text": raw_text,
            "normalized_text": raw_text.lower(),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": parsed.get("command_kind"),
            "slots": {k: None for k in self.FIXED_SLOT_KEYS},
            "steps": [],
            "extras": {"legacy_slots": {}},
            "context": {"lang": self.lang, "source": "parser"},
        }

        # steps normalization
        steps = parsed.get("steps") or []
        for s in steps:
            if isinstance(s, dict):
                action = s.get("action")
                args = s.get("args", s.get("fields", {})) or {}
                payload["steps"].append({"action": action, "args": args})
        # legacy slots if any
        if isinstance(parsed.get("slots"), dict):
            payload["extras"]["legacy_slots"] = parsed["slots"]

        # fill fixed slots from steps (+ legacy slots if useful)
        self._extract_fixed_slots(payload)

        return payload

    def _extract_fixed_slots(self, payload: Dict[str, Any]):
        slots = payload["slots"]
        steps = payload.get("steps", [])

        # helper: set if empty
        def put(k: str, v: Any):
            if k not in slots:
                return
            if slots[k] is None and v is not None and str(v).strip() != "":
                slots[k] = v

        # from actions mapping
        for st in steps:
            action = st.get("action")
            args = st.get("args", {}) or {}

            # normalize dict place fields to string
            if "place" in args:
                args["place"] = _value_to_str_place(args["place"])
            if "from_place" in args:
                args["from_place"] = _value_to_str_place(args["from_place"])
            if "to_place" in args:
                args["to_place"] = _value_to_str_place(args["to_place"])

            # --- action -> fixed slots ---
            if action == "go_to_location":
                put("source_room", args.get("room") or args.get("location") or args.get("place"))

            elif action == "find_object_in_room":
                put("source_room", args.get("room"))
                put("object_category", args.get("object_or_category"))

            elif action == "take_object_from_place":
                put("source_place", args.get("place"))
                put("object_category", args.get("object_or_category"))
                put("object", args.get("object"))

            elif action == "bring_object_to_operator":
                put("object", args.get("object"))
                put("source_place", args.get("source_place") or args.get("place"))

            elif action == "deliver_object_to_person_in_room":
                put("destination_room", args.get("room"))
                put("person", args.get("person_filter") or args.get("person"))

            elif action == "guide_named_person_from_place_to_place":
                put("person", args.get("name"))
                put("source_place", args.get("from_place"))
                put("destination_place", args.get("to_place"))

            elif action == "guide_person_from_place_to_place":
                put("source_place", args.get("from_place"))
                put("destination_place", args.get("to_place"))

            elif action == "greet_person_with_clothes_in_room":
                put("source_room", args.get("room"))
                put("attribute", args.get("clothes_description"))

            elif action == "place_object_on_place":
                put("destination_place", args.get("place"))

            elif action == "tell_object_property_on_place":
                put("source_place", args.get("place"))
                put("comparison", args.get("comparison"))

            elif action == "count_persons_in_room":
                put("source_room", args.get("room"))

            elif action == "talk_to_person_in_room":
                put("source_room", args.get("room"))
                put("gesture", args.get("person_filter"))

        # final cleanup: ensure places are plain strings
        for k in ["source_place", "destination_place"]:
            if isinstance(slots.get(k), dict):
                slots[k] = _value_to_str_place(slots[k])

    def _publish(self, payload: Dict[str, Any]):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.spin()


if __name__ == "__main__":
    main()
