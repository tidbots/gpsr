#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_parser_node.py (ROS1 / Noetic)

Utterance-end driven GPSR parser node.

- Subscribes:
    * ~text_topic (std_msgs/String)             default: /gpsr/asr/text
    * ~utterance_end_topic (std_msgs/Bool)     default: /gpsr/asr/utterance_end
    * ~confidence_topic (std_msgs/Float32)     default: /gpsr/asr/confidence (optional)

- Publishes:
    * ~intent_topic (std_msgs/String JSON)     default: /gpsr/intent

Key features:
- Parse ONLY on utterance_end(True)
- Vocabulary is externalized via YAML (so yearly rulebook updates do not require code changes)
    * ~vocab_yaml: path to vocab.yaml
- Robust to parser return type:
    * dict-like (already a payload / command dict)
    * dataclass object with to_json()
    * pydantic-like object with model_dump()
    * plain object with __dict__

This node does NOT change the parser logic; it only:
- Loads vocab and instantiates GpsrParser
- Converts parser output into a single "gpsr_intent_v1" JSON consistently
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, Optional, Tuple, List

import rospy
from std_msgs.msg import String, Bool, Float32

# Allow importing gpsr_parser.py colocated in scripts/
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from gpsr_parser import GpsrParser  # noqa: E402


def _safe_load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML into dict. Raises a clear error if PyYAML is missing."""
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to use ~vocab_yaml. "
            "Install inside the container: `pip3 install pyyaml`"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"vocab_yaml must be a mapping/dict. got={type(data)} path={path}")
    return data


def _normalize_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        # allow comma-separated
        return [s.strip() for s in x.split(",") if s.strip()]
    return [str(x).strip()] if str(x).strip() else []


def _load_vocab_from_yaml(path: str) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Return (names, locations, placements, rooms, objects, categories_plural, categories_singular)."""
    y = _safe_load_yaml(path)

    # Accept a few aliases to be friendly
    person_names = _normalize_list(y.get("names") or y.get("person_names") or y.get("Names"))
    location_names = _normalize_list(y.get("locations") or y.get("location_names") or y.get("Locations"))
    placement_location_names = _normalize_list(
        y.get("placements") or y.get("placement_locations") or y.get("placement_location_names") or y.get("PlacementLocations")
    )
    room_names = _normalize_list(y.get("rooms") or y.get("room_names") or y.get("Rooms"))
    object_names = _normalize_list(y.get("objects") or y.get("object_names") or y.get("Objects"))

    # Categories can be:
    # 1) categories: [{singular: drink, plural: drinks}, ...]
    # 2) categories_singular / categories_plural lists
    cat_sing = _normalize_list(y.get("categories_singular") or y.get("object_categories_singular"))
    cat_plur = _normalize_list(y.get("categories_plural") or y.get("object_categories_plural"))

    if not (cat_sing and cat_plur):
        cats = y.get("categories") or y.get("object_categories") or []
        if isinstance(cats, list):
            tmp_s, tmp_p = [], []
            for c in cats:
                if isinstance(c, dict):
                    s = (c.get("singular") or c.get("sing") or c.get("s") or "").strip()
                    p = (c.get("plural") or c.get("plur") or c.get("p") or "").strip()
                    if s:
                        tmp_s.append(s)
                    if p:
                        tmp_p.append(p)
                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                    tmp_s.append(str(c[0]).strip())
                    tmp_p.append(str(c[1]).strip())
            cat_sing = cat_sing or [v for v in tmp_s if v]
            cat_plur = cat_plur or [v for v in tmp_p if v]

    # If placements omitted, fallback to locations
    if not placement_location_names:
        placement_location_names = list(location_names)

    return person_names, location_names, placement_location_names, room_names, object_names, cat_plur, cat_sing


def _as_dict(parsed: Any) -> Dict[str, Any]:
    """Convert various parser outputs into a plain dict."""
    if parsed is None:
        return {}
    if isinstance(parsed, dict):
        return parsed
    if hasattr(parsed, "to_json") and callable(getattr(parsed, "to_json")):
        try:
            return json.loads(parsed.to_json())
        except Exception:
            pass
    if hasattr(parsed, "model_dump") and callable(getattr(parsed, "model_dump")):
        d = parsed.model_dump()
        return d if isinstance(d, dict) else {}
    if hasattr(parsed, "__dict__"):
        return dict(parsed.__dict__)
    return {"value": str(parsed)}


class GpsrParserNode:
    def __init__(self):
        # ---- Params ----
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")

        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.max_text_age_sec = float(rospy.get_param("~max_text_age_sec", 1.0))
        self.utt_end_retry_count = int(rospy.get_param("~utt_end_retry_count", 8))
        self.utt_end_retry_sleep = float(rospy.get_param("~utt_end_retry_sleep", 0.02))
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # Vocabulary externalization
        self.vocab_yaml = rospy.get_param("~vocab_yaml", "")

        # ---- Latest ASR ----
        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf: Optional[float] = None
        self._latest_conf_stamp = rospy.Time(0)

        # ---- Parser ----
        self.parser = self._build_parser()

        # ---- ROS pub/sub ----
        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)

        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)
        rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)  # optional

        rospy.loginfo(
            "gpsr_parser_node: text=%s utterance_end=%s conf=%s -> intent=%s vocab_yaml=%s",
            self.text_topic, self.utt_end_topic, self.conf_topic, self.intent_topic, (self.vocab_yaml or "(builtin)"),
        )

    def _build_parser(self) -> GpsrParser:
        if self.vocab_yaml:
            if not os.path.isabs(self.vocab_yaml):
                self.vocab_yaml = os.path.abspath(self.vocab_yaml)
            if not os.path.exists(self.vocab_yaml):
                raise RuntimeError(f"vocab_yaml not found: {self.vocab_yaml}")

            person_names, location_names, placement_location_names, room_names, object_names, cat_plur, cat_sing = \
                _load_vocab_from_yaml(self.vocab_yaml)

            rospy.loginfo(
                "gpsr_parser_node: loaded vocab: names=%d rooms=%d locations=%d placements=%d objects=%d categories=%d",
                len(person_names), len(room_names), len(location_names), len(placement_location_names), len(object_names), len(cat_sing),
            )
            return GpsrParser(
                person_names=person_names,
                location_names=location_names,
                placement_location_names=placement_location_names,
                room_names=room_names,
                object_names=object_names,
                object_categories_plural=cat_plur,
                object_categories_singular=cat_sing,
            )

        # --- Built-in fallback (kept minimal; prefer YAML) ---
        person_names = ["Adel", "Angel", "Axel", "Charlie", "Jane", "Jules", "Morgan", "Paris", "Robin", "Simone"]
        location_names = ["bed", "bedside table", "shelf", "trashbin", "dishwasher", "potted plant", "kitchen table",
                          "chairs", "pantry", "refrigerator", "sink", "cabinet", "coatrack", "desk", "armchair",
                          "desk lamp", "waste basket", "tv stand", "storage rack", "lamp", "side tables", "sofa",
                          "bookshelf", "entrance", "exit"]
        placement_location_names = list(location_names)
        room_names = ["bedroom", "kitchen", "office", "living room", "bathroom"]
        object_names = ["red wine", "sponge", "cleanser"]
        cat_sing = ["drink", "cleaning supply", "food", "fruit", "toy", "snack", "dish"]
        cat_plur = ["drinks", "cleaning supplies", "food", "fruits", "toys", "snacks", "dishes"]

        return GpsrParser(
            person_names=person_names,
            location_names=location_names,
            placement_location_names=placement_location_names,
            room_names=room_names,
            object_names=object_names,
            object_categories_plural=cat_plur,
            object_categories_singular=cat_sing,
        )

    def _on_text(self, msg: String):
        self._latest_text = msg.data or ""
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)
        self._latest_conf_stamp = rospy.Time.now()

    def _publish(self, payload: Dict[str, Any]):
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(out)

    def _on_utt_end(self, msg: Bool):
        if not bool(msg.data):
            return

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec()
        text = (self._latest_text or "").strip()

        # Race guard: utterance_end may arrive slightly before the last /gpsr/asr/text message.
        if (not text) or (age > self.max_text_age_sec):
            for _ in range(self.utt_end_retry_count):
                rospy.sleep(self.utt_end_retry_sleep)
                now = rospy.Time.now()
                age = (now - self._latest_text_stamp).to_sec()
                text = (self._latest_text or "").strip()
                if text and (age <= self.max_text_age_sec):
                    break

        if (not text) or (age > self.max_text_age_sec):
            rospy.logwarn("gpsr_parser_node: utterance_end but no fresh text (age=%.3f, text='%s')", age, text)
            return

        # Optional confidence gating
        if self.min_confidence >= 0.0 and self._latest_conf is not None and self._latest_conf < self.min_confidence:
            self._publish({
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "raw_text": text,
                "normalized_text": (text.strip().lower()),
                "confidence": self._latest_conf,
                "source": "parser",
                "command_kind": None,
                "slots": {},
                "steps": [],
                "extras": {"reason": "low_confidence"},
                "context": {"lang": "en", "source": "parser"},
            })
            return

        rospy.loginfo("parse: %s", text)

        try:
            parsed = self.parser.parse(text)
        except Exception as e:
            rospy.logerr("gpsr_parser_node: parse exception: %s", str(e))
            self._publish({
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "raw_text": text,
                "normalized_text": (text.strip().lower()),
                "confidence": self._latest_conf,
                "source": "parser",
                "command_kind": None,
                "slots": {},
                "steps": [],
                "extras": {"reason": "parse_exception", "exception": str(e)},
                "context": {"lang": "en", "source": "parser"},
            })
            return

        if parsed is None:
            self._publish({
                "schema": "gpsr_intent_v1",
                "ok": False,
                "need_confirm": True,
                "intent_type": "other",
                "raw_text": text,
                "normalized_text": (text.strip().lower()),
                "confidence": self._latest_conf,
                "source": "parser",
                "command_kind": None,
                "slots": {},
                "steps": [],
                "extras": {"reason": "parse_failed"},
                "context": {"lang": "en", "source": "parser"},
            })
            return

        cmd = _as_dict(parsed)

        # If parser already returns a gpsr_intent_v1-like payload, pass through.
        if cmd.get("schema") == "gpsr_intent_v1":
            payload = cmd
            payload.setdefault("schema", "gpsr_intent_v1")
            payload.setdefault("raw_text", text)
            payload.setdefault("normalized_text", (text.strip().lower()))
            payload.setdefault("confidence", self._latest_conf)
            payload.setdefault("source", "parser")
            payload.setdefault("context", {"lang": "en", "source": "parser"})
            self._publish(payload)
            return

        # Otherwise: interpret as legacy command dict: {kind, steps:[{action, fields}], raw_text}
        kind = cmd.get("kind")
        steps = cmd.get("steps") or []
        raw = cmd.get("raw_text") or text

        payload = self._to_intent_v1_from_command(kind=kind, steps=steps, raw_text=raw)
        payload["confidence"] = self._latest_conf
        self._publish(payload)

    def _to_intent_v1_from_command(self, kind: Optional[str], steps: Any, raw_text: str) -> Dict[str, Any]:
        # Normalize steps
        norm_steps: List[Dict[str, Any]] = []
        if isinstance(steps, list):
            for s in steps:
                if isinstance(s, dict):
                    act = s.get("action") or ""
                    fields = s.get("fields") if isinstance(s.get("fields"), dict) else {}
                    norm_steps.append({"action": act, "args": fields})
                else:
                    norm_steps.append({
                        "action": str(getattr(s, "action", "")) or str(s),
                        "args": dict(getattr(s, "fields", {}) or {}),
                    })

        slots = self._extract_fixed_slots(norm_steps)

        return {
            "schema": "gpsr_intent_v1",
            "ok": True,
            "need_confirm": False,
            "intent_type": self._coarse_intent(kind, norm_steps),
            "raw_text": raw_text,
            "normalized_text": (raw_text.strip().lower()),
            "confidence": None,
            "source": "parser",
            "command_kind": kind,
            "slots": slots,
            "steps": norm_steps,
            "extras": {"legacy_slots": {}},
            "context": {"lang": "en", "source": "parser"},
        }

    def _coarse_intent(self, kind: Optional[str], steps: List[Dict[str, Any]]) -> str:
        key = ((kind or "") + " " + " ".join([s.get("action", "") or "" for s in steps])).lower()
        if any(w in key for w in ["guide", "escort", "lead", "follow"]):
            return "guide"
        if any(w in key for w in ["bring", "deliver", "take_object", "take_object_from", "fetch", "hand", "give"]):
            return "bring"
        if any(w in key for w in ["count", "tell", "answer", "describe", "talk_to_person"]):
            return "answer"
        return "other"

    def _extract_fixed_slots(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill fixed slots; includes small action->slot補完."""
        slots: Dict[str, Any] = {
            "object": None,
            "object_category": None,
            "quantity": None,
            "source_room": None,
            "source_place": None,
            "destination_room": None,
            "destination_place": None,
            "person": None,
            "person_at_source": None,
            "person_at_destination": None,
            "question_type": None,
            "attribute": None,
            "comparison": None,
            "gesture": None,
        }

        merged: Dict[str, Any] = {}
        for s in steps:
            a = s.get("args") or {}
            if isinstance(a, dict):
                merged.update(a)

        if merged.get("object"):
            slots["object"] = merged.get("object")
        if merged.get("object_or_category"):
            slots["object_category"] = merged.get("object_or_category")
        if merged.get("category"):
            slots["object_category"] = merged.get("category")

        if merged.get("room"):
            slots["source_room"] = merged.get("room")
        if merged.get("from_room"):
            slots["source_room"] = merged.get("from_room")
        if merged.get("to_room"):
            slots["destination_room"] = merged.get("to_room")

        if merged.get("place"):
            slots["source_place"] = merged.get("place")
        if merged.get("from_place"):
            slots["source_place"] = merged.get("from_place")
        if merged.get("to_place"):
            slots["destination_place"] = merged.get("to_place")
        if merged.get("source_place"):
            slots["source_place"] = merged.get("source_place")
        if merged.get("destination_place"):
            slots["destination_place"] = merged.get("destination_place")

        if merged.get("name"):
            slots["person"] = merged.get("name")
        if merged.get("person"):
            slots["person"] = merged.get("person")
        if merged.get("person_filter"):
            slots["person"] = merged.get("person_filter")

        if merged.get("comparison"):
            slots["comparison"] = merged.get("comparison")
        if merged.get("attribute"):
            slots["attribute"] = merged.get("attribute")
        if merged.get("gesture"):
            slots["gesture"] = merged.get("gesture")

        # --- Action-specific補完 ---
        for s in steps:
            act = (s.get("action") or "")
            args = s.get("args") or {}

            # greet_person_with_clothes_in_room → source_room, attribute
            if act == "greet_person_with_clothes_in_room":
                if args.get("room"):
                    slots["source_room"] = args.get("room")
                if args.get("clothes_description"):
                    slots["attribute"] = args.get("clothes_description")

            # place_object_on_place → destination_place
            if act == "place_object_on_place" and args.get("place"):
                slots["destination_place"] = args.get("place")

            # take_object_from_place → source_place / object_category
            if act == "take_object_from_place":
                if args.get("place"):
                    slots["source_place"] = args.get("place")
                if args.get("object_or_category"):
                    slots["object_category"] = args.get("object_or_category")

            # bring_object_to_operator → object / source_place
            if act == "bring_object_to_operator":
                if args.get("object"):
                    slots["object"] = args.get("object")
                if args.get("source_place"):
                    slots["source_place"] = args.get("source_place")

        return slots


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.loginfo("gpsr_parser_node started (utterance_end driven, vocab_yaml supported).") 
    rospy.spin()


if __name__ == "__main__":
    main()
