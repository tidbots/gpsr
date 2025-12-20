#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import yaml
import rospy
from std_msgs.msg import String, Bool, Float32

# ---- PATH FIX (rosrun wrapper/exec 対策) ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    import rospkg
    _pkg_dir = rospkg.RosPack().get_path("hsr_audio_pipeline")
    _scripts_dir = os.path.join(_pkg_dir, "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
except Exception:
    pass

from gpsr_parser import GpsrParser


def _normalize_ws(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s


def _collapse_duplicated_sentence(text: str) -> str:
    """
    ASRが同一文を2回繰り返すケースを1回に畳む。
    例: "X. X." -> "X."
    """
    t = _normalize_ws(text)
    if not t:
        return t
    # 末尾の句点/ピリオドを正規化
    t2 = t.replace("..", ".")
    # "A. A." パターン（完全一致）だけ畳む
    parts = [p.strip() for p in t2.split(".") if p.strip()]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        return parts[0] + "."
    return t


def _place_to_name(x):
    """YAML由来の dict や Parser由来の dict/obj を 'name' 文字列に正規化."""
    if x is None:
        return None
    if isinstance(x, str):
        # "{'name': 'xxx', 'placement': True}" のような文字列化dictも吸収
        s = x.strip()
        if s.startswith("{") and "name" in s:
            # 雑だけど安全寄りに抽出
            import re
            m = re.search(r"'name'\s*:\s*'([^']+)'", s)
            if m:
                return m.group(1)
        return _normalize_ws(x) or None
    if isinstance(x, dict):
        if "name" in x:
            return _normalize_ws(str(x["name"])) or None
        return _normalize_ws(str(x)) or None
    # オブジェクトで name 属性を持つ場合
    if hasattr(x, "name"):
        try:
            return _normalize_ws(str(getattr(x, "name"))) or None
        except Exception:
            pass
    return _normalize_ws(str(x)) or None


def _safe_to_dict(obj):
    """GpsrCommand / dataclass / dict / json(str) を dict に寄せる."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {"raw": obj}
    # dataclass / object
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        return d
    # fallback: attributes
    d = {}
    for k in ["ok", "need_confirm", "intent_type", "command_kind", "slots", "steps"]:
        if hasattr(obj, k):
            try:
                d[k] = getattr(obj, k)
            except Exception:
                pass
    return d


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

    # action -> intent_type / command_kind / slot補完ルール
    ACTION_HINTS = {
        # bring系
        "bring_object_to_operator": ("bring", "bringMeObjFromPlcmt"),
        "deliver_object_to_operator": ("bring", "bringMeObjFromPlcmt"),
        "take_object_from_place": ("bring", "takeObjFromPlcmt"),
        "deliver_object_to_person_in_room": ("bring", "deliverObjToPrsInRoom"),
        # answer / talk
        "answer_to_person_in_room": ("answer", "answerToPrsInRoom"),
        "talk_to_person_in_room": ("answer", "talkInfoToGestPrsInRoom"),
        "count_persons_in_room": ("answer", "countPrsInRoom"),
        "tell_object_property_on_place": ("answer", "tellObjPropOnPlcmt"),
        # guide/greet
        "guide_named_person_from_place_to_place": ("guide", "guideNameFromBeacToBeac"),
        "guide_person_to_dest": ("guide", "guidePersonToDest"),
        "greet_person_with_clothes_in_room": ("composite", "greetClothDscInRm"),
        # place/manip
        "place_object_on_place": ("composite", "placeObjOnPlcmt"),
        "take_object": ("composite", "takeObj"),
        "find_object_in_room": ("composite", "findObjInRoom"),
        "go_to_location": ("composite", "goToLoc"),
    }

    def __init__(self):
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.lang = rospy.get_param("~lang", "en")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", "")

        self.max_text_age = float(rospy.get_param("~max_text_age_sec", 1.0))
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # utterance_end arrives before text sometimes → retry a bit
        self.utt_retry_count = int(rospy.get_param("~utt_end_retry_count", 8))
        self.utt_retry_sleep = float(rospy.get_param("~utt_end_retry_sleep", 0.02))

        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf = None

        vocab = self._load_vocab(self.vocab_yaml)
        # 重要：複合語優先（desk lamp が desk より先にマッチ）
        vocab["location_names"] = sorted(vocab["location_names"], key=lambda s: (-len(s), s.lower()))
        vocab["placement_location_names"] = sorted(vocab["placement_location_names"], key=lambda s: (-len(s), s.lower()))
        vocab["object_names"] = sorted(vocab["object_names"], key=lambda s: (-len(s), s.lower()))
        vocab["object_categories_plural"] = sorted(vocab["object_categories_plural"], key=lambda s: (-len(s), s.lower()))
        vocab["object_categories_singular"] = sorted(vocab["object_categories_singular"], key=lambda s: (-len(s), s.lower()))

        self.parser = GpsrParser(
            person_names=vocab["person_names"],
            location_names=vocab["location_names"],
            placement_location_names=vocab["placement_location_names"],
            room_names=vocab["room_names"],
            object_names=vocab["object_names"],
            object_categories_plural=vocab["object_categories_plural"],
            object_categories_singular=vocab["object_categories_singular"],
        )

        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)
        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)
        try:
            rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)
        except Exception:
            pass

        rospy.loginfo("gpsr_parser_node ready (text=%s, utt_end=%s, vocab=%s)",
                      self.text_topic, self.utt_end_topic, self.vocab_yaml or "defaults")

    def _load_vocab(self, yaml_path: str):
        if yaml_path and os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}

            names = [n for n in (y.get("names") or []) if isinstance(n, str)]
            rooms = [r for r in (y.get("rooms") or []) if isinstance(r, str)]

            location_names = []
            placement_location_names = []
            for loc in (y.get("locations") or []):
                if not isinstance(loc, dict) or "name" not in loc:
                    continue
                nm = _normalize_ws(str(loc["name"]))
                if not nm:
                    continue
                location_names.append(nm)
                if bool(loc.get("placement", False)):
                    placement_location_names.append(nm)

            object_names = []
            cats_s = []
            cats_p = []
            for c in (y.get("categories") or []):
                if not isinstance(c, dict):
                    continue
                s = _normalize_ws(str(c.get("singular", "")))
                p = _normalize_ws(str(c.get("plural", "")))
                if s:
                    cats_s.append(s)
                if p:
                    cats_p.append(p)
                for o in (c.get("objects") or []):
                    if isinstance(o, str):
                        object_names.append(_normalize_ws(o.replace("_", " ")))

            return dict(
                person_names=names,
                room_names=rooms,
                location_names=location_names,
                placement_location_names=placement_location_names,
                object_names=sorted(set([x for x in object_names if x])),
                object_categories_plural=sorted(set([x for x in cats_p if x])),
                object_categories_singular=sorted(set([x for x in cats_s if x])),
            )

        # fallback minimal
        return dict(
            person_names=["Adel", "Angel", "Axel", "Charlie", "Jane", "Jules", "Morgan", "Paris", "Robin", "Simone"],
            room_names=["bedroom", "kitchen", "living room", "office", "bathroom"],
            location_names=["desk lamp", "desk", "storage rack", "sofa", "sink", "refrigerator"],
            placement_location_names=["desk lamp", "desk", "storage rack", "sofa", "sink", "refrigerator"],
            object_names=["red wine", "mustard", "chocolate jello"],
            object_categories_plural=["drinks", "foods", "cleaning supplies"],
            object_categories_singular=["drink", "food", "cleaning supply"],
        )

    def _on_text(self, msg: String):
        self._latest_text = _normalize_ws(msg.data)
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)

    def _wait_for_text_if_needed(self):
        """utterance_endが先に来た場合に少しだけ待つ"""
        for _ in range(max(0, self.utt_retry_count)):
            if self._latest_text:
                return
            time.sleep(max(0.0, self.utt_retry_sleep))

    def _on_utt_end(self, msg: Bool):
        if not msg.data:
            return

        # race対策：少し待つ
        if not self._latest_text:
            self._wait_for_text_if_needed()

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec() if self._latest_text_stamp != rospy.Time(0) else 1e9

        if not self._latest_text or age > self.max_text_age:
            rospy.logwarn("gpsr_parser_node: utterance_end but no fresh text (age=%.3f, text='%s')",
                          age, self._latest_text)
            return

        if self.min_confidence >= 0.0 and self._latest_conf is not None and self._latest_conf < self.min_confidence:
            rospy.logwarn("gpsr_parser_node: confidence gate (conf=%.3f < %.3f) skip",
                          self._latest_conf, self.min_confidence)
            return

        raw = _collapse_duplicated_sentence(self._latest_text)
        rospy.loginfo("parse: %s", raw.lower())

        try:
            parsed_obj = self.parser.parse(raw)
        except Exception as e:
            rospy.logerr("parse failed: %s", e)
            return

        payload = self._coerce_to_v1(parsed_obj, raw)
        self.pub_intent.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _coerce_to_v1(self, parsed_obj, raw_text: str):
        parsed = _safe_to_dict(parsed_obj)

        payload = {
            "schema": "gpsr_intent_v1",
            "ok": bool(parsed.get("ok", True)),
            "need_confirm": bool(parsed.get("need_confirm", False)),
            "intent_type": parsed.get("intent_type"),
            "raw_text": raw_text,
            "normalized_text": raw_text.lower(),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": parsed.get("command_kind"),
            "slots": {k: None for k in self.FIXED_SLOT_KEYS},
            "steps": [],
            "extras": {"legacy_slots": parsed.get("slots", {}) or {}},
            "context": {"lang": self.lang, "source": "parser"},
        }

        # steps 抽出（dict/obj 両対応）
        steps_in = parsed.get("steps") or []
        norm_steps = []
        for s in steps_in:
            if isinstance(s, dict):
                action = s.get("action")
                args = s.get("args", s.get("fields", {})) or {}
            else:
                action = getattr(s, "action", None)
                args = getattr(s, "args", None) or getattr(s, "fields", None) or {}
            if not action:
                continue
            norm_steps.append({"action": action, "args": dict(args)})

        payload["steps"] = norm_steps

        # action から intent_type / command_kind を補完
        if payload["steps"]:
            a0 = payload["steps"][0]["action"]
            hint = self.ACTION_HINTS.get(a0)
            if not payload["intent_type"] and hint:
                payload["intent_type"] = hint[0]
            if not payload["command_kind"] and hint:
                payload["command_kind"] = hint[1]

        if not payload["intent_type"]:
            payload["intent_type"] = "other"

        # slots補完（最小限）
        self._fill_slots_from_steps(payload)

        # place文字列化（dict混入対策）
        for k in ["source_place", "destination_place"]:
            payload["slots"][k] = _place_to_name(payload["slots"].get(k))

        # steps 内の place/from/to の dict も正規化
        for st in payload["steps"]:
            a = st.get("args", {})
            if "place" in a:
                a["place"] = _place_to_name(a.get("place"))
            if "from_place" in a:
                a["from_place"] = _place_to_name(a.get("from_place"))
            if "to_place" in a:
                a["to_place"] = _place_to_name(a.get("to_place"))

        return payload

    def _fill_slots_from_steps(self, payload: dict):
        slots = payload["slots"]
        steps = payload.get("steps", [])

        def set_if_empty(key, val):
            if val is None:
                return
            if slots.get(key) is None:
                slots[key] = val

        for st in steps:
            action = st["action"]
            args = st.get("args", {})

            # bring_object_to_operator
            if action in ("bring_object_to_operator",):
                set_if_empty("object", args.get("object"))
                set_if_empty("source_place", args.get("source_place") or args.get("place"))

            # take_object_from_place
            if action in ("take_object_from_place",):
                set_if_empty("object_category", args.get("object_or_category"))
                set_if_empty("source_place", args.get("place"))

            # deliver_object_to_person_in_room
            if action in ("deliver_object_to_person_in_room",):
                set_if_empty("person", args.get("person_filter"))
                set_if_empty("destination_room", args.get("room"))

            # answer_to_person_in_room / talk_to_person_in_room
            if action in ("answer_to_person_in_room", "talk_to_person_in_room"):
                set_if_empty("gesture", "raising right arm" if "right arm" in str(args.get("person_filter","")) else None)
                set_if_empty("destination_room", args.get("room"))
                # ルールブック表現を attribute に寄せる（最低限）
                if "person_filter" in args:
                    set_if_empty("attribute", args.get("person_filter"))

            # count_persons_in_room
            if action == "count_persons_in_room":
                set_if_empty("source_room", args.get("room"))
                set_if_empty("question_type", "count_people")
                set_if_empty("attribute", args.get("person_filter_plural"))

            # greet_person_with_clothes_in_room → source_room, attribute
            if action == "greet_person_with_clothes_in_room":
                set_if_empty("source_room", args.get("room"))
                set_if_empty("attribute", args.get("clothes_description"))

            # place_object_on_place → destination_place
            if action == "place_object_on_place":
                set_if_empty("destination_place", args.get("place"))

            # go_to_location → source_room (room) に寄せる
            if action == "go_to_location":
                set_if_empty("source_room", args.get("room"))

            # guide_named_person_from_place_to_place
            if action == "guide_named_person_from_place_to_place":
                set_if_empty("person", args.get("name"))
                set_if_empty("source_place", args.get("from_place"))
                set_if_empty("destination_place", args.get("to_place"))


def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.spin()


if __name__ == "__main__":
    main()
