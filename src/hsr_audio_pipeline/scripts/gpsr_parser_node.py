#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import threading
import yaml
import re
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
    t2 = t.replace("..", ".")
    parts = [p.strip() for p in t2.split(".") if p.strip()]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        return parts[0] + "."
    return t


def _place_to_name(x):
    """YAML由来の dict や Parser由来の dict/obj を 'name' 文字列に正規化."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{") and "name" in s:
            m = re.search(r"'name'\s*:\s*'([^']+)'", s)
            if m:
                return m.group(1)
        return _normalize_ws(x) or None
    if isinstance(x, dict):
        if "name" in x:
            return _normalize_ws(str(x["name"])) or None
        return _normalize_ws(str(x)) or None
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
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    d = {}
    for k in ["ok", "need_confirm", "intent_type", "command_kind", "slots", "steps"]:
        if hasattr(obj, k):
            try:
                d[k] = getattr(obj, k)
            except Exception:
                pass
    return d


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _load_vocab_from_md(names_md: str, rooms_md: str, locations_md: str, objects_md: str, test_objects_md: str = ""):
    """
    /data/vocab/*.md から語彙を生成（generator系の表形式を想定）
      - names.md
      - room_names.md
      - location_names.md (末尾 (p) は placement)
      - objects.md (# Class plural singular ... / table)
      - test_objects.md (任意：hotwords増量用途だが parser 側でも objectsに加えてOK)
    返り値は parser_node が期待する辞書形式：
      person_names, room_names, location_names, placement_location_names,
      object_names, object_categories_plural, object_categories_singular
    """
    # names
    names = re.findall(r"\|\s*([A-Za-z]+)\s*\|", _safe_read(names_md))
    names = [x.strip() for x in names][1:] if names else []

    # rooms
    rooms = re.findall(r"\|\s*(\w+ \w*)\s*\|", _safe_read(rooms_md))
    rooms = [x.strip() for x in rooms][1:] if rooms else []

    # locations (+ placement)
    loc_pairs = re.findall(r"\|\s*([0-9]+)\s*\|\s*([A-Za-z,\s\(\)]+)\|", _safe_read(locations_md))
    locs = [b.strip() for (_, b) in loc_pairs]
    placement = [x.replace("(p)", "").strip() for x in locs if x.strip().endswith("(p)")]
    locs = [x.replace("(p)", "").strip() for x in locs]

    # objects + categories
    md_obj = _safe_read(objects_md)
    obj_names = re.findall(r"\|\s*(\w+)\s*\|", md_obj)
    obj_names = [o for o in obj_names if o != "Objectname"]
    obj_names = [o.replace("_", " ").strip() for o in obj_names if o.strip()]

    cats = re.findall(r"# Class \s*([\w,\s\(\)]+)\s*", md_obj)
    cats = [c.strip().replace("(", "").replace(")", "") for c in cats]
    cat_plur, cat_sing = [], []
    for c in cats:
        parts = c.split()
        if len(parts) >= 2:
            cat_plur.append(parts[0].replace("_", " "))
            cat_sing.append(parts[1].replace("_", " "))

    # test_objects (任意)
    if test_objects_md and os.path.exists(test_objects_md):
        md_test = _safe_read(test_objects_md)
        test_objs = re.findall(r"\|\s*(\w+)\s*\|", md_test)
        test_objs = [o for o in test_objs if o != "Objectname"]
        test_objs = [o.replace("_", " ").strip() for o in test_objs if o.strip()]
        obj_names = sorted(set(obj_names + test_objs))

    return dict(
        person_names=[n for n in names if n],
        room_names=[r for r in rooms if r],
        location_names=[l for l in locs if l],
        placement_location_names=sorted(set([p for p in placement if p])),
        object_names=sorted(set([o for o in obj_names if o])),
        object_categories_plural=sorted(set([x for x in cat_plur if x])),
        object_categories_singular=sorted(set([x for x in cat_sing if x])),
    )


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
        "tell_category_property_on_place": ("answer", "tellCatPropOnPlcmt"),
        "tell_person_info": ("answer", "tellPersonInfoInRoom"),
        "tell_person_info_from_loc_to_loc": ("answer", "tellPersonInfoFromLocToLoc"),
        # guide/greet
        "guide_named_person_from_place_to_place": ("guide", "guideNameFromBeacToBeac"),
        "guide_person_to_dest": ("guide", "guidePersonToDest"),
        "greet_person_with_clothes_in_room": ("composite", "greetClothDscInRm"),
        # place/manip
        "place_object_on_place": ("composite", "placeObjOnPlcmt"),
        "take_object": ("composite", "takeObj"),
        "find_object_in_room": ("composite", "findObjInRoom"),
        "go_to_location": ("composite", "goToLoc"),
        "follow_person_to_dest": ("guide", "followPersonToDest"),
    }

    def __init__(self):
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.lang = rospy.get_param("~lang", "en")

        # --- vocab paths ---
        self.vocab_dir = rospy.get_param("~vocab_dir", "/data/vocab")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", os.path.join(self.vocab_dir, "vocab.yaml"))

        # MD vocab (preferred if all exist)
        self.names_md = rospy.get_param("~names_md", os.path.join(self.vocab_dir, "names.md"))
        self.rooms_md = rospy.get_param("~rooms_md", os.path.join(self.vocab_dir, "room_names.md"))
        self.locations_md = rospy.get_param("~locations_md", os.path.join(self.vocab_dir, "location_names.md"))
        self.objects_md = rospy.get_param("~objects_md", os.path.join(self.vocab_dir, "objects.md"))
        self.test_objects_md = rospy.get_param("~test_objects_md", os.path.join(self.vocab_dir, "test_objects.md"))

        self.max_text_age = float(rospy.get_param("~max_text_age_sec", 1.0))
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # utterance_end arrives before text sometimes → retry a bit
        self.utt_retry_count = int(rospy.get_param("~utt_end_retry_count", 8))
        self.utt_retry_sleep = float(rospy.get_param("~utt_end_retry_sleep", 0.02))

        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf = None

        # debounce duplicated publishes (same normalized text within window)
        self.debounce_same_text_sec = float(rospy.get_param("~debounce_same_text_sec", 0.8))
        self._last_pub_norm = ""
        self._last_pub_wall = 0.0
        self._pub_lock = threading.Lock()

        # ---- load vocab (MD preferred; YAML fallback) ----
        use_md = all([
            self.names_md and os.path.exists(self.names_md),
            self.rooms_md and os.path.exists(self.rooms_md),
            self.locations_md and os.path.exists(self.locations_md),
            self.objects_md and os.path.exists(self.objects_md),
        ])

        vocab = None
        if use_md:
            try:
                vocab = _load_vocab_from_md(
                    self.names_md,
                    self.rooms_md,
                    self.locations_md,
                    self.objects_md,
                    self.test_objects_md
                )
                rospy.loginfo("gpsr_parser_node: vocab loaded from MD dir=%s", self.vocab_dir)
            except Exception as e:
                rospy.logwarn("gpsr_parser_node: failed to load MD vocab (fallback to YAML): %s", e)
                vocab = None

        if vocab is None:
            vocab = self._load_vocab(self.vocab_yaml)
            rospy.loginfo("gpsr_parser_node: vocab loaded from YAML=%s", self.vocab_yaml)

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

        rospy.loginfo(
            "gpsr_parser_node ready (text=%s, utt_end=%s, vocab_dir=%s, yaml=%s, md=%s)",
            self.text_topic,
            self.utt_end_topic,
            self.vocab_dir,
            self.vocab_yaml or "none",
            "on" if use_md else "off"
        )

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

        # fallback minimal（元コードのまま）
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

        # debounce: skip same text arriving twice (ASR repeat / utterance_end double-fire)
        norm = raw.lower().strip()
        now_wall = time.time()
        with self._pub_lock:
            if self.debounce_same_text_sec > 0 and norm and norm == self._last_pub_norm and (now_wall - self._last_pub_wall) < self.debounce_same_text_sec:
                rospy.logwarn("gpsr_parser_node: debounce skip (dt=%.3f, text='%s')", now_wall - self._last_pub_wall, norm)
                return
            self._last_pub_norm = norm
            self._last_pub_wall = now_wall

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
            if "location" in a:
                a["location"] = _place_to_name(a.get("location"))
            if "source_location" in a:
                a["source_location"] = _place_to_name(a.get("source_location"))
            if "target_location" in a:
                a["target_location"] = _place_to_name(a.get("target_location"))

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
                if "person_filter" in args:
                    set_if_empty("attribute", args.get("person_filter"))

            # count_persons_in_room
            if action == "count_persons_in_room":
                set_if_empty("source_room", args.get("room"))
                set_if_empty("question_type", "count_people")
                set_if_empty("attribute", args.get("person_filter_plural"))

            # find_object_in_room
            if action == "find_object_in_room":
                set_if_empty("source_room", args.get("room"))
                if args.get("object_or_category"):
                    set_if_empty("object_category", args.get("object_or_category"))

            # greet_person_with_clothes_in_room
            if action == "greet_person_with_clothes_in_room":
                set_if_empty("source_room", args.get("room"))
                set_if_empty("attribute", args.get("clothes_description"))

            # place_object_on_place
            if action == "place_object_on_place":
                set_if_empty("destination_place", args.get("place"))

            # follow_person_to_dest
            if action == "follow_person_to_dest":
                set_if_empty("destination_place", args.get("location"))
                if args.get("person_filter"):
                    set_if_empty("person", args.get("person_filter"))
                    set_if_empty("attribute", args.get("person_filter"))

            # go_to_location
            if action == "go_to_location":
                set_if_empty("source_room", args.get("room"))

            # tell_category_property_on_place
            if action == "tell_category_property_on_place":
                set_if_empty("question_type", "category_property")
                set_if_empty("comparison", args.get("comparison"))
                set_if_empty("object_category", args.get("object_category"))
                set_if_empty("source_place", args.get("place"))

            # tell_person_info
            if action == "tell_person_info":
                set_if_empty("question_type", "person_info")
                set_if_empty("attribute", args.get("person_info"))
                set_if_empty("source_room", args.get("room"))
                if args.get("location"):
                    set_if_empty("source_place", args.get("location"))

            # tell_person_info_from_loc_to_loc
            if action == "tell_person_info_from_loc_to_loc":
                set_if_empty("question_type", "person_info")
                set_if_empty("attribute", args.get("person_info"))
                set_if_empty("person_at_source", args.get("source_location"))
                set_if_empty("person_at_destination", args.get("target_location"))

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
