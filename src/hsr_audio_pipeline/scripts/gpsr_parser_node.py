#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_parser_node.py (ROS1 Noetic)

- Subscribes:
    ~text_topic (std_msgs/String): ASR final text
    ~utterance_end_topic (std_msgs/Bool): utterance end trigger
    ~confidence_topic (std_msgs/Float32): optional ASR confidence
- Publishes:
    ~intent_topic (std_msgs/String): JSON payload (schema=gpsr_intent_v1)

This node is designed to work with gpsr_parser.py (GpsrParser.parse()) and focuses on:
- vocab loading (MD preferred; YAML fallback)
- debounce + confidence gate
- coerce parse result into gpsr_intent_v1
- lightweight reference resolution for it/them across steps
- slot filling for downstream SMACH/BT executors
"""

import os
import sys
import json
import time
import threading
import yaml
import re
import rospy
from std_msgs.msg import String, Bool, Float32

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# optional: also allow importing from hsr_audio_pipeline/scripts when running inside that package
try:
    import rospkg  # type: ignore
    _pkg_dir = rospkg.RosPack().get_path("hsr_audio_pipeline")
    _scripts_dir = os.path.join(_pkg_dir, "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
except Exception:
    pass

from gpsr_parser import GpsrParser  # noqa: E402


def _normalize_ws(s: str) -> str:
    s = (s or "").strip()
    return " ".join(s.split())


def _collapse_duplicated_sentence(text: str) -> str:
    """
    Some ASR pipelines produce duplicated sentence like:
        "do X. do X."
    Collapse it to one.
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
    if x is None:
        return None
    if isinstance(x, str):
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
    The MD formats are those used by RoboCup@Home command generator repos.
    This parser is intentionally tolerant (regex-based).
    """
    names = re.findall(r"\|\s*([A-Za-z]+)\s*\|", _safe_read(names_md))
    names = [x.strip() for x in names][1:] if names else []

    rooms = re.findall(r"\|\s*([A-Za-z][A-Za-z\s]*)\s*\|", _safe_read(rooms_md))
    rooms = [x.strip() for x in rooms][1:] if rooms else []

    loc_pairs = re.findall(r"\|\s*([0-9]+)\s*\|\s*([A-Za-z0-9,\s\(\)\-]+)\|", _safe_read(locations_md))
    locs = [b.strip() for (_, b) in loc_pairs]
    placement = [x.replace("(p)", "").strip() for x in locs if x.strip().endswith("(p)")]
    locs = [x.replace("(p)", "").strip() for x in locs]

    md_obj = _safe_read(objects_md)
    obj_names = re.findall(r"\|\s*([A-Za-z0-9_]+)\s*\|", md_obj)
    obj_names = [o for o in obj_names if o.lower() != "objectname"]
    obj_names = [o.replace("_", " ").strip() for o in obj_names if o.strip()]

    cats = re.findall(r"#\s*Class\s*([\w,\s\(\)\-]+)\s*", md_obj)
    cats = [c.strip().replace("(", "").replace(")", "") for c in cats]
    cat_plur, cat_sing = [], []
    for c in cats:
        parts = c.split()
        if len(parts) >= 2:
            cat_plur.append(parts[0].replace("_", " "))
            cat_sing.append(parts[1].replace("_", " "))

    if test_objects_md and os.path.exists(test_objects_md):
        md_test = _safe_read(test_objects_md)
        test_objs = re.findall(r"\|\s*([A-Za-z0-9_]+)\s*\|", md_test)
        test_objs = [o for o in test_objs if o.lower() != "objectname"]
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
    # fixed slots for downstream components
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
        "pose",
        "info_type",
        "info_text",
    ]

    # action -> (intent_type, command_kind)
    ACTION_HINTS = {
        # core manipulation / navigation
        "bring_object_to_operator": ("bring", "bringMeObjFromPlcmt"),
        "deliver_object_to_person_in_room": ("bring", "deliverObjToPrsInRoom"),
        "place_object_on_place": ("composite", "placeObjOnPlcmt"),
        "take_object": ("composite", "takeObj"),
        "take_object_from_place": ("composite", "takeObjFromPlcmt"),
        "find_object_in_room": ("composite", "findObjInRoom"),
        "go_to_location": ("composite", "goToLoc"),

        # follow/guide family
        "follow_last_person": ("guide", "followLastPrs"),
        "guide_last_person_to_place": ("guide", "guideLastPrsToPlc"),
        "follow_person_at_place": ("guide", "followPrsAtBeac"),
        "follow_named_person_in_room": ("guide", "followNameInRoom"),
        "follow_named_person_at_place": ("guide", "followNameAtBeac"),
        "follow_named_person_from_place_to_place": ("guide", "followNameFromBeacToBeac"),
        "guide_named_person_from_place_to_place": ("guide", "guideNameFromBeacToBeac"),

        # social
        "meet_person_in_room": ("guide", "meetPrsInRoom"),
        "greet_named_person_in_room": ("guide", "greetNameInRoom"),
        "greet_person_in_room": ("guide", "greetPrsInRoom"),
        "introduce_self_to_named_person_in_room": ("guide", "introSelfToNameInRoom"),
        "introduce_self_to_person_in_room": ("guide", "introSelfToPrsInRoom"),

        # Q&A / perception
        "count_persons_in_room": ("answer", "countPrsInRoom"),
        "count_objects_on_place": ("answer", "countObjOnPlcmt"),
        "compare_object_on_place": ("answer", "tellCompareObjOnPlcmt"),
        "answer_to_person_in_room": ("answer", "answerToPrsInRoom"),
        "tell_name_of_person_at_place": ("answer", "tellNameOfPersonAtPlc"),

        # B: tell/say information
        "tell_information_to_person_in_room": ("answer", "tellInfoToPrsInRoom"),
        "tell_information_to_person_at_place": ("answer", "tellInfoToPrsAtPlc"),

        # C2: tell pose/gesture (attribute) about a person
        "tell_person_attribute_in_room": ("answer", "tellGestOfPrsInRoom"),
        "tell_person_attribute_at_place": ("answer", "tellGestOfPrsAtPlc"),
    }

    def __init__(self):
        # topics
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.lang = rospy.get_param("~lang", "en")

        # vocab paths (prefer /data/vocab for docker bind-mount)
        self.vocab_dir = rospy.get_param("~vocab_dir", "/data/vocab")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", os.path.join(self.vocab_dir, "vocab.yaml"))

        self.names_md = rospy.get_param("~names_md", os.path.join(self.vocab_dir, "names.md"))
        self.rooms_md = rospy.get_param("~rooms_md", os.path.join(self.vocab_dir, "room_names.md"))
        self.locations_md = rospy.get_param("~locations_md", os.path.join(self.vocab_dir, "location_names.md"))
        self.objects_md = rospy.get_param("~objects_md", os.path.join(self.vocab_dir, "objects.md"))
        self.test_objects_md = rospy.get_param("~test_objects_md", os.path.join(self.vocab_dir, "test_objects.md"))

        # gating
        self.max_text_age = float(rospy.get_param("~max_text_age_sec", 30.0))
        self.min_confidence = float(rospy.get_param("~min_confidence", -1.0))

        # when utterance_end arrives slightly before final text
        self.utt_retry_count = int(rospy.get_param("~utt_end_retry_count", 8))
        self.utt_retry_sleep = float(rospy.get_param("~utt_end_retry_sleep", 0.02))

        # debounce duplicated publish
        self.debounce_same_text_sec = float(rospy.get_param("~debounce_same_text_sec", 1.5))

        # state
        self._latest_text = ""
        self._latest_text_stamp = rospy.Time(0)
        self._latest_conf = None
        self._last_pub_norm = ""
        self._last_pub_wall = 0.0
        self._pub_lock = threading.Lock()

        vocab = self._load_vocab()
        self._obj_set = set([s.strip().lower() for s in vocab["object_names"] if isinstance(s, str)])
        self._cat_set = set([s.strip().lower() for s in (vocab["object_categories_singular"] + vocab["object_categories_plural"]) if isinstance(s, str)])

        # prefer longer tokens
        vocab["location_names"] = sorted(vocab["location_names"], key=lambda s: (-len(s), s.lower()))
        vocab["placement_location_names"] = sorted(vocab["placement_location_names"], key=lambda s: (-len(s), s.lower()))
        vocab["object_names"] = sorted(vocab["object_names"], key=lambda s: (-len(s), s.lower()))
        vocab["object_categories_plural"] = sorted(vocab["object_categories_plural"], key=lambda s: (-len(s), s.lower()))
        vocab["object_categories_singular"] = sorted(vocab["object_categories_singular"], key=lambda s: (-len(s), s.lower()))
        vocab["person_names"] = sorted(vocab["person_names"], key=lambda s: (-len(s), s.lower()))
        vocab["room_names"] = sorted(vocab["room_names"], key=lambda s: (-len(s), s.lower()))

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
            "gpsr_parser_node ready (text=%s, utt_end=%s, intent=%s, vocab_dir=%s)",
            self.text_topic, self.utt_end_topic, self.intent_topic, self.vocab_dir
        )

    def _load_vocab(self):
        use_md = all([
            self.names_md and os.path.exists(self.names_md),
            self.rooms_md and os.path.exists(self.rooms_md),
            self.locations_md and os.path.exists(self.locations_md),
            self.objects_md and os.path.exists(self.objects_md),
        ])

        if use_md:
            try:
                v = _load_vocab_from_md(
                    self.names_md, self.rooms_md, self.locations_md,
                    self.objects_md, self.test_objects_md
                )
                rospy.loginfo("gpsr_parser_node: vocab loaded from MD dir=%s", self.vocab_dir)
                return v
            except Exception as e:
                rospy.logwarn("gpsr_parser_node: failed to load MD vocab (fallback to YAML): %s", e)

        v = self._load_vocab_yaml(self.vocab_yaml)
        rospy.loginfo("gpsr_parser_node: vocab loaded from YAML=%s", self.vocab_yaml)
        return v

    def _load_vocab_yaml(self, yaml_path: str):
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

        # empty fallback
        return dict(
            person_names=[],
            room_names=[],
            location_names=[],
            placement_location_names=[],
            object_names=[],
            object_categories_plural=[],
            object_categories_singular=[],
        )

    def _on_text(self, msg: String):
        self._latest_text = _normalize_ws(msg.data)
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg: Float32):
        self._latest_conf = float(msg.data)

    def _wait_for_text_if_needed(self):
        for _ in range(max(0, self.utt_retry_count)):
            if self._latest_text:
                return
            time.sleep(max(0.0, self.utt_retry_sleep))

    def _on_utt_end(self, msg: Bool):
        if not msg.data:
            return

        if not self._latest_text:
            self._wait_for_text_if_needed()

        now = rospy.Time.now()
        age = (now - self._latest_text_stamp).to_sec() if self._latest_text_stamp != rospy.Time(0) else 1e9

        if not self._latest_text or age > self.max_text_age:
            rospy.logwarn("gpsr_parser_node: utterance_end but no fresh text (age=%.3f, text='%s')", age, self._latest_text)
            return

        if self.min_confidence >= 0.0 and self._latest_conf is not None and self._latest_conf < self.min_confidence:
            rospy.logwarn("gpsr_parser_node: confidence gate (conf=%.3f < %.3f) skip", self._latest_conf, self.min_confidence)
            return

        raw = _collapse_duplicated_sentence(self._latest_text)
        norm = raw.lower().strip()
        now_wall = time.time()
        with self._pub_lock:
            if self.debounce_same_text_sec > 0 and norm and norm == self._last_pub_norm and (now_wall - self._last_pub_wall) < self.debounce_same_text_sec:
                rospy.logwarn("gpsr_parser_node: debounce skip (dt=%.3f, text='%s')", now_wall - self._last_pub_wall, norm)
                return
            self._last_pub_norm = norm
            self._last_pub_wall = now_wall

        rospy.loginfo("parse: %s", raw)

        try:
            parsed_obj = self.parser.parse(raw)
        except Exception as e:
            rospy.logerr("gpsr_parser_node: parse exception: %s", e)
            # publish a fail payload for downstream recovery
            payload = self._make_fail_payload(raw_text=raw, reason=str(e))
            self.pub_intent.publish(String(data=json.dumps(payload, ensure_ascii=False)))
            return

        payload = self._coerce_to_v1(parsed_obj, raw)
        self.pub_intent.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _make_fail_payload(self, raw_text: str, reason: str):
        return {
            "schema": "gpsr_intent_v1",
            "ok": False,
            "need_confirm": True,
            "intent_type": "",
            "raw_text": raw_text,
            "normalized_text": (raw_text or "").lower(),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": "",
            "slots": {k: None for k in self.FIXED_SLOT_KEYS},
            "steps": [],
            "extras": {"error": reason},
            "context": {"lang": self.lang, "source": "parser"},
        }

    # ---------- classifier ----------
    def _classify_obj_or_cat(self, s: str):
        if not s:
            return None, None
        v = _normalize_ws(str(s)).lower()
        if not v:
            return None, None

        if v in self._obj_set:
            return "object", _normalize_ws(str(s))
        if v in self._cat_set:
            return "category", _normalize_ws(str(s))

        if v.endswith("s") and v[:-1] in self._cat_set:
            return "category", _normalize_ws(str(s))
        if (v + "s") in self._cat_set:
            return "category", _normalize_ws(str(s))

        # default: treat as category (safer for "a snack"/"a drink")
        return "category", _normalize_ws(str(s))

    def _canonicalize_object_fields(self, args: dict) -> dict:
        if not isinstance(args, dict):
            return {}

        if args.get("object_or_category"):
            kind, val = self._classify_obj_or_cat(args.get("object_or_category"))
            if kind == "object":
                args.setdefault("object", val)
            elif kind == "category":
                args.setdefault("object_category", val)
            args.pop("object_or_category", None)

        if args.get("object") and not args.get("object_category"):
            kind, val = self._classify_obj_or_cat(args.get("object"))
            if kind == "category":
                args["object_category"] = val
                args.pop("object", None)

        if args.get("object_category"):
            args["object_category"] = _normalize_ws(str(args["object_category"]))
        if args.get("object"):
            args["object"] = _normalize_ws(str(args["object"]))

        return args

    # ---------- reference resolver (it/them / last person) ----------
    def _apply_reference_resolution_and_unify(self, steps: list) -> dict:
        """
        Keeps minimal state across steps:
        - last object / category
        - last person (for follow/guide to use "them")
        """
        state = {"object": None, "object_category": None, "person": None, "person_filter": None}

        def update_state_from_args(a: dict):
            a = self._canonicalize_object_fields(a)

            if a.get("object"):
                state["object"] = a["object"]
                state["object_category"] = None
            elif a.get("object_category"):
                state["object_category"] = a["object_category"]

            # person/name tracking
            if a.get("name"):
                state["person"] = a.get("name")
                state["person_filter"] = None
            elif a.get("person_filter"):
                state["person_filter"] = a.get("person_filter")

            return a

        for st in steps:
            action = st.get("action")
            args = st.get("args", {}) or {}

            args = update_state_from_args(args)

            def fill_obj_if_missing():
                if args.get("object") or args.get("object_category"):
                    return
                if state["object"]:
                    args["object"] = state["object"]
                elif state["object_category"]:
                    args["object_category"] = state["object_category"]

            # actions where "it" is likely
            if action in (
                "take_object",
                "take_object_from_place",
                "place_object_on_place",
                "bring_object_to_operator",
                "deliver_object_to_person_in_room",
            ):
                fill_obj_if_missing()

            st["args"] = update_state_from_args(args)

        return state

    def _coerce_to_v1(self, parsed_obj, raw_text: str):
        parsed = _safe_to_dict(parsed_obj)

        payload = {
            "schema": "gpsr_intent_v1",
            "ok": bool(parsed.get("ok", True)),
            "need_confirm": bool(parsed.get("need_confirm", False)),
            "intent_type": parsed.get("intent_type"),
            "raw_text": raw_text,
            "normalized_text": (raw_text or "").lower(),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": parsed.get("command_kind"),
            "slots": {k: None for k in self.FIXED_SLOT_KEYS},
            "steps": [],
            "extras": {"legacy_slots": parsed.get("slots", {}) or {}},
            "context": {"lang": self.lang, "source": "parser"},
        }

        # normalize steps
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

        # reference resolution + unify object/category fields
        state = self._apply_reference_resolution_and_unify(norm_steps)
        payload["steps"] = norm_steps

        # infer intent_type/command_kind from first step when missing
        if payload["steps"]:
            a0 = payload["steps"][0]["action"]
            hint = self.ACTION_HINTS.get(a0)
            if not payload["intent_type"] and hint:
                payload["intent_type"] = hint[0]
            if not payload["command_kind"] and hint:
                payload["command_kind"] = hint[1]

        if not payload["intent_type"]:
            payload["intent_type"] = "other"

        self._fill_slots_from_steps(payload)

        self._refine_command_kind_and_slots(payload)

        # reflect last object/category into slots if still empty
        if payload["slots"].get("object") is None and state.get("object"):
            payload["slots"]["object"] = state["object"]
        if payload["slots"].get("object_category") is None and state.get("object_category"):
            payload["slots"]["object_category"] = state["object_category"]

        # propagate object/category into step args when missing (helps SMACH executor)
        obj = payload["slots"].get("object")
        cat = payload["slots"].get("object_category")
        if obj is not None or cat is not None:
            for st in payload.get("steps", []):
                act = st.get("action")
                args = st.get("args") or {}
                if "object_or_category" not in args and "object" not in args and "object_category" not in args:
                    if act in (
                        "bring_object_to_operator",
                        "take_object",
                        "place_object_on_place",
                        "deliver_object_to_person_in_room",
                        "give_object_to_person",
                        "bring_object_to_person",
                    ):
                        if obj is not None:
                            args["object"] = obj
                        if cat is not None:
                            args["object_category"] = cat
                st["args"] = args

        # stringify place-like fields
        for k in ["source_place", "destination_place"]:
            payload["slots"][k] = _place_to_name(payload["slots"].get(k))

        for st in payload["steps"]:
            a = st.get("args", {})
            for key in [
                "place", "from_place", "to_place", "location",
                "source_place", "destination_place", "from_location", "to_location"
            ]:
                if key in a:
                    a[key] = _place_to_name(a.get(key))

        return payload


    def _fill_slots_from_steps(self, payload: dict):
        """Back-fill slots from steps for downstream consistency.

        This is intentionally conservative: it only fills slots that are currently None.
        """
        slots = payload.get("slots") or {}
        steps = payload.get("steps", []) or []
        norm = (payload.get("normalized_text") or payload.get("raw_text") or "").lower()

        def set_if_empty(key, val):
            if val is None:
                return
            if isinstance(val, str) and not val.strip():
                return
            if slots.get(key) is None:
                slots[key] = val

        def extract_pose(text: str):
            t = (text or "").lower()
            for p in ("lying", "sitting", "standing"):
                if p in t:
                    return p
            return None

        def place_name(x):
            return _place_to_name(x)

        for st in steps:
            action = st.get("action")
            args = st.get("args", {}) or {}

            # --- navigation ---
            if action == "go_to_location":
                set_if_empty("destination_place", args.get("location") or args.get("place"))
            elif action == "go_to_room":
                set_if_empty("destination_room", args.get("room"))

            # --- perception / search ---
            elif action == "find_object_in_room":
                set_if_empty("source_room", args.get("room"))
                obj = args.get("object") or args.get("object_or_category") or args.get("object_category")
                if obj:
                    # If it matches known objects, prefer object; otherwise treat as category
                    if hasattr(self.vocab, "objects") and obj in self.vocab.objects:
                        set_if_empty("object", obj)
                    else:
                        # categories like 'cleaning supply'
                        set_if_empty("object_category", obj)
            elif action == "find_object_at_place":
                set_if_empty("source_place", args.get("place"))
                obj = args.get("object") or args.get("object_or_category") or args.get("object_category")
                if obj:
                    if hasattr(self.vocab, "objects") and obj in self.vocab.objects:
                        set_if_empty("object", obj)
                    else:
                        set_if_empty("object_category", obj)

            # --- grasp / take ---
            elif action in ("take_object", "grasp_object", "pick_object"):
                obj = args.get("object") or args.get("object_or_category") or args.get("object_category")
                if obj:
                    if hasattr(self.vocab, "objects") and obj in self.vocab.objects:
                        set_if_empty("object", obj)
                    else:
                        set_if_empty("object_category", obj)

            # --- place / put ---
            elif action in ("place_object_on_place", "put_object_on_place"):
                set_if_empty("destination_place", args.get("place") or args.get("location"))
                obj = args.get("object") or args.get("object_or_category") or args.get("object_category")
                if obj:
                    if hasattr(self.vocab, "objects") and obj in self.vocab.objects:
                        set_if_empty("object", obj)
                    else:
                        set_if_empty("object_category", obj)

            # --- deliver / give ---
            elif action in (
                "deliver_object_to_person_in_room",
                "give_object_to_person_in_room",
                "bring_object_to_person_in_room",
                "deliver_object_to_person_at_place",
                "give_object_to_person_at_place",
                "bring_object_to_person_at_place",
                "bring_object_to_operator",
                "bring_object_to_operator_in_room",
            ):
                # destination
                if "room" in args and args.get("room"):
                    set_if_empty("destination_room", args.get("room"))
                if "place" in args and args.get("place"):
                    set_if_empty("destination_place", args.get("place"))
                # person
                pf = args.get("person_filter") or args.get("name") or args.get("person")
                set_if_empty("person_at_destination", pf)
                # pose
                set_if_empty("pose", extract_pose(pf))

            # --- Q/A and description ---
            elif action in ("answer_to_person_in_room", "tell_person_info_in_room"):
                # keep destination room if present
                if args.get("room"):
                    set_if_empty("destination_room", args.get("room"))
                pf = args.get("person_filter") or args.get("name") or args.get("person")
                set_if_empty("person_at_destination", pf)
                set_if_empty("pose", extract_pose(pf))

        # --- fallback from normalized text for common categories ---
        if slots.get("object_category") is None:
            # minimal, safe heuristics for GPSR frequent categories
            for cat in ("cleaning supply", "drink", "snack", "fruit", "food"):
                if cat in norm:
                    set_if_empty("object_category", cat)
                    break

        # normalize place-like slots at end
        slots["source_place"] = place_name(slots.get("source_place"))
        slots["destination_place"] = place_name(slots.get("destination_place"))
        payload["slots"] = slots



    def _refine_command_kind_and_slots(self, payload: dict):
        """
        Post-process composite patterns so downstream executors can rely on stable command_kind/slots.
        """
        steps = payload.get("steps", []) or []
        actions = [s.get("action") for s in steps if isinstance(s, dict)]
        # find -> take -> deliver pattern
        if actions == ["find_object_in_room", "take_object", "deliver_object_to_person_in_room"] or (
            len(actions) >= 3
            and actions[0] in ("go_to_location", "find_object_in_room")
            and "find_object_in_room" in actions
            and "take_object" in actions
            and "deliver_object_to_person_in_room" in actions
        ):
            payload["intent_type"] = "composite"
            payload["command_kind"] = "findTakeDeliverObj"

        # pose normalization: if attribute mentions pose but pose slot empty
        slots = payload.get("slots", {})
        if slots.get("pose") is None:
            attr = slots.get("attribute")
            if attr:
                mpose = re.search(r"\b(lying|sitting|standing)\b", str(attr))
                if mpose:
                    slots["pose"] = mpose.group(1)

def main():
    rospy.init_node("gpsr_parser_node")
    _ = GpsrParserNode()
    rospy.spin()


if __name__ == "__main__":
    main()

            if action == "bring_object_to_operator":
                # Typical args: {source_place: "..."} (and/or source_room) plus optional object/object_category
                sp = args.get("source_place") or args.get("place")
                if sp and not slots.get("source_place"):
                    slots["source_place"] = sp
                sr = args.get("source_room") or args.get("room")
                if sr and not slots.get("source_room"):
                    slots["source_room"] = sr
