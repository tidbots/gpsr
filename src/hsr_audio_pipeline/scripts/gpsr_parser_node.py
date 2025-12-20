#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import rospy

from std_msgs.msg import String, Bool, Float32

# ---- PATH FIX ----
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
# ------------------

from gpsr_parser import GpsrParser


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
        # ===============================
        # ROS params
        # ===============================
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")
        self.vocab_yaml = rospy.get_param("~vocab_yaml", None)
        self.lang = rospy.get_param("~lang", "en")

        self._latest_text = ""
        self._latest_conf = None
        self._latest_text_stamp = rospy.Time(0)

        # ===============================
        # Load vocabulary
        # ===============================
        vocab = self._load_vocab(self.vocab_yaml)

        # ===============================
        # Build GPSR parser
        # ===============================
        self.parser = GpsrParser(
            person_names=vocab["person_names"],
            location_names=vocab["location_names"],
            placement_location_names=vocab["placement_location_names"],
            room_names=vocab["room_names"],
            object_names=vocab["object_names"],
            object_categories_plural=vocab["object_categories_plural"],
            object_categories_singular=vocab["object_categories_singular"],
        )

        self._all_places = vocab["location_names"]

        # ===============================
        # ROS I/O
        # ===============================
        self.pub_intent = rospy.Publisher(self.intent_topic, String, queue_size=10)
        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)

        try:
            rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)
        except Exception:
            pass

        rospy.loginfo("gpsr_parser_node ready (vocab=%s)",
                      self.vocab_yaml if self.vocab_yaml else "internal defaults")

    # ==========================================================
    # Vocabulary loader
    # ==========================================================
    def _load_vocab(self, yaml_path):
        if yaml_path and os.path.exists(yaml_path):
            rospy.loginfo("Loading GPSR vocab from YAML: %s", yaml_path)
            with open(yaml_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)

            names = y.get("names", [])
            rooms = y.get("rooms", [])

            location_names = []
            placement_location_names = []

            for loc in y.get("locations", []):
                name = loc["name"]
                location_names.append(name)
                if loc.get("placement", False):
                    placement_location_names.append(name)

            object_names = []
            cats_s = []
            cats_p = []

            for c in y.get("categories", []):
                cats_s.append(c["singular"])
                cats_p.append(c["plural"])
                for o in c.get("objects", []):
                    object_names.append(o.replace("_", " "))

            return dict(
                person_names=names,
                room_names=rooms,
                location_names=location_names,
                placement_location_names=placement_location_names,
                object_names=sorted(set(object_names)),
                object_categories_singular=sorted(set(cats_s)),
                object_categories_plural=sorted(set(cats_p)),
            )

        # ---------- fallback ----------
        rospy.logwarn("vocab_yaml not set or not found → using built-in defaults")
        return self._default_vocab()

    def _default_vocab(self):
        return dict(
            person_names=["Adel", "Angel", "Axel", "Charlie", "Jane", "Jules", "Morgan", "Paris", "Robin", "Simone"],
            room_names=["bedroom", "kitchen", "living room", "office", "bathroom"],
            location_names=["bed", "bedside table", "shelf", "trashbin", "dishwasher",
                            "kitchen table", "pantry", "refrigerator", "sink", "cabinet",
                            "desk", "tv stand", "storage rack", "side tables", "sofa", "bookshelf"],
            placement_location_names=["bed", "bedside table", "shelf", "dishwasher",
                                      "kitchen table", "pantry", "refrigerator", "sink",
                                      "cabinet", "desk", "tv stand", "storage rack",
                                      "side tables", "sofa", "bookshelf"],
            object_names=["red wine", "milk", "cola", "juice", "sponge", "cleanser"],
            object_categories_singular=["drink", "food", "cleaning supply"],
            object_categories_plural=["drinks", "food", "cleaning supplies"],
        )

    # ==========================================================
    # Callbacks
    # ==========================================================
    def _on_text(self, msg):
        self._latest_text = msg.data.strip()
        self._latest_text_stamp = rospy.Time.now()

    def _on_conf(self, msg):
        self._latest_conf = msg.data

    def _on_utt_end(self, msg):
        if not msg.data or not self._latest_text:
            return

        raw = self._latest_text
        rospy.loginfo("parse: %s", raw)

        try:
            parsed = self.parser.parse(raw)
        except Exception as e:
            rospy.logerr("parse failed: %s", e)
            return

        payload = self._coerce_to_v1(parsed, raw)
        self._publish(payload)

    # ==========================================================
    # Intent shaping
    # ==========================================================
    def _coerce_to_v1(self, parsed, raw_text):
        if isinstance(parsed, str):
            parsed = json.loads(parsed)

        payload = {
            "schema": "gpsr_intent_v1",
            "ok": parsed.get("ok", True),
            "need_confirm": parsed.get("need_confirm", False),
            "intent_type": parsed.get("intent_type", "other"),
            "raw_text": raw_text,
            "normalized_text": raw_text.lower(),
            "confidence": self._latest_conf,
            "source": "parser",
            "command_kind": parsed.get("command_kind"),
            "steps": [],
            "slots": {k: None for k in self.FIXED_SLOT_KEYS},
            "extras": {"legacy_slots": parsed.get("slots", {})},
            "context": {"lang": self.lang, "source": "parser"},
        }

        for s in parsed.get("steps", []):
            payload["steps"].append({
                "action": s["action"],
                "args": s.get("args", s.get("fields", {}))
            })

        return payload

    def _publish(self, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(msg)


def main():
    rospy.init_node("gpsr_parser_node")
    GpsrParserNode()
    rospy.spin()


if __name__ == "__main__":
    main()
