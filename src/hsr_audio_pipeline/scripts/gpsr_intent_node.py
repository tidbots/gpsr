#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
import re
from std_msgs.msg import String


# === RoboCup@Home 用語彙（すべて小文字で比較） ===

NAMES = [
    "adel", "angel", "axel", "charlie", "jane", "jules",
    "morgan", "paris", "robin", "simone",
]

LOCATIONS = [
    "bed", "bedside table", "shelf", "trashbin", "dishwasher", "potted plant",
    "kitchen table", "chairs", "pantry", "refrigerator", "sink", "cabinet",
    "coatrack", "desk", "armchair", "desk lamp", "waste basket", "tv stand",
    "storage rack", "lamp", "side tables", "sofa", "bookshelf",
    "entrance", "exit",
]

PICKABLE_LOCATIONS = [
    "bed", "bedside table", "shelf", "dishwasher", "kitchen table", "pantry",
    "refrigerator", "sink", "cabinet", "desk", "tv stand", "storage rack",
    "side tables", "sofa", "bookshelf",
]

ROOMS = ["bedroom", "kitchen", "office", "living room", "bathroom"]

OBJECTS = [
    "juice pack", "cola", "milk", "orange juice", "tropical juice", "red wine",
    "iced tea", "tennis ball", "rubiks cube", "baseball", "soccer ball", "dice",
    "orange", "pear", "peach", "strawberry", "apple", "lemon", "banana", "plum",
    "cornflakes", "pringles", "cheezit", "cup", "bowl", "fork", "plate", "knife",
    "spoon", "chocolate jello", "coffee grounds", "mustard", "tomato soup",
    "tuna", "strawberry jello", "spam", "sugar", "cleanser", "sponge",
]

CATEGORIES = {
    "drink": ["drinks"],
    "toy": ["toys"],
    "fruit": ["fruits"],
    "snack": ["snacks"],
    "dish": ["dishes"],
    "food": ["food"],  # 単複同形
    "cleaning supply": ["cleaning supplies"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def find_first_match(text: str, candidates):
    """
    text に含まれている candidates（単語またはフレーズ）のうち
    一番左に出現するものを返す（見つからなければ ""）。
    """
    text_l = text.lower()
    best = ""
    best_pos = len(text_l) + 1
    for c in candidates:
        pos = text_l.find(c)
        if pos != -1 and pos < best_pos:
            best = c
            best_pos = pos
    return best


def detect_category(text: str):
    text_l = text.lower()
    for singular, plurals in CATEGORIES.items():
        if singular in text_l:
            return singular, singular  # "food" など
        for p in plurals:
            if p in text_l:
                return singular, p
    return "", ""


class GpsrIntentNode(object):
    """
    /asr/text に流れてくる音声認識結果（基本は英語）を受け取り、
    /gpsr/intent に「意図っぽい情報」を JSON 文字列で publish するノード。

    intent = {
      "raw_text": <認識テキスト>,
      "intent_type": "bring|guide|answer|other",
      "slots": {
        "object": "",
        "object_category": "",
        "source": "",
        "source_room": "",
        "destination": "",
        "destination_room": "",
        "person": "",
        "person_at_source": "",
        "person_at_destination": "",
        "question_type": "",
        "attribute": "",
        "comparison": "",
        "pose": "",
        "count_target": "",
      }
    }
    """

    def __init__(self):
        asr_topic = rospy.get_param("~asr_topic", "/asr/text")
        intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

        self.pub_intent = rospy.Publisher(intent_topic, String, queue_size=10)
        rospy.Subscriber(asr_topic, String, self.asr_callback)

        rospy.loginfo("GpsrIntentNode: subscribing %s, publishing %s",
                      asr_topic, intent_topic)

    def asr_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        rospy.loginfo("GpsrIntentNode: received ASR text: '%s'", text)

        intent_type, slots = self.interpret_command(text)

        intent = {
            "raw_text": text,
            "intent_type": intent_type,
            "slots": slots,
        }
        intent_json = json.dumps(intent, ensure_ascii=False)
        rospy.loginfo("GpsrIntentNode: intent=%s", intent_json)
        self.pub_intent.publish(String(data=intent_json))

    # === メインの解釈関数 ===

    def interpret_command(self, text: str):
        t_norm = normalize(text)

        # 初期スロット
        slots = {
            "object": "",
            "object_category": "",
            "source": "",
            "source_room": "",
            "destination": "",
            "destination_room": "",
            "person": "",
            "person_at_source": "",
            "person_at_destination": "",
            "question_type": "",
            "attribute": "",
            "comparison": "",
            "pose": "",
            "count_target": "",
        }

        intent_type = "other"

        # --- 1) how many ... 系 ---
        if "how many" in t_norm:
            intent_type = "answer"

            # 人数カウント（部屋／姿勢含む）
            if "people in the" in t_norm or "persons in the" in t_norm or "lying persons" in t_norm:
                slots["question_type"] = "count_people"
                slots["count_target"] = "people"

                # 部屋
                room = find_first_match(t_norm, ROOMS)
                if room:
                    slots["source_room"] = room

                # 姿勢
                if "lying" in t_norm:
                    slots["pose"] = "lying"

                # 服装などの属性（"wearing ..." の後ろを全部）
                m = re.search(r"wearing (.+)", t_norm)
                if m:
                    slots["attribute"] = "wearing " + m.group(1).strip()

                return intent_type, slots

            # 物体カテゴリのカウント
            cat, cat_word = detect_category(t_norm)
            if cat:
                slots["question_type"] = "count_objects"
                slots["object_category"] = cat
                slots["count_target"] = cat_word
                loc = find_first_match(t_norm, LOCATIONS)
                if loc:
                    slots["source"] = loc
                return intent_type, slots

        # --- 2) biggest/lightest ... cleaning supply on X ---
        if ("biggest" in t_norm or "lightest" in t_norm) and "cleaning" in t_norm:
            intent_type = "answer"
            slots["question_type"] = "compare_objects"
            slots["object_category"] = "cleaning supply"
            slots["comparison"] = "biggest" if "biggest" in t_norm else "lightest"
            loc = find_first_match(t_norm, LOCATIONS)
            if loc:
                slots["source"] = loc
            return intent_type, slots

        # --- 3) tell the gesture/name ... to the person at ... ---
        if t_norm.startswith("tell the gesture of the person at"):
            intent_type = "answer"
            slots["question_type"] = "person_attribute"
            slots["attribute"] = "gesture"

            # "person at the X to the person at the Y"
            src_loc = find_first_match(t_norm, LOCATIONS)
            # "to the person at the Y" の Y を探す
            # 一番後ろの location を destination とみなす
            dst_loc = ""
            for loc in LOCATIONS:
                if "to the person at the " + loc in t_norm:
                    dst_loc = loc
                    break

            if src_loc:
                slots["source"] = src_loc
                slots["person_at_source"] = f"person at the {src_loc}"
            if dst_loc:
                slots["destination"] = dst_loc
                slots["person_at_destination"] = f"person at the {dst_loc}"
            return intent_type, slots

        if t_norm.startswith("tell the name of the person at"):
            intent_type = "answer"
            slots["question_type"] = "person_attribute"
            slots["attribute"] = "name"

            src_loc = find_first_match(t_norm, LOCATIONS)
            dst_loc = ""
            for loc in LOCATIONS:
                if "to the person at the " + loc in t_norm:
                    dst_loc = loc
                    break

            if src_loc:
                slots["source"] = src_loc
                slots["person_at_source"] = f"person at the {src_loc}"
            if dst_loc:
                slots["destination"] = dst_loc
                slots["person_at_destination"] = f"person at the {dst_loc}"
            return intent_type, slots

        # --- 4) bring / fetch 系 ---
        if t_norm.startswith("bring") or t_norm.startswith("fetch"):
            intent_type = "bring"

            # 物体 or カテゴリ
            obj = find_first_match(t_norm, OBJECTS)
            if obj:
                slots["object"] = obj
            cat, cat_word = detect_category(t_norm)
            if cat and not slots["object"]:
                slots["object_category"] = cat
                slots["object"] = cat  # とりあえずカテゴリ名を入れておく

            # source
            src_loc = find_first_match(t_norm, PICKABLE_LOCATIONS)
            if src_loc:
                slots["source"] = src_loc

            # destination
            if "to me" in t_norm or "to the operator" in t_norm:
                slots["destination"] = "operator"
                slots["person"] = "operator"

            return intent_type, slots

        # --- 5) navigate ... then look for X and grasp it and give it to ... ---
        if t_norm.startswith("navigate to"):
            # とりあえず bring として扱う（ナビ＋ピック＆プレース）
            intent_type = "bring"

            # "navigate to the desk lamp"
            nav_loc = ""
            for loc in LOCATIONS:
                if f"navigate to the {loc}" in t_norm or f"navigate to {loc}" in t_norm:
                    nav_loc = loc
                    break
            if nav_loc:
                slots["source"] = nav_loc  # 最初に向かう場所

            # tuna などのオブジェクト
            obj = find_first_match(t_norm, OBJECTS)
            if obj:
                slots["object"] = obj

            # "lying person in the bathroom" のような表現
            if "lying person" in t_norm and "bathroom" in t_norm:
                slots["destination_room"] = "bathroom"
                slots["pose"] = "lying"
                slots["person"] = "lying person in the bathroom"
                slots["person_at_destination"] = "lying person in the bathroom"

            return intent_type, slots

        # --- 6) それ以外の質問系 "tell me what/..." ---
        if t_norm.startswith("tell me"):
            intent_type = "answer"
            # ここではとりあえず question_type などは空のまま返す
            return intent_type, slots

        # デフォルト
        return intent_type, slots


def main():
    rospy.init_node("gpsr_intent_node")
    node = GpsrIntentNode()
    rospy.loginfo("GpsrIntentNode started.")
    rospy.spin()


if __name__ == "__main__":
    main()
