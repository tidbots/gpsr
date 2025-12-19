#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPSR Intent node with vocabulary "snap" layer.
- Subscribes: /asr/text (std_msgs/String)
- Publishes :
    /gpsr/normalized_text (std_msgs/String)  : snapped & normalized text
    /gpsr/intent          (std_msgs/String)  : JSON-like dict of parsed intent/slots or {need_confirm: true}

Key features:
- Damerau–Levenshtein による語彙吸着（bed side table→bedside table, dish washer→dishwasher など）
- よくある音韻誤りを quick fix（foot→food, past room→bathroom など）
- "* room" で終わる未知語は ROOMS へ最終吸着する保険
- スロットが既知語彙に収まらないほど信頼度を下げ、閾値超過で need_confirm を返す
"""
import json
import re
import unicodedata
import rospy
from std_msgs.msg import String

NODE_NAME = "gpsr_intent_node"

# === Fixed GPSR vocabulary (from user) ===
NAMES = [
    'Adel','Angel','Axel','Charlie','Jane','Jules','Morgan','Paris','Robin','Simone'
]
LOCATIONS = [
    'bed','bedside table','shelf','trashbin','dishwasher','potted plant','kitchen table',
    'chairs','pantry','refrigerator','sink','cabinet','coatrack','desk','armchair',
    'desk lamp','waste basket','tv stand','storage rack','lamp','side tables','sofa',
    'bookshelf','entrance','exit'
]
ROOMS = ['bedroom','kitchen','office','living room','bathroom']
OBJECTS = [
    'juice pack','cola','milk','orange juice','tropical juice','red wine','iced tea',
    'tennis ball','rubiks cube','baseball','soccer ball','dice','orange','pear','peach',
    'strawberry','apple','lemon','banana','plum','cornflakes','pringles','cheezit','cup',
    'bowl','fork','plate','knife','spoon','chocolate jello','coffee grounds','mustard',
    'tomato soup','tuna','strawberry jello','spam','sugar','cleanser','sponge'
]
CATEGORIES = [
    ('drink','drinks'), ('toy','toys'), ('fruit','fruits'),
    ('snack','snacks'), ('dish','dishes'), ('food','food'),
    ('cleaning supply','cleaning supplies')
]

CATEGORY_SINGULARS = [s for s,_ in CATEGORIES]
CATEGORY_PLURALS   = [p for _,p in CATEGORIES]
CATEGORY_ALL       = CATEGORY_SINGULARS + CATEGORY_PLURALS

VOCAB_PHRASES = sorted(set(NAMES + LOCATIONS + ROOMS + OBJECTS + CATEGORY_ALL),
                       key=lambda x: -len(x))  # 長い語から優先

# ------------------------
# Text normalization utils
# ------------------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dl_distance(a: str, b: str) -> int:
    """
    Simple Damerau–Levenshtein distance
    """
    d = {}
    len_a, len_b = len(a), len(b)
    for i in range(-1, len_a+1):
        d[(i, -1)] = i+1
    for j in range(-1, len_b+1):
        d[(-1, j)] = j+1
    for i in range(len_a):
        for j in range(len_b):
            cost = 0 if a[i] == b[j] else 1
            d[(i, j)] = min(
                d[(i-1, j)] + 1,
                d[(i, j-1)] + 1,
                d[(i-1, j-1)] + cost
            )
            if i and j and a[i] == b[j-1] and a[i-1] == b[j]:
                d[(i, j)] = min(d[(i, j)], d[(i-2, j-2)] + cost)
    return d[(len_a-1, len_b-1)]

def snap_phrase_to_vocab(phrase: str, vocab_phrases, rel_thresh=0.34) -> str:
    """
    単語/フレーズを所与の語彙に相対距離（dist/len）で吸着。閾値を超えたら元の文字列を返す。
    """
    p0 = phrase.strip().lower()
    best = (p0, 1.0)
    for v in vocab_phrases:
        dist = dl_distance(p0, v)
        rel = dist / float(max(1, len(v)))
        if rel < best[1]:
            best = (v, rel)
    return best[0] if best[1] <= rel_thresh else p0

def _snap_room_like(token: str) -> str:
    """
    "* room" で終わるが既知ROOMSに含まれない場合、ROOMSへ緩めに最終吸着（保険）。
    """
    t = token.strip().lower()
    if t.endswith(" room") and t not in ROOMS:
        return snap_phrase_to_vocab(t, ROOMS, rel_thresh=0.60)
    return t

def vocab_snap_text(text: str) -> str:
    """
    入力文全体に語彙吸着を適用。
    - quick fixes（頻出誤りの辞書置換）
    - 左から貪欲な n-gram（最大4語）で既知フレーズへ吸着
    - 最後に "* room" の保険吸着
    """
    t = " " + normalize_text(text) + " "

    # --- quick common fixes（まずは辞書置換で大きな崩れを潰す）---
    t = t.replace(" foot ", " food ").replace(" dish washer ", " dishwasher ")
    # bathroom 誤認（音韻的に起きやすい）
    for wrong in [" past room ", " bus room ", " bad room "]:
        t = t.replace(wrong, " bathroom ")

    words = t.split()
    out = []
    i = 0
    while i < len(words):
        snapped = None
        best_len = 1
        # 長い n-gram から試す（最大4語）
        for n in range(4, 0, -1):
            if i + n > len(words):
                continue
            span = " ".join(words[i:i+n])
            candidate = snap_phrase_to_vocab(span, VOCAB_PHRASES, rel_thresh=0.34)
            if candidate in VOCAB_PHRASES:
                snapped = candidate
                best_len = n
                break
        out.append(snapped if snapped else words[i])
        i += best_len

    # 連結と最終クリーニング
    t2 = " ".join(out)
    t2 = re.sub(r"\s+", " ", t2).strip()

    # 最後の保険：未知の "* room" を既知ROOMSに吸着
    toks = t2.split()
    for k, w in enumerate(toks):
        if w.endswith("room"):
            joined = w
            # "living room" のような2語構成もあるので近傍を確認
            if k > 0 and toks[k-1] == "living":
                joined = "living room"
                toks[k-1] = joined
                toks[k] = ""
            else:
                joined = _snap_room_like(w if " " in w else f"{w}")
                toks[k] = joined
    t3 = " ".join([x for x in toks if x])
    return t3

# ------------------------
# Grammar patterns（代表例）
# ------------------------
GRAMMARS = [
    # tell me how many X there are on Y
    ("count_on",
     re.compile(r"\btell me how many (?P<cat>\w+(?: \w+)*) (?:there are |there're |are )?on (?:the )?(?P<loc>.+)$")),

    # bring/fetch object from source to me|here
    ("bring_from_to_me",
     re.compile(r"\b(bring|fetch|get) (?:me )?(?:a |an |the )?(?P<obj>.+?) from (?:the )?(?P<src>.+?) (?:to|into|onto)? (?:me|here)$")),

    # fetch category（場所や人数が後続で省略される系）
    ("fetch_category",
     re.compile(r"\b(fetch|get|bring) (?:me )?(?:some )?(?P<cat>.+)$")),

    # navigate then look for and give
    ("navigate_look_give",
     re.compile(r"\bnavigate to (?:the )?(?P<loc>.+?) then look for (?:a |an |the )?(?P<obj>.+?) and (?:grasp|pick) it and (?:give|deliver) it to (?:the )?(?P<who>.+)$")),

    # tell the gesture of the person at A to the person at B
    ("tell_gesture",
     re.compile(r"\btell (?:me )?the gesture of (?:the )?person at (?:the )?(?P<src>.+?) to (?:the )?person at (?:the )?(?P<dst>.+)$")),

    # tell the name of the person at A to the person at B
    ("tell_name",
     re.compile(r"\btell (?:me )?the name of (?:the )?person at (?:the )?(?P<src>.+?) to (?:the )?person at (?:the )?(?P<dst>.+)$")),

    # tell me how many people in the ROOM are wearing X
    ("count_people_wearing",
     re.compile(r"\btell me how many people in (?:the )?(?P<room>.+?) are wearing (?P<what>.+)$")),
]

# ------------------------
# Slot snapping helpers
# ------------------------
def best_category_token(s: str):
    s_norm = s.strip().lower()
    for c in CATEGORY_ALL:
        if s_norm == c:
            return c
    for c in CATEGORY_ALL:
        if c in s_norm:
            return c
    return snap_phrase_to_vocab(s_norm, CATEGORY_ALL, rel_thresh=0.34)

def best_location_phrase(s: str):
    snapped = snap_phrase_to_vocab(s, LOCATIONS + ROOMS, rel_thresh=0.34)
    # "* room" で未知なら最後に ROOMS へ保険吸着
    snapped = _snap_room_like(snapped)
    return snapped

def best_object_phrase(s: str):
    return snap_phrase_to_vocab(s, OBJECTS, rel_thresh=0.34)

# ------------------------
# Interpreter
# ------------------------
def interpret(text: str):
    """
    Return (intent, slots, confidence, snapped_text)
    confidence: 小さいほど良い（距離に相当）。簡易に OOV の数で加点。
    """
    t = vocab_snap_text(text)
    for intent, pat in GRAMMARS:
        m = pat.search(t)
        if not m:
            continue
        slots = {k: v.strip() for k, v in m.groupdict().items() if v}
        conf = 0.0
        if intent == "count_on":
            slots["category"] = best_category_token(slots.pop("cat", ""))
            slots["location"] = best_location_phrase(slots.pop("loc", ""))
        elif intent == "bring_from_to_me":
            slots["object"] = best_object_phrase(slots.pop("obj", ""))
            slots["source"] = best_location_phrase(slots.pop("src", ""))
        elif intent == "fetch_category":
            slots["category"] = best_category_token(slots.pop("cat", ""))
        elif intent == "navigate_look_give":
            slots["location"] = best_location_phrase(slots.pop("loc", ""))
            slots["object"] = best_object_phrase(slots.pop("obj", ""))
            slots["recipient"] = slots.pop("who", "lying person in the bathroom")
        elif intent in ("tell_gesture", "tell_name"):
            slots["source"] = best_location_phrase(slots.pop("src", ""))
            slots["destination"] = best_location_phrase(slots.pop("dst", ""))
        elif intent == "count_people_wearing":
            slots["room"] = best_location_phrase(slots.pop("room", ""))
            # what は自由語（wearing white sweaters 等）

        # 簡易 confidence：既知語彙に入らないスロットごとに罰点
        for v in slots.values():
            if v not in VOCAB_PHRASES and v not in CATEGORY_ALL:
                conf += 0.5
        return intent, slots, max(conf, 0.0), t

    # パターン外
    return None, {}, 9e9, t

# ------------------------
# ROS Node
# ------------------------
class GPSRIntentNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)
        self.pub_norm = rospy.Publisher("/gpsr/normalized_text", String, queue_size=10)
        self.pub_intent = rospy.Publisher("/gpsr/intent", String, queue_size=10)
        self.sub = rospy.Subscriber("/asr/text", String, self.cb, queue_size=10)
        # 閾値は運用で調整（0.8〜1.5 目安）
        self.confirm_threshold = rospy.get_param("~confirm_threshold", 1.25)
        rospy.loginfo("[%s] Ready. Subscribed to /asr/text", NODE_NAME)

    def cb(self, msg: String):
        text = msg.data or ""
        intent, slots, conf, snapped = interpret(text)
        self.pub_norm.publish(String(data=snapped))
        if intent and conf <= self.confirm_threshold:
            payload = {"intent": intent, "slots": slots, "confidence": conf}
        else:
            payload = {"need_confirm": True, "heard": snapped}
        self.pub_intent.publish(String(data=json.dumps(payload)))

if __name__ == "__main__":
    try:
        node = GPSRIntentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
