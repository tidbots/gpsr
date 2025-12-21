#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_parser.py

GPSR command parser.
- clauseベースで step を作る
- it/them などの参照は gpsr_parser_node.py 側で state 補完する前提
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re


# ================= 正規化ユーティリティ =================

def normalize_text(text: str) -> str:
    t = text.strip()
    t = t.lower()
    t = re.sub(r"[!?]", "", t)
    t = t.replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def split_into_clauses(text: str) -> List[str]:
    t = normalize_text(text).strip().strip(".")
    t = re.sub(r"\band then\b", " then ", t)
    t = re.sub(r"\bthen\b", " | ", t)
    t = re.sub(r"\band\b", " | ", t)
    parts = [p.strip() for p in t.split("|")]
    return [p for p in parts if p]


def best_match(token: str, candidates: List[str]) -> Optional[str]:
    token = token.strip().lower()
    if not token:
        return None

    exact = [c for c in candidates if token == c.strip().lower()]
    if exact:
        return max(exact, key=lambda s: len(s.strip()))

    scored = []
    for c in candidates:
        cl = c.strip().lower()
        if not cl:
            continue
        score = -1
        if token.startswith(cl) or cl.startswith(token) or token.endswith(cl) or cl.endswith(token):
            score = 80
        if token in cl or cl in token:
            score = max(score, 60)
        if score >= 0:
            scored.append((score, len(cl), c))

    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


# ================= 内部表現 =================

@dataclass
class GpsrStep:
    action: str
    fields: Dict[str, Any]


@dataclass
class GpsrCommand:
    ok: bool
    need_confirm: bool
    intent_type: str
    command_kind: str
    steps: List[GpsrStep]


# ================= パーサ本体 =================

class GpsrParser:
    def __init__(
        self,
        person_names: List[str],
        location_names: List[str],
        placement_location_names: List[str],
        room_names: List[str],
        object_names: List[str],
        object_categories_plural: List[str],
        object_categories_singular: List[str],
    ) -> None:
        self.person_names = person_names
        self.location_names = location_names
        self.placement_location_names = placement_location_names
        self.room_names = room_names
        self.object_names = object_names
        self.cat_plur = object_categories_plural
        self.cat_sing = object_categories_singular

        self.find_verbs = ["find", "locate", "look for", "search for"]
        self.count_prefixes = ["tell me how many", "how many"]
        self.guide_verbs = ["guide", "lead", "escort"]
        self.bring_verbs = ["bring", "fetch", "get", "grasp", "take"]
        self.give_verbs = ["give", "hand", "deliver", "bring"]  # it系で使う

    # ================= helper =================
    def _strip_articles(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[\.;:!?]$", "", s)
        s = re.sub(r"^(?:a|an|the)\s+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def match_object_or_category(self, s: str) -> Optional[str]:
        token = self._strip_articles(s)
        if not token:
            return None
        cand = self.object_names + self.cat_sing + self.cat_plur
        m = best_match(token, cand)
        if m:
            return m
        # fallback（短い語句は残す）
        if len(token) <= 40 and re.fullmatch(r"[a-z0-9\- ]+", token) and 1 <= len(token.split()) <= 4:
            return token
        return None

    def match_object(self, s: str) -> Optional[str]:
        token = self._strip_articles(s)
        if not token:
            return None
        m = best_match(token, self.object_names)
        if m:
            return m
        if len(token) <= 40 and re.fullmatch(r"[a-z0-9\- ]+", token) and 1 <= len(token.split()) <= 4:
            return token
        return None

    def match_place(self, s: str) -> Optional[str]:
        token = self._strip_articles(s)
        if not token:
            return None
        return best_match(token, self.placement_location_names + self.location_names)

    def match_room(self, s: str) -> Optional[str]:
        token = self._strip_articles(s)
        if not token:
            return None
        return best_match(token, self.room_names)

    def match_name(self, s: str) -> Optional[str]:
        token = self._strip_articles(s)
        if not token:
            return None
        return best_match(token, self.person_names)

    # ================= entry =================
    def parse(self, text: str) -> Optional[GpsrCommand]:
        t = normalize_text(text)
        clauses = split_into_clauses(t)

        steps: List[GpsrStep] = []
        kinds: List[str] = []

        for c in clauses:
            step, kind = self._parse_clause(c)
            if step:
                steps.append(step)
            if kind:
                kinds.append(kind)

        if not steps:
            return GpsrCommand(ok=False, need_confirm=True, intent_type="other", command_kind="unknown", steps=[])

        intent_type = "composite"
        command_kind = kinds[0] if kinds else "unknown"

        if command_kind.startswith("bring") or "bring" in command_kind or "deliver" in command_kind:
            intent_type = "bring"
        elif command_kind.startswith("guide") or "guide" in command_kind or "follow" in command_kind:
            intent_type = "guide"
        elif command_kind.startswith("tell") or command_kind.startswith("count") or command_kind.startswith("answer"):
            intent_type = "answer"

        return GpsrCommand(ok=True, need_confirm=False, intent_type=intent_type, command_kind=command_kind, steps=steps)

    # ================= clause parser =================
    def _parse_clause(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        c = clause.strip().strip(".")
        if not c:
            return None, None

        for fn in [
            # 追加：it系（bring/give/deliver）
            self._parse_bring_it_to_me,
            self._parse_bring_it_to_person_in_room,
            self._parse_give_it_to_named_person,
            self._parse_give_it_to_person_in_room,

            # 既存
            self._parse_bring_me_obj_from_place,
            self._parse_find_obj_in_room,
            self._parse_place_obj_on_place,
            self._parse_take_object,
            self._parse_count_people_in_room,
            self._parse_answer_quiz_in_room,
            self._parse_guide_name_from_to,
            self._parse_go_to_location,
        ]:
            step, kind = fn(c)
            if step:
                return step, kind

        return None, None

    # -------------------------------------------------
    # NEW: bring/give/deliver it patterns
    # -------------------------------------------------
    def _parse_bring_it_to_me(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "bring it to me" / "deliver it to me"
        if not re.match(r"^(?:bring|deliver|give|hand)\s+(?:it|them)\s+to\s+me$", clause):
            return None, None
        # object は node 側が state から補完する
        return GpsrStep(action="bring_object_to_operator", fields={}), "bringItToMe"

    def _parse_bring_it_to_person_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "bring it to the sitting person in the office"
        m = re.match(r"^(?:bring|deliver)\s+(?:it|them)\s+to\s+the\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None
        person_filter = m.group(1).strip()
        room = self.match_room(m.group(2).strip())
        if not room:
            return None, None
        # deliver step（object は state補完）
        return (
            GpsrStep(action="deliver_object_to_person_in_room", fields={"room": room, "person_filter": person_filter}),
            "deliverItToPrsInRoom",
        )

    def _parse_give_it_to_named_person(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "give it to charlie" / "hand it to jules"
        m = re.match(r"^(?:give|hand|deliver)\s+(?:it|them)\s+to\s+(.+)$", clause)
        if not m:
            return None, None
        name_raw = m.group(1).strip()
        # "to the ..." へは別関数で処理したいので除外
        if name_raw.startswith("the "):
            return None, None
        name = self.match_name(name_raw) or name_raw
        # room 不明なので、SMACH側で person を探す前提のステップ
        return (
            GpsrStep(action="give_object_to_named_person", fields={"name": name}),
            "giveItToName",
        )

    def _parse_give_it_to_person_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "give it to the lying person in the kitchen"
        m = re.match(r"^(?:give|hand)\s+(?:it|them)\s+to\s+the\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None
        person_filter = m.group(1).strip()
        room = self.match_room(m.group(2).strip())
        if not room:
            return None, None
        return (
            GpsrStep(action="deliver_object_to_person_in_room", fields={"room": room, "person_filter": person_filter}),
            "giveItToPrsInRoom",
        )

    # -------------------------------------------------
    # existing patterns
    # -------------------------------------------------
    def _parse_bring_me_obj_from_place(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "bring me a red wine from the bedside table"
        m = re.search(r"^(?:bring|fetch|get|grasp|take)\s+me\s+(?:a|an|the)?\s*(.+?)\s+from the\s+(.+)$", clause)
        if not m:
            # "fetch a drink from the sink"
            m = re.search(r"^(?:bring|fetch|get|grasp|take)\s+(?:a|an|the)?\s*(.+?)\s+from the\s+(.+)$", clause)
            if not m:
                return None, None

        obj_str = m.group(1).strip()
        plc_str = m.group(2).strip()
        obj = self.match_object_or_category(obj_str)
        plc = self.match_place(plc_str)
        if not plc:
            return None, None

        fields = {"source_place": plc}
        if obj:
            fields["object"] = obj  # 後段で object/category に寄せられる想定でもOK
        return GpsrStep(action="bring_object_to_operator", fields=fields), "bringMeObjFromPlcmt"

    def _parse_find_obj_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.find_verbs):
            return None, None
        if "in the" not in clause:
            return None, None

        m = re.search(r"(?:find|locate|look for|search for)\s+(?:a|an|the)?\s*(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None

        obj_str = m.group(1).strip()
        room_str = m.group(2).strip()
        room = self.match_room(room_str)
        if not room:
            return None, None

        fields = {"room": room}
        obj = self.match_object_or_category(obj_str)
        if obj:
            fields["object_or_category"] = obj
        return GpsrStep(action="find_object_in_room", fields=fields), "findObjInRoom"

    def _parse_place_obj_on_place(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "put it on the refrigerator" / "place it on the bed"
        m = re.search(r"^(?:put|place)\s+(?:it|them)\s+(?:on|in)\s+the\s+(.+)$", clause)
        if m:
            plc = self.match_place(m.group(1).strip())
            if not plc:
                return None, None
            return GpsrStep(action="place_object_on_place", fields={"place": plc}), "placeObjOnPlcmt"

        # "place a drink on the cabinet"
        m = re.search(r"^(?:put|place)\s+(?:a|an|the)?\s*(.+?)\s+(?:on|in)\s+the\s+(.+)$", clause)
        if not m:
            return None, None
        obj = self.match_object_or_category(m.group(1).strip())
        plc = self.match_place(m.group(2).strip())
        if not plc:
            return None, None
        fields = {"place": plc}
        if obj:
            fields["object_or_category"] = obj
        return GpsrStep(action="place_object_on_place", fields=fields), "placeObjOnPlcmt"

    def _parse_take_object(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if clause in ("get it", "take it", "grab it", "pick it up", "get them", "take them"):
            return GpsrStep(action="take_object", fields={}), "takeObj"

        m = re.search(r"^(?:get|take|grasp|grab)\s+(?:a|an|the)?\s*(.+)$", clause)
        if m and "from the" not in clause and "in the" not in clause:
            obj = self.match_object_or_category(m.group(1).strip())
            if obj:
                return GpsrStep(action="take_object", fields={"object_or_category": obj}), "takeObj"
        return None, None

    def _parse_count_people_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(clause.startswith(p) for p in self.count_prefixes):
            return None, None
        m = re.search(r"(?:tell me how many|how many)\s+(.+?)\s+are in the\s+(.+)$", clause)
        if not m:
            return None, None
        ppl = self._strip_articles(m.group(1).strip())
        room = self.match_room(m.group(2).strip())
        if not room:
            return None, None
        return GpsrStep(action="count_persons_in_room", fields={"room": room, "person_filter_plural": ppl}), "countPrsInRoom"

    def _parse_answer_quiz_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not clause.startswith("answer"):
            return None, None
        m = re.search(r"answer (?:the )?(?:quiz|question)\s+of the\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None
        person_filter = m.group(1).strip()
        room = self.match_room(m.group(2).strip())
        if not room:
            return None, None
        return GpsrStep(action="answer_to_person_in_room", fields={"room": room, "person_filter": person_filter}), "answerToPrsInRoom"

    def _parse_guide_name_from_to(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(clause.startswith(v) for v in self.guide_verbs):
            return None, None
        m = re.search(r"^(?:guide|lead|escort)\s+(.+?)\s+from the\s+(.+?)\s+to the\s+(.+)$", clause)
        if not m:
            return None, None
        name = self.match_name(m.group(1).strip()) or m.group(1).strip()
        fr = self.match_place(m.group(2).strip())
        to = self.match_place(m.group(3).strip())
        if not fr or not to:
            return None, None
        return (
            GpsrStep(action="guide_named_person_from_place_to_place", fields={"name": name, "from_place": fr, "to_place": to}),
            "guideNameFromBeacToBeac",
        )

    def _parse_go_to_location(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        m = re.search(r"^(?:go|navigate|move)\s+to the\s+(.+)$", clause)
        if not m:
            return None, None
        dest = m.group(1).strip()
        room = self.match_room(dest)
        if room:
            return GpsrStep(action="go_to_location", fields={"room": room}), "goToLoc"
        plc = self.match_place(dest)
        if plc:
            return GpsrStep(action="go_to_location", fields={"location": plc}), "goToLoc"
        return None, None
