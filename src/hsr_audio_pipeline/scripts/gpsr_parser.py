#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_parser.py

RoboCup@Home GPSR command parser.

Whisper/Faster-Whisper の文字起こし結果 (英語テキスト) を、
gpsr_commands.CommandGenerator が使っているコマンド種別 (kind) と
スロット (room / object / category / person など) に逆変換する。

NOTE:
- it/them の参照解決（object省略補完）は gpsr_parser_node.py 側で行う前提。
  ここでは "bring it ..." などをステップとして生成することに集中する。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re


# ================= 正規化ユーティリティ =================

def normalize_text(text: str) -> str:
    """
    テキストを粗く正規化する（ASR用）。
    - lower
    - !? を除去
    - , は空白扱い
    - 余分な空白を潰す
    """
    t = text.strip()
    t = t.lower()
    t = re.sub(r"[!?]", "", t)
    t = t.replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def split_into_clauses(text: str) -> List[str]:
    """
    GPSR向け：then / and then を主な句切りにする。
    'and' 単体は名詞句や修飾の中で頻出なので、原則として句切りに使わない。

    ただし then の後に "get it and bring it ..." のような複合が出るので、
    その分解は _parse_clause_multi 側で行う。
    """
    t = normalize_text(text).strip().strip(".")
    t = re.sub(r"\band then\b", " then ", t)
    parts = [p.strip() for p in re.split(r"\bthen\b", t) if p.strip()]
    return parts


def best_match(token: str, candidates: List[str]) -> Optional[str]:
    """
    ASR ノイズを多少含んでいても、候補リストの中から最も近いものを返す。
    - 完全一致 > 前後方一致 > 部分一致
    - 同点は最長一致
    """
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
    """1つのサブタスク"""
    action: str
    fields: Dict[str, Any]


@dataclass
class GpsrCommand:
    ok: bool
    need_confirm: bool
    intent_type: str
    command_kind: str
    steps: List[GpsrStep]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "need_confirm": self.need_confirm,
            "intent_type": self.intent_type,
            "command_kind": self.command_kind,
            "steps": [{"action": s.action, "args": s.fields} for s in self.steps],
        }


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
        # 語彙（ジェネレータと同じものを渡す）
        self.person_names = person_names
        self.location_names = location_names
        self.placement_location_names = placement_location_names
        self.room_names = room_names
        self.object_names = object_names
        self.cat_plur = object_categories_plural
        self.cat_sing = object_categories_singular

        # よく使う prefix
        self.find_verbs = ["find", "locate", "look for", "search for"]
        self.count_prefixes = ["tell me how many", "how many"]
        self.tell_prefixes = ["tell me what is", "tell me what", "tell me the"]
        self.answer_prefixes = ["answer", "answer the"]
        self.guide_verbs = ["guide", "lead", "escort"]
        self.follow_verbs = ["follow", "look for"]
        self.bring_verbs = ["bring", "fetch", "get", "grasp", "take"]

    # ================= helper: robust matching =================
    def _strip_articles(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"[\.;:!?]$", "", s)
        s = re.sub(r"^(?:a|an|the)\s+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def match_object_or_category(self, s: str) -> Optional[str]:
        """object/category の最良マッチ。語彙が壊れていても短い語句は落とさない。"""
        token = self._strip_articles(s)
        if not token:
            return None
        cand = self.object_names + self.cat_sing + self.cat_plur
        m = best_match(token, cand)
        if m:
            return m
        # fallback（1〜4語の短いフレーズは採用）
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

    # ================= エントリ =================

    def parse(self, text: str) -> Optional[GpsrCommand]:
        """
        文字列全体を受け取り、複数 clause に分割しながらステップに変換する。
        """
        t = normalize_text(text)
        clauses = split_into_clauses(t)

        steps: List[GpsrStep] = []
        kinds: List[str] = []

        for c in clauses:
            s_list, kind = self._parse_clause_multi(c)  # ★複数step対応
            if s_list:
                steps.extend(s_list)
            if kind:
                kinds.append(kind)

        if not steps:
            return GpsrCommand(
                ok=False,
                need_confirm=True,
                intent_type="other",
                command_kind="unknown",
                steps=[],
            )

        # 大まか intent_type を決める（先頭の kind ベース）
        intent_type = "composite"
        command_kind = kinds[0] if kinds else "unknown"

        if command_kind.startswith("bring") or command_kind.startswith("deliver") or "bring" in command_kind:
            intent_type = "bring"
        elif command_kind.startswith("guide") or "guide" in command_kind:
            intent_type = "guide"
        elif command_kind.startswith("follow") or "follow" in command_kind:
            intent_type = "guide"
        elif command_kind.startswith("tell") or command_kind.startswith("count") or command_kind.startswith("answer"):
            intent_type = "answer"

        return GpsrCommand(
            ok=True,
            need_confirm=False,
            intent_type=intent_type,
            command_kind=command_kind,
            steps=steps,
        )

    # ================= clause parser =================

    def _parse_clause_multi(self, clause: str) -> tuple[List[GpsrStep], Optional[str]]:
        """
        1 clause から複数 step を作れる版。
        典型： "get it and bring it to me"
        """
        c = clause.strip().strip(".")
        if not c:
            return [], None

        # まず "get it and <rest>" を分解（GPSRの典型）
        m = re.match(r"^(get it|take it|grab it|pick it up|get them|take them|grab them)\s+and\s+(.+)$", c)
        if m:
            first = m.group(1).strip()
            rest = m.group(2).strip()

            steps: List[GpsrStep] = []
            s1, k1 = self._parse_clause(first)
            if s1:
                steps.append(s1)

            s2, k2 = self._parse_clause(rest)
            if s2:
                steps.append(s2)

            # kind は後段優先（bring/deliver が取れたらそっち）
            return steps, (k2 or k1 or "takeObj")

        # その他は従来通り
        s, k = self._parse_clause(c)
        return ([s] if s else []), k

    def _parse_clause(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        c = clause.strip().strip(".")
        if not c:
            return None, None

        # order matters (more specific first)
        for fn in [
            # ★追加：it/them を目的語にした bring/deliver/give/hand
            self._parse_bring_it_to_me,
            self._parse_bring_it_to_person_in_room,

            # existing
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

    # ---- bring it to me (NEW) ----
    def _parse_bring_it_to_me(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "bring it to me" / "deliver it to me" / "give it to me" / "hand it to me"
        if not re.match(r"^(?:bring|deliver|give|hand)\s+(?:it|them)\s+to\s+me$", clause):
            return None, None
        # object は gpsr_parser_node.py 側が state から補完
        return GpsrStep(action="bring_object_to_operator", fields={}), "bringItToMe"

    # ---- bring it to the sitting person in the office (NEW) ----
    def _parse_bring_it_to_person_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "bring it to the sitting person in the office"
        m = re.match(r"^(?:bring|deliver|give|hand)\s+(?:it|them)\s+to\s+the\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None
        person_filter = m.group(1).strip()
        room = self.match_room(m.group(2).strip())
        if not room:
            return None, None
        return (
            GpsrStep(action="deliver_object_to_person_in_room", fields={"room": room, "person_filter": person_filter}),
            "deliverItToPrsInRoom",
        )

    # ---- bringMeObjFromPlcmt ----
    def _parse_bring_me_obj_from_place(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "bring me a red wine from the bedside table"
        m = re.search(r"^(?:bring|fetch|get|grasp|take)\s+me\s+(?:a|an|the)?\s*(.+?)\s+from the\s+(.+)$", clause)
        if not m:
            # "fetch a drink from the sink and bring it to me"
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
        # ここで obj がカテゴリでも落とさない（cleaning supply 対策）
        if obj:
            # bring は物体が自然だが、ASR的にカテゴリで来ても許容
            fields["object"] = obj

        step = GpsrStep(action="bring_object_to_operator", fields=fields)
        return step, "bringMeObjFromPlcmt"

    # ---- findObjInRoom ----
    def _parse_find_obj_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.find_verbs):
            return None, None
        if "in the" not in clause:
            return None, None

        # "find a cleaning supply in the bedroom"
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

        step = GpsrStep(action="find_object_in_room", fields=fields)
        return step, "findObjInRoom"

    # ---- placeObjOnPlcmt ----
    def _parse_place_obj_on_place(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "put it on the refrigerator" / "place it on the bed"
        m = re.search(r"^(?:put|place)\s+(?:it|them)\s+(?:on|in)\s+the\s+(.+)$", clause)
        if not m:
            # "place a drink on the cabinet" みたいな直接指定も拾う
            m = re.search(r"^(?:put|place)\s+(?:a|an|the)?\s*(.+?)\s+(?:on|in)\s+the\s+(.+)$", clause)
            if not m:
                return None, None
            # 物体指定あり
            obj = self.match_object_or_category(m.group(1).strip())
            plc = self.match_place(m.group(2).strip())
            if not plc:
                return None, None
            fields = {"place": plc}
            if obj:
                fields["object"] = obj
            return GpsrStep(action="place_object_on_place", fields=fields), "placeObjOnPlcmt"

        plc_str = m.group(1).strip()
        plc = self.match_place(plc_str)
        if not plc:
            return None, None

        step = GpsrStep(action="place_object_on_place", fields={"place": plc})
        return step, "placeObjOnPlcmt"

    # ---- takeObj ----
    def _parse_take_object(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "get it" / "take it"
        if clause in ("get it", "take it", "grab it", "pick it up", "get them", "take them", "grab them"):
            return GpsrStep(action="take_object", fields={}), "takeObj"
        # "get a drink" のような単独
        m = re.search(r"^(?:get|take|grasp|grab)\s+(?:a|an|the)?\s*(.+)$", clause)
        if m and "from the" not in clause and "in the" not in clause:
            obj = self.match_object_or_category(m.group(1).strip())
            if obj:
                return GpsrStep(action="take_object", fields={"object_or_category": obj}), "takeObj"
        return None, None

    # ---- countPrsInRoom ----
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

        fields = {"room": room, "person_filter_plural": ppl}
        return GpsrStep(action="count_persons_in_room", fields=fields), "countPrsInRoom"

    # ---- answerToPrsInRoom / quiz ----
    def _parse_answer_quiz_in_room(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "answer the quiz of the person raising their right arm in the office"
        if not clause.startswith("answer"):
            return None, None

        m = re.search(r"answer (?:the )?(?:quiz|question)\s+of the\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None

        person_filter = m.group(1).strip()
        room = self.match_room(m.group(2).strip())
        if not room:
            return None, None

        return (
            GpsrStep(action="answer_to_person_in_room", fields={"room": room, "person_filter": person_filter}),
            "answerToPrsInRoom",
        )

    # ---- guideNameFromBeacToBeac ----
    def _parse_guide_name_from_to(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "guide jules from the sink to the sofa"
        if not any(clause.startswith(v) for v in self.guide_verbs):
            return None, None
        m = re.search(r"^(?:guide|lead|escort)\s+(.+?)\s+from the\s+(.+?)\s+to the\s+(.+)$", clause)
        if not m:
            return None, None

        name_str = m.group(1).strip()
        fr_str = m.group(2).strip()
        to_str = m.group(3).strip()

        name = self.match_name(name_str) or name_str
        fr = self.match_place(fr_str)
        to = self.match_place(to_str)
        if not fr or not to:
            return None, None

        fields = {"name": name, "from_place": fr, "to_place": to}
        return GpsrStep(action="guide_named_person_from_place_to_place", fields=fields), "guideNameFromBeacToBeac"

    # ---- goToLoc / goToRoom ----
    def _parse_go_to_location(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "navigate to the living room"
        m = re.search(r"^(?:go|navigate|move)\s+to the\s+(.+)$", clause)
        if not m:
            return None, None
        dest = m.group(1).strip()
        # room優先
        room = self.match_room(dest)
        if room:
            return GpsrStep(action="go_to_location", fields={"room": room}), "goToLoc"
        plc = self.match_place(dest)
        if plc:
            return GpsrStep(action="go_to_location", fields={"location": plc}), "goToLoc"
        return None, None
