#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_parser.py

RoboCup@Home GPSR command parser.

Whisper/Faster-Whisper の文字起こし結果 (英語テキスト) を、
gpsr_commands.CommandGenerator が使っているコマンド種別 (kind) と
スロット (room / object / category / person など) に逆変換する。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import re


# ================= 共通ユーティリティ =================

def normalize_text(text: str) -> str:
    """ASR 出力をパースしやすいように正規化。"""
    t = text.strip()
    t = t.lower()
    # 句読点など最低限の削除
    t = re.sub(r"[!?]", "", t)
    # カンマは空白扱い
    t = t.replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    return t


def split_into_clauses(text: str) -> List[str]:
    """
    "then" / "and then" / "and" でざっくり節に分割する。
    例:
        "find a sponge in the living room then get it and bring it to me"
        -> ["find a sponge in the living room",
            "get it",
            "bring it to me"]
    """
    # "then" を優先して区切る
    tmp = re.split(r"\bthen\b", text)
    clauses: List[str] = []
    for part in tmp:
        part = part.strip()
        if not part:
            continue
        # " and " も試しに区切るが、「and bring it to me」は 1 節扱いにしたい
        # ので "and bring it to me" のようなパターンはまとめて扱う。
        sub = re.split(r"\band\b", part)
        for s in sub:
            s = s.strip()
            if not s:
                continue
            clauses.append(s)
    return clauses


def best_match(token: str, candidates: List[str]) -> Optional[str]:
    """
    ASR ノイズを多少含んでいても、候補リストの中から最も近いものを返す。

    改善点（重要）:
    - 候補順に依存しない（"bed" が先にあっても "bedside table" を取れる）
    - 完全一致 > 前後方一致 > 部分一致
    - 同点の場合は「最長一致」を優先
    """
    token = token.strip().lower()
    if not token:
        return None

    # 1) 完全一致（最優先）
    exact = [c for c in candidates if token == c.strip().lower()]
    if exact:
        return max(exact, key=lambda s: len(s.strip()))

    scored = []  # (score, length, candidate)
    for c in candidates:
        cl = c.strip().lower()
        if not cl:
            continue

        score = -1

        # 2) 前方/後方一致（強い）
        if token.startswith(cl) or cl.startswith(token) or token.endswith(cl) or cl.endswith(token):
            score = 80

        # 3) 部分一致（弱い）
        if token in cl or cl in token:
            score = max(score, 60)

        if score >= 0:
            scored.append((score, len(cl), c))

    if not scored:
        return None

    # score 最大、同点なら length 最大（= 最長一致）を採用
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


# ================= 内部表現 =================

@dataclass
class GpsrStep:
    """1つのサブタスク（節）に対応する意味表現。"""
    action: str
    fields: Dict[str, Any]


@dataclass
class GpsrCommand:
    """1発話に対応するコマンドの構造化表現。"""
    kind: str                 # CommandGenerator のコマンド名 (goToLoc, findObjInRoom, ...)
    steps: List[GpsrStep]     # then / and で分割したサブタスク
    raw_text: str             # 元テキスト

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "steps": [
                    {"action": s.action, "fields": s.fields} for s in self.steps
                ],
                "raw_text": self.raw_text,
            },
            ensure_ascii=False,
        )


# ================= パーサ本体 =================

class GpsrParser:
    """
    gpsr_commands.CommandGenerator のテンプレートを“逆向き”に解釈するパーサ。
    - すべてのコマンド種別をカバー
    - 動詞・前置詞などはキーワードの組合せで判定
    """

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

        # People commands
        self.person_kinds = [
            "goToLoc",
            "findPrsInRoom",
            "meetPrsAtBeac",
            "countPrsInRoom",
            "tellPrsInfoInLoc",
            "talkInfoToGestPrsInRoom",
            "answerToGestPrsInRoom",
            "followNameFromBeacToRoom",
            "guideNameFromBeacToBeac",
            "guidePrsFromBeacToBeac",
            "guideClothPrsFromBeacToBeac",
            "greetClothDscInRm",
            "greetNameInRm",
            "meetNameAtLocThenFindInRm",
            "countClothPrsInRoom",
            "tellPrsInfoAtLocToPrsAtLoc",
            "followPrsAtLoc",
        ]

        # Object commands
        self.object_kinds = [
            "goToLoc",
            "takeObjFromPlcmt",
            "findObjInRoom",
            "countObjOnPlcmt",
            "tellObjPropOnPlcmt",
            "bringMeObjFromPlcmt",
            "tellCatPropOnPlcmt",
        ]

        # 動詞・表現（CommandGenerator の verb_dict と概ね揃える）
        self.go_verbs = ["go", "navigate", "move"]
        self.find_verbs = ["find", "locate", "look for", "search for"]
        self.take_verbs = ["take", "get", "grasp", "fetch", "pick up"]
        self.place_verbs = ["put", "place"]
        self.deliver_verbs = ["bring", "give", "deliver", "hand"]
        self.count_prefixes = ["tell me how many", "how many"]
        self.tell_verbs = ["tell", "inform", "say", "describe"]
        self.meet_verbs = ["meet"]
        self.answer_verbs = ["answer"]
        self.follow_verbs = ["follow"]
        self.guide_verbs = ["guide", "escort", "take", "lead"]
        self.greet_verbs = ["greet", "salute", "say hello to", "introduce yourself to"]

    # ---------- 公開 API ----------

    def parse(self, text: str) -> Optional[GpsrCommand]:
        """
        Whisper の 1 文を受け取り、GpsrCommand にパースする。
        - then/and で節に分割 → 各節を GpsrStep として解釈
        - kind は最初の節の構造から決定
        """
        if not text:
            return None

        norm = normalize_text(text)
        clauses = split_into_clauses(norm)
        if not clauses:
            return None

        steps: List[GpsrStep] = []

        # 1) 最初の節 → トップレベル kind を決定
        first_clause = clauses[0]
        first_step, kind = self._parse_top_level(first_clause)
        if first_step is None or kind is None:
            return None
        steps.append(first_step)

        # 2) 残りの節 → followup として解釈
        for c in clauses[1:]:
            st = self._parse_followup_clause(c, previous_step=steps[-1])
            if st is not None:
                if isinstance(st, list):
                    steps.extend(st)
                else:
                    steps.append(st)

        return GpsrCommand(kind=kind, steps=steps, raw_text=text)

    # ---------- トップレベルの判定 ----------

    def _parse_top_level(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        """
        文の最初の節から「どのコマンド種別か」を推定し、
        その節の GpsrStep と kind を返す。
        """
        # 各コマンドの専用パーサを順番に試す
        parsers = [
            self._parse_go_to_loc,
            self._parse_find_obj_in_room,
            self._parse_take_obj_from_plcmt,
            self._parse_count_obj_on_plcmt,
            self._parse_tell_obj_prop_on_plcmt,
            self._parse_bring_me_obj_from_plcmt,
            self._parse_tell_cat_prop_on_plcmt,
            self._parse_find_prs_in_room,
            self._parse_meet_prs_at_beac,
            self._parse_count_prs_in_room,
            self._parse_tell_prs_info_in_loc,
            self._parse_talk_info_to_gest_prs_in_room,
            self._parse_answer_to_gest_prs_in_room,
            self._parse_follow_name_from_beac_to_room,
            self._parse_guide_name_from_beac_to_beac,
            self._parse_guide_prs_from_beac_to_beac,
            self._parse_guide_cloth_prs_from_beac_to_beac,
            self._parse_greet_cloth_dsc_in_rm,
            self._parse_greet_name_in_rm,
            self._parse_meet_name_at_loc_then_find_in_rm,
            self._parse_count_cloth_prs_in_room,
            self._parse_tell_prs_info_at_loc_to_prs_at_loc,
            self._parse_follow_prs_at_loc,
        ]

        for f in parsers:
            step, kind = f(clause)
            if step is not None:
                return step, kind

        return None, None

    # ---------- clause → followup step(s) ----------

    def _parse_followup_clause(
        self,
        clause: str,
        previous_step: GpsrStep,
    ) -> Optional[GpsrStep | List[GpsrStep]]:
        """
        2節目以降 (then/and の後ろ) を解釈する。
        CommandGenerator.generate_command_followup の各サブコマンドに対応。
        """
        c = clause.strip()

        # placeObjOnPlcmt: "put/place it on the X"
        if any(v in c for v in self.place_verbs) and " it " in (" " + c + " ") and " on the " in c:
            place = self._extract_after_phrase(c, "on the ")
            plc = best_match(place, self.placement_location_names)
            return GpsrStep(
                action="place_object_on_place",
                fields={"place": plc},
            )

        # deliverObjToMe: "bring/give/deliver it to me"
        if any(v in c for v in self.deliver_verbs) and "it to me" in c:
            return GpsrStep(
                action="deliver_object_to_operator",
                fields={},
            )

        # deliverObjToPrsInRoom: "... to the waving person in the kitchen"
        if any(v in c for v in self.deliver_verbs) and "it to the" in c and "in the" in c:
            # 非常にラフな抽出
            # "it to the <person_filter> in the <room>"
            m = re.search(r"it to the (.+) in the (.+)$", c)
            person_filter = None
            room = None
            if m:
                person_filter = m.group(1).strip()
                room = best_match(m.group(2), self.room_names)
            return GpsrStep(
                action="deliver_object_to_person_in_room",
                fields={"person_filter": person_filter, "room": room},
            )

        # deliverObjToNameAtBeac: "bring it to <name> in the kitchen"
        if any(v in c for v in self.deliver_verbs) and "it to" in c and "in the" in c:
            m = re.search(r"it to (.+?) in the (.+)$", c)
            if m:
                name_str = m.group(1).strip()
                room_str = m.group(2).strip()
                name = best_match(name_str, self.person_names)
                room = best_match(room_str, self.room_names)
                if name or room:
                    return GpsrStep(
                        action="deliver_object_to_named_person",
                        fields={"name": name, "room": room},
                    )

        # talkInfo: "tell something about yourself", "tell the time", ...
        if any(v in c for v in self.tell_verbs) and (
            "something about yourself" in c
            or "the time" in c
            or "your teams" in c
        ):
            return GpsrStep(
                action="talk_information",
                fields={"topic": c},
            )

        # answerQuestion: "answer a question/quiz"
        if any(v in c for v in self.answer_verbs) and "question" in c:
            return GpsrStep(
                action="answer_question",
                fields={},
            )

        # followPrs: "follow them"
        if "follow them" in c and "to the" not in c:
            return GpsrStep(
                action="follow_person",
                fields={},
            )

        # followPrsToRoom: "follow them to the bedroom/sofa"
        if "follow them" in c and "to the" in c:
            dest = self._extract_after_phrase(c, "to the ")
            dest_room = best_match(dest, self.room_names)
            dest_loc = best_match(dest, self.location_names)
            fields: Dict[str, Any] = {}
            if dest_room:
                fields["room"] = dest_room
            if dest_loc:
                fields["location"] = dest_loc
            return GpsrStep(
                action="follow_person_to_dest",
                fields=fields,
            )

        # guidePrsToBeacon: "guide them to the <loc or room>"
        if any(v in c for v in self.guide_verbs) and "them" in c and "to the" in c:
            dest = self._extract_after_phrase(c, "to the ")
            dest_room = best_match(dest, self.room_names)
            dest_loc = best_match(dest, self.location_names)
            fields: Dict[str, Any] = {}
            if dest_room:
                fields["room"] = dest_room
            if dest_loc:
                fields["location"] = dest_loc
            return GpsrStep(
                action="guide_person_to_dest",
                fields=fields,
            )

        # takeObj: "take it" (→ hasObj: place/deliver につながる)
        if any(v in c for v in self.take_verbs) and "it" in c:
            return GpsrStep(
                action="take_object",
                fields={},
            )

        # findPrs: "find the waving person" (フォローアップで再利用されることあり)
        if any(v in c for v in self.find_verbs) and "person" in c:
            return self._parse_find_prs_in_room(c)[0]

        # meetName: "meet <name>"
        if any(v in c for v in self.meet_verbs):
            return self._parse_meet_prs_at_beac(c)[0]

        # うまく分類できない followup は None
        return None

    # ---------- 各コマンド種別ごとのパーサ ----------

    # ---- goToLoc ----
    def _parse_go_to_loc(self, clause: str) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.go_verbs):
            return None, None
        if "to the" not in clause:
            return None, None

        dest = self._extract_after_phrase(clause, "to the ")
        dest_room = best_match(dest, self.room_names)
        dest_loc = best_match(dest, self.location_names)

        if not dest_room and not dest_loc:
            return None, None

        fields: Dict[str, Any] = {}
        if dest_room:
            fields["room"] = dest_room
        if dest_loc:
            fields["location"] = dest_loc

        step = GpsrStep(
            action="go_to_location",
            fields=fields,
        )
        return step, "goToLoc"

    # ---- takeObjFromPlcmt ----
    def _parse_take_obj_from_plcmt(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.take_verbs):
            return None, None
        if "from the" not in clause:
            return None, None

        # "take a sponge from the shelf"
        m = re.search(r"(?:take|get|grasp|fetch|pick up)\s+(?:a|an|the)?\s*(.+?)\s+from the\s+(.+)$", clause)
        if m:
            obj_str = m.group(1).strip()
            plc_str = m.group(2).strip()
        else:
            # fallback
            parts = clause.split("from the")
            obj_str = parts[0]
            plc_str = parts[1] if len(parts) > 1 else ""

        obj = best_match(obj_str, self.object_names + self.cat_sing)
        plc = best_match(plc_str, self.placement_location_names)

        step = GpsrStep(
            action="take_object_from_place",
            fields={"object_or_category": obj, "place": plc},
        )
        return step, "takeObjFromPlcmt"

    # ---- findObjInRoom ----
    def _parse_find_obj_in_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.find_verbs):
            return None, None
        if "in the" not in clause:
            return None, None

        # "find a fruit in the bathroom"
        m = re.search(r"(?:find|locate|look for|search for)\s+(?:a|an|the)?\s*(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None

        obj_str = m.group(1).strip()
        room_str = m.group(2).strip()
        obj = best_match(obj_str, self.object_names + self.cat_sing)
        room = best_match(room_str, self.room_names)
        if not room:
            return None, None

        fields = {"room": room}
        if obj:
            fields["object_or_category"] = obj

        step = GpsrStep(
            action="find_object_in_room",
            fields=fields,
        )
        return step, "findObjInRoom"

    # ---- countObjOnPlcmt ----
    def _parse_count_obj_on_plcmt(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(clause.startswith(p) for p in self.count_prefixes):
            return None, None
        if "there are on the" not in clause:
            return None, None

        # "tell me how many cleaning supplies there are on the dishwasher"
        m = re.search(r"how many\s+(.+?)\s+there are on the\s+(.+)$", clause)
        if not m:
            return None, None

        cat_str = m.group(1).strip()
        plc_str = m.group(2).strip()
        cat = best_match(cat_str, self.cat_plur)
        plc = best_match(plc_str, self.placement_location_names)

        step = GpsrStep(
            action="count_objects_on_place",
            fields={"object_category_plural": cat, "place": plc},
        )
        return step, "countObjOnPlcmt"

    # ---- tellObjPropOnPlcmt ----
    def _parse_tell_obj_prop_on_plcmt(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not clause.startswith("tell me what is the"):
            return None, None
        if "object on the" not in clause:
            return None, None

        # "tell me what is the thinnest object on the bookshelf"
        m = re.search(
            r"tell me what is the\s+(.+?)\s+object on the\s+(.+)$", clause
        )
        if not m:
            return None, None

        comp = m.group(1).strip()
        plc_str = m.group(2).strip()
        plc = best_match(plc_str, self.placement_location_names)

        step = GpsrStep(
            action="tell_object_property_on_place",
            fields={"comparison": comp, "place": plc},
        )
        return step, "tellObjPropOnPlcmt"

    # ---- bringMeObjFromPlcmt ----
    def _parse_bring_me_obj_from_plcmt(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.deliver_verbs):
            return None, None
        if "me" not in clause or "from the" not in clause:
            return None, None

        # "bring me a sponge from the kitchen table"
        m = re.search(
            r"(?:bring|give|deliver|hand)\s+me\s+(?:a|an|the)?\s*(.+?)\s+from the\s+(.+)$",
            clause,
        )
        if not m:
            return None, None

        obj_str = m.group(1).strip()
        plc_str = m.group(2).strip()
        obj = best_match(obj_str, self.object_names)
        plc = best_match(plc_str, self.placement_location_names)

        step = GpsrStep(
            action="bring_object_to_operator",
            fields={"object": obj, "source_place": plc},
        )
        return step, "bringMeObjFromPlcmt"

    # ---- tellCatPropOnPlcmt ----
    def _parse_tell_cat_prop_on_plcmt(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "tell me what is the thinnest drink on the shelf"
        if not clause.startswith("tell me what is the"):
            return None, None
        if " on the " not in clause:
            return None, None

        m = re.search(
            r"tell me what is the\s+(.+?)\s+(.+?)\s+on the\s+(.+)$", clause
        )
        if not m:
            return None, None

        comp = m.group(1).strip()
        cat_str = m.group(2).strip()
        plc_str = m.group(3).strip()

        cat = best_match(cat_str, self.cat_sing)
        plc = best_match(plc_str, self.placement_location_names)

        step = GpsrStep(
            action="tell_category_property_on_place",
            fields={"comparison": comp, "object_category": cat, "place": plc},
        )
        return step, "tellCatPropOnPlcmt"

    # ---- findPrsInRoom ----
    def _parse_find_prs_in_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.find_verbs):
            return None, None
        if "person" not in clause:
            return None, None
        if "in the" not in clause:
            return None, None

        # "find a person raising their right arm in the office"
        m = re.search(r"find\s+a\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            # 他の find 同義語にも対応
            m = re.search(r"(?:find|locate)\s+a\s+(.+?)\s+in the\s+(.+)$", clause)
        if not m:
            return None, None

        pfilter = m.group(1).strip()
        room_str = m.group(2).strip()
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="find_person_in_room",
            fields={"room": room, "person_filter": pfilter},
        )
        return step, "findPrsInRoom"

    # ---- meetPrsAtBeac / meetPrsAtBeac に相当するもの ----
    def _parse_meet_prs_at_beac(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(v in clause for v in self.meet_verbs):
            return None, None
        # "meet charlie in the kitchen"
        if "in the" not in clause and "at the" not in clause:
            return None, None

        m = re.search(r"meet\s+(.+?)\s+(?:in|at) the\s+(.+)$", clause)
        if not m:
            return None, None

        name_str = m.group(1).strip()
        room_str = m.group(2).strip()
        name = best_match(name_str, self.person_names)
        loc_room = best_match(room_str, self.room_names + self.location_names)

        step = GpsrStep(
            action="meet_person_at_place",
            fields={"name": name, "place": loc_room},
        )
        return step, "meetPrsAtBeac"

    # ---- countPrsInRoom ----
    def _parse_count_prs_in_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        if not any(clause.startswith(p) for p in self.count_prefixes):
            return None, None
        if "persons" not in clause or "in the" not in clause:
            return None, None

        m = re.search(r"how many\s+(.+?)\s+are in the\s+(.+)$", clause)
        if not m:
            return None, None

        pfilter_pl = m.group(1).strip()
        room_str = m.group(2).strip()
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="count_persons_in_room",
            fields={"room": room, "person_filter_plural": pfilter_pl},
        )
        return step, "countPrsInRoom"

    # ---- tellPrsInfoInLoc ----
    def _parse_tell_prs_info_in_loc(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "tell me the name of the person in the kitchen"
        if not clause.startswith("tell me the"):
            return None, None
        if "of the person" not in clause:
            return None, None

        m = re.search(r"tell me the\s+(.+?)\s+of the person\s+(.+)$", clause)
        if not m:
            return None, None

        info = m.group(1).strip()
        tail = m.group(2).strip()  # "in the kitchen" or "at the chairs"
        room = None
        loc = None
        if "in the" in tail:
            room = best_match(self._extract_after_phrase(tail, "in the "), self.room_names)
        if "at the" in tail:
            loc = best_match(self._extract_after_phrase(tail, "at the "), self.location_names)

        step = GpsrStep(
            action="tell_person_info",
            fields={"person_info": info, "room": room, "location": loc},
        )
        return step, "tellPrsInfoInLoc"

    # ---- talkInfoToGestPrsInRoom ----
    def _parse_talk_info_to_gest_prs_in_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "tell something about yourself to the waving person in the kitchen"
        if not any(v in clause for v in self.tell_verbs):
            return None, None
        if "to the" not in clause or "in the" not in clause:
            return None, None

        m = re.search(
            r"(tell .+?)\s+to the\s+(.+?)\s+in the\s+(.+)$", clause
        )
        if not m:
            return None, None

        talk_content = m.group(1).strip()
        pfilter = m.group(2).strip()
        room_str = m.group(3).strip()
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="talk_to_person_in_room",
            fields={"talk": talk_content, "person_filter": pfilter, "room": room},
        )
        return step, "talkInfoToGestPrsInRoom"

    # ---- answerToGestPrsInRoom ----
    def _parse_answer_to_gest_prs_in_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "answer the question of the waving person in the kitchen"
        if not any(v in clause for v in self.answer_verbs):
            return None, None
        if "of the" not in clause or "in the" not in clause:
            return None, None

        m = re.search(
            r"answer\s+the\s+(.+?)\s+of the\s+(.+?)\s+in the\s+(.+)$", clause
        )
        if not m:
            return None, None

        question = m.group(1).strip()
        pfilter = m.group(2).strip()
        room_str = m.group(3).strip()
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="answer_to_person_in_room",
            fields={"question": question, "person_filter": pfilter, "room": room},
        )
        return step, "answerToGestPrsInRoom"

    # ---- followNameFromBeacToRoom ----
    def _parse_follow_name_from_beac_to_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "follow charlie from the sofa to the bedroom"
        if not any(v in clause for v in self.follow_verbs):
            return None, None
        if "from the" not in clause or "to the" not in clause:
            return None, None

        m = re.search(
            r"follow\s+(.+?)\s+from the\s+(.+?)\s+to the\s+(.+)$", clause
        )
        if not m:
            return None, None

        name_str = m.group(1).strip()
        from_str = m.group(2).strip()
        to_str = m.group(3).strip()
        name = best_match(name_str, self.person_names)
        from_loc = best_match(from_str, self.location_names)
        to_room = best_match(to_str, self.room_names)

        step = GpsrStep(
            action="follow_named_person_from_loc_to_room",
            fields={"name": name, "from_location": from_loc, "to_room": to_room},
        )
        return step, "followNameFromBeacToRoom"

    # ---- guideNameFromBeacToBeac ----
    def _parse_guide_name_from_beac_to_beac(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "guide charlie from the sofa to the desk"
        if not any(v in clause for v in self.guide_verbs):
            return None, None
        if "from the" not in clause or "to the" not in clause:
            return None, None

        m = re.search(
            r"(?:guide|escort|take|lead)\s+(.+?)\s+from the\s+(.+?)\s+to the\s+(.+)$",
            clause,
        )
        if not m:
            return None, None

        name_str = m.group(1).strip()
        from_str = m.group(2).strip()
        to_str = m.group(3).strip()
        name = best_match(name_str, self.person_names)
        from_loc = best_match(from_str, self.location_names + self.room_names)
        to_loc_room = best_match(to_str, self.location_names + self.room_names)

        step = GpsrStep(
            action="guide_named_person_from_place_to_place",
            fields={"name": name, "from_place": from_loc, "to_place": to_loc_room},
        )
        return step, "guideNameFromBeacToBeac"

    # ---- guidePrsFromBeacToBeac ----
    def _parse_guide_prs_from_beac_to_beac(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "guide the waving person from the chairs to the sofa"
        if not any(v in clause for v in self.guide_verbs):
            return None, None
        if "the" not in clause or "from the" not in clause or "to the" not in clause:
            return None, None
        if "person" not in clause:
            return None, None

        m = re.search(
            r"(?:guide|escort|take|lead)\s+the\s+(.+?)\s+from the\s+(.+?)\s+to the\s+(.+)$",
            clause,
        )
        if not m:
            return None, None

        pfilter = m.group(1).strip()
        from_str = m.group(2).strip()
        to_str = m.group(3).strip()
        from_loc = best_match(from_str, self.location_names + self.room_names)
        to_loc_room = best_match(to_str, self.location_names + self.room_names)

        step = GpsrStep(
            action="guide_person_from_place_to_place",
            fields={"person_filter": pfilter, "from_place": from_loc, "to_place": to_loc_room},
        )
        return step, "guidePrsFromBeacToBeac"

    # ---- guideClothPrsFromBeacToBeac ----
    def _parse_guide_cloth_prs_from_beac_to_beac(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "guide the person wearing a red t shirt from the sofa to the shelf"
        if "person wearing" not in clause:
            return None, None
        st, _ = self._parse_guide_prs_from_beac_to_beac(clause)
        if st is None:
            return None, None
        # clothes_description は clause から素直に抜いておく
        m = re.search(r"person wearing (.+?) from the", clause)
        clothes = m.group(1).strip() if m else None
        st.fields["clothes_description"] = clothes
        return st, "guideClothPrsFromBeacToBeac"

    # ---- greetClothDscInRm ----
    def _parse_greet_cloth_dsc_in_rm(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "greet the person wearing a red t shirt in the kitchen"
        if not any(v in clause for v in self.greet_verbs):
            return None, None
        if "person wearing" not in clause or "in the" not in clause:
            return None, None

        m = re.search(
            r"(?:greet|salute|say hello to|introduce yourself to)\s+the\s+person wearing\s+(?:a|an|the)?\s*(.+?)\s+in the\s+(.+)$",
            clause,
        )
        if not m:
            return None, None

        clothes = m.group(1).strip()
        room_str = m.group(2).strip()
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="greet_person_with_clothes_in_room",
            fields={"room": room, "clothes_description": clothes},
        )
        return step, "greetClothDscInRm"

    # ---- greetNameInRm ----
    def _parse_greet_name_in_rm(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "greet charlie in the office"
        if not any(v in clause for v in self.greet_verbs):
            return None, None
        if "in the" not in clause:
            return None, None

        m = re.search(
            r"(?:greet|salute|say hello to|introduce yourself to)\s+(.+?)\s+in the\s+(.+)$",
            clause,
        )
        if not m:
            return None, None

        name_str = m.group(1).strip()
        room_str = m.group(2).strip()
        name = best_match(name_str, self.person_names)
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="greet_named_person_in_room",
            fields={"name": name, "room": room},
        )
        return step, "greetNameInRm"

    # ---- meetNameAtLocThenFindInRm ----
    def _parse_meet_name_at_loc_then_find_in_rm(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # CommandGenerator では "meet ... at the <loc>" までがこの節、
        # "then find them in the <room>" は次節の扱いにしてよい。
        if not any(v in clause for v in self.meet_verbs):
            return None, None
        if "at the" not in clause:
            return None, None

        m = re.search(r"meet\s+(.+?)\s+at the\s+(.+)$", clause)
        if not m:
            return None, None

        name_str = m.group(1).strip()
        loc_str = m.group(2).strip()
        name = best_match(name_str, self.person_names)
        loc = best_match(loc_str, self.location_names)

        step = GpsrStep(
            action="meet_named_person_at_location",
            fields={"name": name, "location": loc},
        )
        return step, "meetNameAtLocThenFindInRm"

    # ---- countClothPrsInRoom ----
    def _parse_count_cloth_prs_in_room(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "tell me how many people in the kitchen are wearing red t shirts"
        if not any(clause.startswith(p) for p in self.count_prefixes):
            return None, None
        if "are wearing" not in clause or "in the" not in clause:
            return None, None

        m = re.search(
            r"how many\s+people\s+in the\s+(.+?)\s+are wearing\s+(.+)$", clause
        )
        if not m:
            return None, None

        room_str = m.group(1).strip()
        clothes = m.group(2).strip()
        room = best_match(room_str, self.room_names)

        step = GpsrStep(
            action="count_people_with_clothes_in_room",
            fields={"room": room, "clothes_description": clothes},
        )
        return step, "countClothPrsInRoom"

    # ---- tellPrsInfoAtLocToPrsAtLoc ----
    def _parse_tell_prs_info_at_loc_to_prs_at_loc(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "tell the name of the person at the sofa to the person at the desk"
        if not clause.startswith("tell"):
            return None, None
        if "of the person at the" not in clause or "to the person at the" not in clause:
            return None, None

        m = re.search(
            r"tell the\s+(.+?)\s+of the person at the\s+(.+?)\s+to the person at the\s+(.+)$",
            clause,
        )
        if not m:
            return None, None

        info = m.group(1).strip()
        loc1_str = m.group(2).strip()
        loc2_str = m.group(3).strip()
        loc1 = best_match(loc1_str, self.location_names)
        loc2 = best_match(loc2_str, self.location_names)

        step = GpsrStep(
            action="tell_person_info_from_loc_to_loc",
            fields={"person_info": info, "source_location": loc1, "target_location": loc2},
        )
        return step, "tellPrsInfoAtLocToPrsAtLoc"

    # ---- followPrsAtLoc ----
    def _parse_follow_prs_at_loc(
        self, clause: str
    ) -> tuple[Optional[GpsrStep], Optional[str]]:
        # "follow the standing person at the chairs"
        if not any(v in clause for v in self.follow_verbs):
            return None, None
        if "the" not in clause or "at the" not in clause:
            return None, None
        if "person" not in clause:
            return None, None

        m = re.search(
            r"follow the\s+(.+?)\s+at the\s+(.+)$", clause
        )
        if not m:
            return None, None

        pfilter = m.group(1).strip()
        loc_str = m.group(2).strip()
        loc = best_match(loc_str, self.location_names + self.room_names)

        step = GpsrStep(
            action="follow_person_at_location",
            fields={"person_filter": pfilter, "location": loc},
        )
        return step, "followPrsAtLoc"

    # ---------- 小さなヘルパー ----------

    def _extract_after_phrase(self, text: str, phrase: str) -> str:
        """`text` 内の `phrase` 以降の文字列を返す（なければ空）。"""
        idx = text.find(phrase)
        if idx < 0:
            return ""
        return text[idx + len(phrase):].strip()


# ================= gpsr_intent_v1 (fixed schema) =================


@dataclass
class GpsrIntentV1:
    """
    固定スキーマの intent 表現（実行は steps を唯一の真実として扱う想定）
    """
    schema: str = "gpsr_intent_v1"
    ok: bool = True
    need_confirm: bool = False
    intent_type: str = "unknown"     # bring / find / go / count / tell / follow / guide / greet / meet / composite / unknown
    slots: Dict[str, Any] = None
    raw_text: str = ""
    normalized_text: str = ""
    confidence: Any = None           # {"asr":..., "parser":...} など。現状は None
    source: str = "parser"
    command_kind: str = ""           # 旧 "kind" をそのまま入れる
    steps: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "ok": self.ok,
            "need_confirm": self.need_confirm,
            "intent_type": self.intent_type,
            "slots": self.slots or {},
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "confidence": self.confidence,
            "source": self.source,
            "command_kind": self.command_kind,
            "steps": self.steps or [],
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _intent_type_from_kind_and_steps(kind: str, steps: List[GpsrStep]) -> str:
    """
    kind/steps から上位 intent_type を決める（ラベル）。
    実行側は command_kind と steps を主に見ることを推奨。
    """
    if not steps:
        return "unknown"
    if len(steps) >= 2:
        return "composite"

    k = (kind or "").lower()

    if "goto" in k:
        return "go"
    if "bring" in k or "deliver" in k:
        return "bring"
    if "takeobj" in k or "take" in k:
        return "take"
    if "findobj" in k or "findprs" in k or "find" in k:
        return "find"
    if "count" in k:
        return "count"
    if "tell" in k or "talk" in k or "answer" in k:
        return "tell"
    if "follow" in k:
        return "follow"
    if "guide" in k:
        return "guide"
    if "greet" in k:
        return "greet"
    if "meet" in k:
        return "meet"
    return "unknown"


def _empty_fixed_slots() -> Dict[str, Any]:
    """
    gpsr_intent_v1 固定スロット（未使用は None）
    """
    return {
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

        "attribute": None,
        "question_type": None,
        "gesture": None,
    }


def _merge_slot(slots: Dict[str, Any], key: str, value: Any):
    """None/空文字の上書きを避けつつ埋める。"""
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        return
    # すでに埋まっているなら維持（後勝ちにしたいならここを変える）
    if slots.get(key) is None:
        slots[key] = value


def _slots_from_steps(steps: List[GpsrStep]) -> Dict[str, Any]:
    """
    既存の steps(fields) から固定 slots を可能な範囲で埋める。
    （実行は steps を見るので、slots は要約・デバッグ用途）
    """
    slots = _empty_fixed_slots()

    for st in steps:
        a = st.action
        f = st.fields or {}

        # --- objects / categories ---
        if "object" in f:
            _merge_slot(slots, "object", f.get("object"))
        if "object_or_category" in f:
            # ここは曖昧だが、find/take 系で多いので category として入れる（必要なら実運用で調整）
            _merge_slot(slots, "object_category", f.get("object_or_category"))
        if "object_category" in f:
            _merge_slot(slots, "object_category", f.get("object_category"))
        if "object_category_plural" in f:
            _merge_slot(slots, "object_category", f.get("object_category_plural"))

        # --- source / destination inference by action ---
        if a in ("find_object_in_room", "find_person_in_room", "count_persons_in_room", "count_people_with_clothes_in_room"):
            _merge_slot(slots, "source_room", f.get("room"))
        if a in ("take_object_from_place", "count_objects_on_place", "tell_object_property_on_place", "tell_category_property_on_place"):
            _merge_slot(slots, "source_place", f.get("place"))
        if a in ("bring_object_to_operator",):
            _merge_slot(slots, "source_place", f.get("source_place"))
        if a in ("place_object_on_place", "place_object_on_place", "place_object_on_place", "place_object_on_place"):
            _merge_slot(slots, "destination_place", f.get("place"))
        if a in ("place_object_on_place", "place_object_on_place"):
            pass
        if a in ("deliver_object_to_named_person",):
            _merge_slot(slots, "person", f.get("name"))
            _merge_slot(slots, "destination_room", f.get("room"))
        if a in ("deliver_object_to_person_in_room",):
            _merge_slot(slots, "destination_room", f.get("room"))
            _merge_slot(slots, "attribute", f.get("person_filter"))

        if a in ("follow_named_person_from_loc_to_room",):
            _merge_slot(slots, "person", f.get("name"))
            _merge_slot(slots, "source_place", f.get("from_location"))
            _merge_slot(slots, "destination_room", f.get("to_room"))

        if a in ("guide_named_person_from_place_to_place",):
            _merge_slot(slots, "person", f.get("name"))
            _merge_slot(slots, "source_place", f.get("from_place"))
            # from_place/to_place は room/place 両方ありうるので destination_place に寄せる
            _merge_slot(slots, "destination_place", f.get("to_place"))

        if a in ("guide_person_from_place_to_place",):
            _merge_slot(slots, "attribute", f.get("person_filter"))
            _merge_slot(slots, "source_place", f.get("from_place"))
            _merge_slot(slots, "destination_place", f.get("to_place"))

        if a in ("greet_named_person_in_room",):
            _merge_slot(slots, "person", f.get("name"))
            _merge_slot(slots, "source_room", f.get("room"))

        if a in ("greet_person_with_clothes_in_room",):
            _merge_slot(slots, "attribute", f.get("clothes_description"))
            _merge_slot(slots, "source_room", f.get("room"))

        if a in ("meet_person_at_place", "meet_named_person_at_location"):
            _merge_slot(slots, "person", f.get("name"))
            # meetは place/location なので source_place に寄せる
            _merge_slot(slots, "source_place", f.get("place") or f.get("location"))

        # --- question / tell ---
        if a in ("talk_to_person_in_room",):
            _merge_slot(slots, "attribute", f.get("person_filter"))
            _merge_slot(slots, "source_room", f.get("room"))
        if a in ("answer_to_person_in_room",):
            _merge_slot(slots, "attribute", f.get("person_filter"))
            _merge_slot(slots, "source_room", f.get("room"))
            _merge_slot(slots, "question_type", f.get("question"))

    return slots


def _steps_to_v1_steps(steps: List[GpsrStep]) -> List[Dict[str, Any]]:
    """
    v1では steps の各要素を {action, args} 形式で出す。
    既存互換のため fields も残したい場合は、ここで併記できるが、
    原則 v1 は args のみ推奨。
    """
    out: List[Dict[str, Any]] = []
    for st in steps:
        out.append(
            {
                "action": st.action,
                "args": st.fields or {},
            }
        )
    return out


# ---- GpsrCommand に v1 出力を追加（後方互換で to_json はそのまま） ----
def gpsrcommand_to_intent_v1(self: GpsrCommand) -> GpsrIntentV1:
    norm = normalize_text(self.raw_text) if self.raw_text else ""
    intent_type = _intent_type_from_kind_and_steps(self.kind, self.steps)
    slots = _slots_from_steps(self.steps)
    v1_steps = _steps_to_v1_steps(self.steps)

    return GpsrIntentV1(
        ok=True,
        need_confirm=False,
        intent_type=intent_type,
        slots=slots,
        raw_text=self.raw_text,
        normalized_text=norm,
        confidence=None,
        source="parser",
        command_kind=self.kind,
        steps=v1_steps,
    )


def gpsrcommand_to_intent_v1_json(self: GpsrCommand) -> str:
    return gpsrcommand_to_intent_v1(self).to_json_str()


# 動的にメソッド注入（既存コードへの影響を最小化）
GpsrCommand.to_intent_v1 = gpsrcommand_to_intent_v1          # type: ignore
GpsrCommand.to_intent_v1_json = gpsrcommand_to_intent_v1_json  # type: ignore
