#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_parser.py

RoboCup@Home GPSR向けの簡易パーサ（ASRテキスト → intent JSON）

変更点（重要）:
- best_match() を「候補順に依存しない最長一致優先」に修正
  例: "bedside table" が "bed" に吸われない
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import re


# ================= ユーティリティ =================

def normalize_text(s: str) -> str:
    s = s.strip()
    # 句読点や余計な空白をならす
    s = re.sub(r"\s+", " ", s)
    return s


def split_into_clauses(text: str) -> List[str]:
    """
    GPSRのテンプレは "and then" / "then" / "," などで複文になるので、
    ざっくり分割して節として扱う。
    """
    text = normalize_text(text)
    if not text:
        return []

    # "and then" を強めに区切る
    text = re.sub(r"\band then\b", " ; ", text, flags=re.IGNORECASE)
    # "then" は状況次第だが区切りとして扱う
    text = re.sub(r"\bthen\b", " ; ", text, flags=re.IGNORECASE)

    parts = re.split(r"[;]", text)
    clauses: List[str] = []
    for p in parts:
        s = p.strip(" ,.")
        if s:
            clauses.append(s)
    return clauses


def best_match(token: str, candidates: List[str]) -> Optional[str]:
    """
    候補リストの中から最も妥当なものを返す（候補順に依存しない）。

    改善点（重要）:
      - 「bed」と「bedside table」のような包含関係では *最長一致* を優先する
      - 完全一致 > 前方/後方一致 > 部分一致 の順でスコア化して最良を選ぶ
    """
    token = token.strip().lower()
    if not token:
        return None

    # 1) 完全一致（最優先）
    exact = [c for c in candidates if token == c.strip().lower()]
    if exact:
        # 同点なら長い方（例: 重複候補があっても安全）
        return max(exact, key=lambda s: len(s.strip()))

    scored: List[tuple] = []
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
            # (score, length) で最大を採用。lengthで最長一致を保証
            scored.append((score, len(cl), c))

    if not scored:
        return None

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


# ================= 内部表現 =================

@dataclass
class GpsrStep:
    """1つのサブタスク（節）に対応する意味表現。"""
    action: str
    fields: Dict[str, Any]


@dataclass
class GpsrIntent:
    schema: str = "gpsr_intent_v1"
    ok: bool = True
    need_confirm: bool = False
    intent_type: str = ""
    slots: Dict[str, Any] = None
    raw_text: str = ""
    confidence: Optional[float] = None
    source: str = "parser"
    command_kind: str = ""
    steps: List[Dict[str, Any]] = None

    def to_json_str(self) -> str:
        payload = {
            "schema": self.schema,
            "ok": self.ok,
            "need_confirm": self.need_confirm,
            "intent_type": self.intent_type,
            "slots": self.slots or {},
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "source": self.source,
            "command_kind": self.command_kind,
            "steps": self.steps or [],
        }
        return json.dumps(payload, ensure_ascii=False)


# ================= パーサ本体 =================

class GpsrParser:
    def __init__(self):
        # ---- 語彙（必要に応じて拡張） ----
        self.placement_location_names = [
            "bed",
            "bedside table",
            "shelf",
            "trashbin",
            "dishwasher",
            "potted plant",
            "kitchen table",
            "chairs",
            "pantry",
            "refrigerator",
            "sink",
            "cabinet",
        ]

        self.object_names = [
            "red wine",
            "wine",
            "water",
            "coke",
            "tea",
            "coffee",
        ]

        # bring / bring me / bring to ...
        self.re_bring_me = re.compile(
            r"^(bring|fetch|get)\s+(me\s+)?(?P<object>.+?)\s+from\s+the\s+(?P<place>.+)$",
            flags=re.IGNORECASE,
        )

    def parse(self, text: str) -> GpsrIntent:
        raw = normalize_text(text)
        clauses = split_into_clauses(raw)

        # いまは単文（最初の節）だけ処理
        target = clauses[0] if clauses else raw
        target = target.strip(" .")

        # bring me X from the Y
        m = self.re_bring_me.match(target)
        if m:
            obj_raw = m.group("object").strip(" .")
            place_raw = m.group("place").strip(" .")

            obj = best_match(obj_raw, self.object_names) or obj_raw
            place = best_match(place_raw, self.placement_location_names) or place_raw

            intent = GpsrIntent(
                intent_type="bring",
                slots={
                    "object": obj,
                    "source_place": place,
                    "destination": "",
                    "person": "",
                },
                raw_text=target if target.endswith(".") else (target + "."),
                command_kind="bringMeObjFromPlcmt",
                steps=[
                    {
                        "action": "bring_object_to_operator",
                        "fields": {"object": obj, "source_place": place},
                    }
                ],
            )
            return intent

        # fallback
        return GpsrIntent(
            ok=False,
            need_confirm=True,
            intent_type="unknown",
            slots={},
            raw_text=raw,
            command_kind="unknown",
            steps=[],
        )


# ================= 動作確認用 =================

def main():
    import sys
    parser = GpsrParser()
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = "Bring me a red wine from the bedside table"
    intent = parser.parse(text)
    print(intent.to_json_str())


if __name__ == "__main__":
    main()
