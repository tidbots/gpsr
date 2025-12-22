#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_projector.py
Project free-form ASR text onto a finite list of *official* GPSR commands.

- Recommended candidates: official command generator output (one command per line)
- Also supports templates containing {slots} (optional)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> List[str]:
    return _norm(s).split()


def _levenshtein(a: str, b: str) -> int:
    a, b = _norm(a), _norm(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _jaccard(a: str, b: str) -> float:
    A, B = set(_tokens(a)), set(_tokens(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


@dataclass
class ProjectionResult:
    projected_text: str
    template: str
    score: float
    slots: Dict[str, str]


class GpsrProjector:
    """
    If candidates are concrete commands (no {...}), it's nearest-neighbor over command list.
    If candidates contain slots, tries to fill them using vocab-based slot extraction.
    """

    def __init__(self, candidates: List[str], vocab: Optional[Dict[str, List[str]]] = None):
        self.candidates = [c.strip() for c in (candidates or []) if c and c.strip()]
        self.vocab = vocab or {}

        self.names = sorted(self.vocab.get("names", []), key=len, reverse=True)
        self.locations = sorted(self.vocab.get("locations", []), key=len, reverse=True)
        self.objects = sorted(self.vocab.get("objects", []), key=len, reverse=True)
        self.categories = sorted(self.vocab.get("object_categories", []), key=len, reverse=True)

    def extract_slots(self, text: str) -> Dict[str, str]:
        t = _norm(text)
        slots: Dict[str, str] = {}

        def find(cands):
            for c in cands:
                cn = _norm(c)
                if cn and cn in t:
                    return c
            return None

        if (v := find(self.names)):
            slots["name"] = v
        if (v := find(self.locations)):
            slots["location"] = v
        if (v := find(self.objects)):
            slots["object"] = v
        if (v := find(self.categories)):
            slots["object_category"] = v

        return slots

    def _fill_if_needed(self, cand: str, slots: Dict[str, str]) -> Optional[str]:
        if "{" not in cand:  # concrete command
            return cand

        needed = re.findall(r"\{([a-z_]+)\}", cand)
        for k in needed:
            if k not in slots:
                return None

        out = cand
        for k, v in slots.items():
            out = out.replace("{" + k + "}", v)
        return out

    def score(self, hyp: str, cand: str) -> float:
        j = _jaccard(hyp, cand)
        d = _levenshtein(hyp, cand)
        L = max(len(_norm(hyp)), len(_norm(cand)), 1)
        ed = 1.0 - d / L
        return 0.65 * j + 0.35 * ed

    def project(self, hyp: str) -> Optional[ProjectionResult]:
        if not self.candidates:
            return None

        slots = self.extract_slots(hyp)
        best: Optional[ProjectionResult] = None

        for c in self.candidates:
            filled = self._fill_if_needed(c, slots)
            if not filled:
                continue
            s = self.score(hyp, filled)
            if best is None or s > best.score:
                best = ProjectionResult(projected_text=filled, template=c, score=s, slots=slots)

        return best
