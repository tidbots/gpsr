# gpsr_projector.py
# GPSR template projection engine

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s)
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
    def __init__(self, templates: List[str], vocab: Dict[str, List[str]]):
        self.templates = templates
        self.vocab = vocab

        self.names = sorted(vocab.get("names", []), key=len, reverse=True)
        self.locations = sorted(vocab.get("locations", []), key=len, reverse=True)
        self.objects = sorted(vocab.get("objects", []), key=len, reverse=True)
        self.categories = sorted(vocab.get("object_categories", []), key=len, reverse=True)

    # --------------------
    def extract_slots(self, text: str) -> Dict[str, str]:
        t = _norm(text)
        slots: Dict[str, str] = {}

        def find(cands):
            for c in cands:
                if _norm(c) in t:
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

    # --------------------
    def fill(self, template: str, slots: Dict[str, str]) -> Optional[str]:
        needed = re.findall(r"\{([a-z_]+)\}", template)
        for k in needed:
            if k not in slots:
                return None

        out = template
        for k, v in slots.items():
            out = out.replace("{" + k + "}", v)
        return out

    # --------------------
    def score(self, hyp: str, cand: str) -> float:
        j = _jaccard(hyp, cand)
        d = _levenshtein(hyp, cand)
        L = max(len(_norm(hyp)), len(_norm(cand)), 1)
        ed = 1.0 - d / L
        return 0.65 * j + 0.35 * ed

    # --------------------
    def project(self, hyp: str) -> Optional[ProjectionResult]:
        slots = self.extract_slots(hyp)
        best = None

        for tpl in self.templates:
            filled = self.fill(tpl, slots)
            if not filled:
                continue

            s = self.score(hyp, filled)
            if best is None or s > best.score:
                best = ProjectionResult(
                    projected_text=filled,
                    template=tpl,
                    score=s,
                    slots=slots,
                )
        return best
