#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jsonl -> corrections.yaml generator for GPSR ASR

Expected JSONL records (one per utterance), ideally containing BOTH:
  - raw_text:  ASR output before corrections/hard-normalization
  - text:      final text after your correction pipeline (what parser sees)

If raw_text is missing, this tool can still work if you provide --gold-field (e.g., "gold")
that contains a manually-corrected target string.

Heuristic:
  - Extract replacement spans from raw -> target using difflib.SequenceMatcher
  - Propose phrase-level corrections: {"<raw span>": "<target span>"}
  - Aggregate counts, filter noisy/too-short/too-long candidates
  - Emit YAML:
      corrections:
        "past room": "bathroom"
        "foot": "food"
      meta:
        generated_at: ...
        source: ...
        params: ...

Usage examples:
  python3 tools/jsonl_to_corrections.py logs/asr_log_20251221_*.jsonl -o corrections.yaml

  # If you stored manual gold in field "gold":
  python3 tools/jsonl_to_corrections.py logs/*.jsonl --gold-field gold -o corrections.yaml

Tips:
  - Keep your logger writing raw_text + text (or gold) for best results.
  - Review output before merging into vocab.yaml; heuristics can produce false positives.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple, Optional


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s


def _lower(s: str) -> str:
    return _norm(s).lower()


_WORD = re.compile(r"[A-Za-z0-9']+")


def _tokenize(s: str) -> List[str]:
    # keep simple tokenization; we operate on text-level spans as well
    return _WORD.findall(_lower(s))


def _safe_yaml_quote(s: str) -> str:
    # We'll let PyYAML handle quoting normally; this is unused unless you want custom formatting.
    return s


@dataclass
class Candidate:
    src: str
    dst: str
    count: int


def _iter_jsonl(paths: List[str]) -> Iterable[dict]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(f"[WARN] skip bad json: {p}:{ln}: {e}", file=sys.stderr)
                    continue


def _extract_span_corrections(raw: str, tgt: str) -> List[Tuple[str, str]]:
    """
    Return list of (raw_span, tgt_span) phrase replacements inferred from diff.
    Works on lowercase, whitespace-normalized strings.
    """
    raw_n = _lower(raw)
    tgt_n = _lower(tgt)
    if not raw_n or not tgt_n or raw_n == tgt_n:
        return []

    sm = SequenceMatcher(a=raw_n, b=tgt_n)
    out: List[Tuple[str, str]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        src = raw_n[i1:i2].strip()
        dst = tgt_n[j1:j2].strip()

        # Only consider true replacements. Insert/delete are often context-dependent.
        if tag != "replace":
            continue

        # Basic cleanup: collapse spaces
        src = " ".join(src.split())
        dst = " ".join(dst.split())

        if not src or not dst:
            continue

        out.append((src, dst))

    return out


def _filter_candidate(src: str, dst: str,
                      min_len: int, max_len: int,
                      allow_multiword: bool,
                      stopwords: set) -> bool:
    """
    Return True if candidate should be kept.
    """
    if src == dst:
        return False
    if len(src) < min_len or len(dst) < min_len:
        return False
    if len(src) > max_len or len(dst) > max_len:
        return False

    # avoid pure punctuation
    if not re.search(r"[a-z0-9]", src) or not re.search(r"[a-z0-9]", dst):
        return False

    # avoid overly broad replacements
    if src in stopwords:
        return False

    # optionally restrict to single-token replacements
    if not allow_multiword:
        if len(_tokenize(src)) != 1 or len(_tokenize(dst)) != 1:
            return False

    return True


def _dump_yaml(mapping: Dict[str, str], meta: dict, out_path: str):
    try:
        import yaml  # PyYAML
    except Exception:
        print("[ERROR] PyYAML is required (pip install pyyaml).", file=sys.stderr)
        sys.exit(2)

    doc = {
        "corrections": mapping,
        "meta": meta,
    }

    # Stable output
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            doc,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=120,
        )


def main():
    ap = argparse.ArgumentParser(description="Generate corrections.yaml from ASR jsonl logs")
    ap.add_argument("inputs", nargs="+", help="JSONL file(s) or glob(s)")
    ap.add_argument("-o", "--out", default="corrections.yaml", help="Output YAML path")
    ap.add_argument("--raw-field", default="raw_text", help="Field name for raw ASR text (default: raw_text)")
    ap.add_argument("--tgt-field", default="text", help="Field name for corrected/target text (default: text)")
    ap.add_argument("--gold-field", default="", help="Optional: field name for manual gold target (overrides --tgt-field if present)")

    ap.add_argument("--min-count", type=int, default=2, help="Minimum occurrences to include")
    ap.add_argument("--min-len", type=int, default=3, help="Minimum span length")
    ap.add_argument("--max-len", type=int, default=40, help="Maximum span length")
    ap.add_argument("--allow-multiword", action="store_true", help="Allow multiword phrase replacements (default: off)")
    ap.add_argument("--max-items", type=int, default=200, help="Max items in output (by frequency)")

    ap.add_argument("--stopwords", default="a,an,the,of,to,from,in,on,at,me,you,please,then,and",
                    help="Comma-separated stopwords to exclude as src replacements")

    args = ap.parse_args()

    # Expand globs
    files: List[str] = []
    for x in args.inputs:
        g = glob.glob(x)
        if g:
            files.extend(g)
        else:
            files.append(x)

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("[ERROR] No input files found.", file=sys.stderr)
        sys.exit(1)

    stopwords = set([_lower(w) for w in args.stopwords.split(",") if _lower(w)])

    pair_counts: Counter[Tuple[str, str]] = Counter()
    raw_counts: Counter[str] = Counter()
    used = 0
    skipped = 0

    for rec in _iter_jsonl(files):
        raw = _norm(str(rec.get(args.raw_field, "") or ""))
        if not raw:
            skipped += 1
            continue

        # Choose target: gold_field if provided and present; else tgt_field
        tgt = ""
        if args.gold_field:
            tgt = _norm(str(rec.get(args.gold_field, "") or ""))
        if not tgt:
            tgt = _norm(str(rec.get(args.tgt_field, "") or ""))

        if not tgt:
            skipped += 1
            continue

        raw_counts[_lower(raw)] += 1

        spans = _extract_span_corrections(raw, tgt)
        if not spans:
            continue

        for src, dst in spans:
            if _filter_candidate(src, dst,
                                 min_len=args.min_len,
                                 max_len=args.max_len,
                                 allow_multiword=args.allow_multiword,
                                 stopwords=stopwords):
                pair_counts[(src, dst)] += 1

        used += 1

    # Aggregate by src: choose most frequent dst for each src
    by_src: Dict[str, Counter[str]] = defaultdict(Counter)
    for (src, dst), c in pair_counts.items():
        by_src[src][dst] += c

    chosen: List[Candidate] = []
    for src, dst_counter in by_src.items():
        dst, c = dst_counter.most_common(1)[0]
        chosen.append(Candidate(src=src, dst=dst, count=c))

    # Filter by min-count and sort
    chosen = [x for x in chosen if x.count >= args.min_count]
    chosen.sort(key=lambda x: (-x.count, -len(x.src), x.src))

    if args.max_items > 0:
        chosen = chosen[: args.max_items]

    mapping = {c.src: c.dst for c in chosen}

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inputs": files,
        "raw_field": args.raw_field,
        "target_field": (args.gold_field or args.tgt_field),
        "min_count": args.min_count,
        "min_len": args.min_len,
        "max_len": args.max_len,
        "allow_multiword": bool(args.allow_multiword),
        "max_items": args.max_items,
        "records_used_for_diff": used,
        "records_skipped": skipped,
        "unique_raw_texts": len(raw_counts),
        "unique_pairs": len(pair_counts),
        "notes": "Review before merging into vocab.yaml. Prefer allow_multiword for phrase fixes like 'past room'->'bathroom'.",
    }

    _dump_yaml(mapping, meta, args.out)

    print(f"[OK] wrote {args.out} ({len(mapping)} corrections)")
    if mapping:
        # preview top 10
        print("Top candidates:")
        for c in chosen[:10]:
            print(f"  ({c.count:>3}x) '{c.src}' -> '{c.dst}'")


if __name__ == "__main__":
    main()
