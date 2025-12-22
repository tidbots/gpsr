#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_eval.py
ASRを通さず、GPSRコマンド文（テキスト）を大量に流して gpsr_parser.py を回帰テストするCLI。

使い方:
  python3 tools/gpsr_eval.py --commands command.txt --out runs
  python3 tools/gpsr_eval.py --commands command.txt --out runs --suggest
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _json_default(o):
    # gpsr_parser.Place をJSON化
    if hasattr(o, "__class__") and o.__class__.__name__ == "Place":
        return {
            "name": getattr(o, "name", None),
            "is_room": getattr(o, "is_room", None),
            "is_placement": getattr(o, "is_placement", None),
        }
    # それ以外はとりあえず文字列化
    return str(o)
# -------------------------
# Import resolver (robust)
# -------------------------

def _add_paths_for_project():
    """
    gpsr_eval.py の位置から、gpsr_parser.py / gpsr_vocab.py が居そうな場所を探索して sys.path に追加。
    想定候補:
      - hsr_audio_pipeline/            (package root)
      - hsr_audio_pipeline/scripts/
      - hsr_audio_pipeline/tools/
    """
    here = Path(__file__).resolve()
    # tools/ -> package root
    pkg_root = here.parents[1]  # .../hsr_audio_pipeline
    candidates = [
        pkg_root,
        pkg_root / "scripts",
        pkg_root / "tools",
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

_add_paths_for_project()

# ---- Project imports (after path fix) ----
from gpsr_parser import GpsrParser, normalize_text  # type: ignore
import gpsr_vocab  # type: ignore


# -------------------------
# Utilities
# -------------------------

def read_commands(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = s.split("\t")[0].strip()
        if s:
            lines.append(s)
    return lines


def ensure_run_dir(out_root: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_parser_from_vocab() -> GpsrParser:
    return GpsrParser(
        person_names=list(getattr(gpsr_vocab, "NAMES", [])),
        location_names=list(getattr(gpsr_vocab, "LOCATIONS", [])),
        placement_location_names=list(getattr(gpsr_vocab, "PLACEMENT_LOCATIONS", [])),
        room_names=list(getattr(gpsr_vocab, "ROOMS", [])),
        object_names=list(getattr(gpsr_vocab, "OBJECTS", [])),
        object_categories_plural=[c[1] for c in getattr(gpsr_vocab, "CATEGORIES", [])],
        object_categories_singular=[c[0] for c in getattr(gpsr_vocab, "CATEGORIES", [])],
    )


def vocab_candidates() -> Dict[str, List[str]]:
    names = [s.lower() for s in getattr(gpsr_vocab, "NAMES", [])]
    rooms = [s.lower() for s in getattr(gpsr_vocab, "ROOMS", [])]
    locs = [s.lower() for s in getattr(gpsr_vocab, "LOCATIONS", [])]
    plc = [s.lower() for s in getattr(gpsr_vocab, "PLACEMENT_LOCATIONS", [])]
    objs = [s.lower() for s in getattr(gpsr_vocab, "OBJECTS", [])]
    cats = getattr(gpsr_vocab, "CATEGORIES", [])
    cat_s = [c[0].lower() for c in cats]
    cat_p = [c[1].lower() for c in cats]
    all_phrases = sorted(set(names + rooms + locs + plc + objs + cat_s + cat_p))
    return dict(
        names=names, rooms=rooms, locations=locs, placement_locations=plc,
        objects=objs, cat_sing=cat_s, cat_plur=cat_p, all_phrases=all_phrases
    )


def suggest_close(phrase: str, candidates: List[str], n: int = 3) -> List[str]:
    phrase = (phrase or "").strip().lower()
    if not phrase:
        return []
    return difflib.get_close_matches(phrase, candidates, n=n, cutoff=0.72)


def extract_phrases_for_debug(text: str) -> Dict[str, List[str]]:
    t = normalize_text(text)
    out: Dict[str, List[str]] = collections.defaultdict(list)

    m = re.search(r"\bin the\s+(.+)$", t)
    if m:
        out["in_the"].append(m.group(1).strip())

    m = re.search(r"\bfrom the\s+(.+?)\s+to the\s+(.+)$", t)
    if m:
        out["places"].append(m.group(1).strip())
        out["places"].append(m.group(2).strip())

    for m in re.finditer(r"\bfrom the\s+(.+?)(?:$|\s+and\s+|\s+then\s+)", t):
        out["places"].append(m.group(1).strip())

    m = re.search(r"\b(?:a|an|the)\s+(.+?)(?:\s+from the|\s+in the|\s+on the|\s+to the|$)", t)
    if m:
        out["obj"].append(m.group(1).strip())

    m = re.search(r"^(?:guide|lead|escort|follow)\s+([a-z]+)\b", t)
    if m:
        out["name"].append(m.group(1).strip())

    return out


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", required=True, help="1行1コマンドのテキストファイル")
    ap.add_argument("--out", default="runs", help="出力ディレクトリ（runsなど）")
    ap.add_argument("--suggest", action="store_true", help="unknown phrase から置換候補YAMLを出力")
    args = ap.parse_args()

    cmd_path = Path(args.commands).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_run_dir(out_root)

    commands = read_commands(cmd_path)
    parser = build_parser_from_vocab()
    vc = vocab_candidates()

    jsonl_path = run_dir / "results.jsonl"
    md_path = run_dir / "summary.md"
    bad_path = run_dir / "failures.txt"
    suggest_path = run_dir / "corrections_suggested.yaml"

    total = 0
    parse_ok = 0
    intent_ctr = collections.Counter()
    kind_ctr = collections.Counter()
    fail_ctr = collections.Counter()
    unknown_ctr = collections.Counter()
    suggestion_map: Dict[str, str] = {}

    with jsonl_path.open("w", encoding="utf-8") as fj, bad_path.open("w", encoding="utf-8") as fb:
        for i, text in enumerate(commands):
            total += 1
            parsed_obj = parser.parse(text)
            rec: Dict[str, Any] = {
                "i": i,
                "text": text,
                "normalized": normalize_text(text),
                "parse_ok": False,
                "parsed": None,
            }

            if parsed_obj is None:
                fail_ctr["parse_returned_none"] += 1
                fb.write(f"[{i}] PARSE_NONE: {text}\n")
            else:
                pd = parsed_obj.to_dict()
                rec["parsed"] = pd
                rec["parse_ok"] = bool(pd.get("ok"))
                if rec["parse_ok"]:
                    parse_ok += 1
                    intent_ctr[pd.get("intent_type", "unknown")] += 1
                    kind_ctr[pd.get("command_kind", "unknown")] += 1
                else:
                    fail_ctr["parse_ok_false"] += 1
                    fb.write(f"[{i}] PARSE_FAIL: {text}\n")
                    fb.write(f"      parsed={pd}\n")

                if args.suggest and not rec["parse_ok"]:
                    ph = extract_phrases_for_debug(text)
                    for x in ph.get("in_the", []):
                        xl = x.lower()
                        if xl not in vc["rooms"] and xl not in vc["locations"]:
                            unknown_ctr[f"in_the:{xl}"] += 1
                            sug = suggest_close(xl, vc["rooms"] + vc["locations"], n=1)
                            if sug:
                                suggestion_map[xl] = sug[0]
                    for x in ph.get("places", []):
                        xl = x.lower()
                        if xl not in vc["locations"] and xl not in vc["placement_locations"]:
                            unknown_ctr[f"place:{xl}"] += 1
                            sug = suggest_close(xl, vc["locations"] + vc["placement_locations"], n=1)
                            if sug:
                                suggestion_map[xl] = sug[0]
                    for x in ph.get("obj", []):
                        xl = x.lower()
                        all_obj = vc["objects"] + vc["cat_sing"] + vc["cat_plur"]
                        if xl not in all_obj:
                            unknown_ctr[f"obj:{xl}"] += 1
                            sug = suggest_close(xl, all_obj, n=1)
                            if sug:
                                suggestion_map[xl] = sug[0]
                    for x in ph.get("name", []):
                        xl = x.lower()
                        if xl not in vc["names"]:
                            unknown_ctr[f"name:{xl}"] += 1
                            sug = suggest_close(xl, vc["names"], n=1)
                            if sug:
                                suggestion_map[xl] = sug[0]

            fj.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

    # summary
    lines: List[str] = []
    lines.append("# GPSR Eval Summary\n")
    lines.append(f"- commands: **{total}**")
    lines.append(f"- parse_ok: **{parse_ok}** ({(parse_ok/total*100 if total else 0):.1f}%)\n")

    lines.append("## Intent distribution (parse_ok)")
    for k, v in intent_ctr.most_common(30):
        lines.append(f"- {k}: {v}")
    lines.append("\n## Command kind distribution (parse_ok)")
    for k, v in kind_ctr.most_common(30):
        lines.append(f"- {k}: {v}")
    lines.append("\n## Fail reasons")
    for k, v in fail_ctr.most_common(50):
        lines.append(f"- {k}: {v}")
    if args.suggest:
        lines.append("\n## Unknown phrases (heuristic)")
        for k, v in unknown_ctr.most_common(50):
            lines.append(f"- {k}: {v}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.suggest and suggestion_map:
        try:
            import yaml
        except ImportError:
            raise SystemExit("PyYAML が必要です: pip install pyyaml")
        doc = {"suggested_corrections": dict(sorted(suggestion_map.items()))}
        suggest_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("OK")
    print("run_dir:", run_dir)
    print("summary:", md_path)
    print("results:", jsonl_path)
    print("failures:", bad_path)
    if args.suggest and suggestion_map:
        print("suggestions:", suggest_path)


if __name__ == "__main__":
    main()

