from pathlib import Path
import textwrap, json, datetime, os, re

base = Path("/mnt/data")
tools_dir = base / "tools"
tools_dir.mkdir(exist_ok=True)

gpsr_eval = tools_dir / "gpsr_eval.py"
content = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpsr_eval.py

GPSR command parser regression tester (ASR-bypass).
- Reads a text file with one command per line (e.g., command.txt)
- Runs gpsr_parser.GpsrParser on each command
- Saves per-line results as JSONL
- Writes a Markdown summary (success rates, failures, unknown phrases, suggestions)
- Optional: compares against a golden YAML file

Usage:
  python3 tools/gpsr_eval.py --commands /mnt/data/command.txt --out runs
  python3 tools/gpsr_eval.py --commands /mnt/data/command.txt --golden golden.yaml --out runs

Notes:
- This script expects gpsr_parser.py and gpsr_vocab.py to be importable (same directory or PYTHONPATH).
- Designed for ROS-less testing: pure Python.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as _dt
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- project imports ---
# Assume this script lives in /mnt/data/tools; add parent to sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpsr_parser import GpsrParser, normalize_text  # type: ignore
import gpsr_vocab  # type: ignore


# -------------------------
# I/O helpers
# -------------------------

def read_commands(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # allow "text<TAB>comment" format if you want later
        s = s.split("\t")[0].strip()
        if s:
            lines.append(s)
    return lines


def ensure_run_dir(root: Path) -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -------------------------
# Golden support (optional)
# -------------------------

def load_golden(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Golden YAML format (minimal):
      - text: "..."
        expect:
          intent_type: "bring"
          steps:
            - action: "bring_object_to_operator"
              args:
                source_place: "sink"
    """
    try:
        import yaml  # pyyaml
    except Exception as e:
        raise SystemExit("PyYAML is required for --golden. pip install pyyaml") from e

    data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or []
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or "").strip()
        expect = item.get("expect") or {}
        if text:
            out[text] = expect
    return out


def shallow_match(actual: Dict[str, Any], expect: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Compare only keys present in expect (partial match).
    Returns (ok, reasons).
    """
    reasons: List[str] = []

    # intent_type
    if "intent_type" in expect:
        if actual.get("intent_type") != expect["intent_type"]:
            reasons.append(f"intent_type mismatch: actual={actual.get('intent_type')} expect={expect['intent_type']}")

    # command_kind
    if "command_kind" in expect:
        if actual.get("command_kind") != expect["command_kind"]:
            reasons.append(f"command_kind mismatch: actual={actual.get('command_kind')} expect={expect['command_kind']}")

    # steps (by index)
    if "steps" in expect:
        exp_steps = expect.get("steps") or []
        act_steps = actual.get("steps") or []
        if len(act_steps) < len(exp_steps):
            reasons.append(f"steps length too short: actual={len(act_steps)} expect>={len(exp_steps)}")
        for i, es in enumerate(exp_steps):
            if i >= len(act_steps):
                break
            a = act_steps[i]
            if "action" in es and a.get("action") != es["action"]:
                reasons.append(f"step[{i}].action mismatch: actual={a.get('action')} expect={es['action']}")
            if "args" in es:
                for k, v in (es.get("args") or {}).items():
                    if a.get("args", {}).get(k) != v:
                        reasons.append(f"step[{i}].args[{k}] mismatch: actual={a.get('args', {}).get(k)} expect={v}")

    return (len(reasons) == 0), reasons


# -------------------------
# Validation rules (parser-health)
# -------------------------

REQUIRED_FIELDS_BY_ACTION: Dict[str, List[str]] = {
    # bring / fetch
    "bring_object_to_operator": [],  # object/source_place can be absent (it/them resolution later)
    "take_object": [],

    # deliver
    "deliver_object_to_person_in_room": ["room", "person_filter"],

    # navigation / guide
    "go_to_location": ["room|location"],  # one-of
    "guide_named_person_from_place_to_place": ["name", "from_place", "to_place"],

    # perception / QA
    "count_persons_in_room": ["room", "person_filter_plural"],
    "answer_to_person_in_room": ["room", "person_filter"],

    # manipulation
    "place_object_on_place": ["place"],
    "find_object_in_room": ["room"],  # object_or_category optional
}

def validate_step(step: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    action = step.get("action", "")
    args = step.get("args") or {}
    required = REQUIRED_FIELDS_BY_ACTION.get(action)
    if required is None:
        # unknown action -> not necessarily error; but we flag as warning-like error for evaluator
        errs.append(f"unknown_action:{action}")
        return errs

    for r in required:
        if "|" in r:
            # one-of
            alts = r.split("|")
            if not any((a in args and args.get(a) not in ("", None)) for a in alts):
                errs.append(f"missing_one_of:{alts}")
        else:
            if r not in args or args.get(r) in ("", None):
                errs.append(f"missing:{r}")
    return errs


# -------------------------
# Unknown phrase extraction + suggestions
# -------------------------

def build_vocab_lists() -> Dict[str, List[str]]:
    names = [x.lower() for x in gpsr_vocab.NAMES]
    rooms = [x.lower() for x in gpsr_vocab.ROOMS]
    locs = [x.lower() for x in gpsr_vocab.LOCATIONS]
    plc = [x.lower() for x in gpsr_vocab.PLACEMENT_LOCATIONS]
    objs = [x.lower() for x in gpsr_vocab.OBJECTS]
    cat_s = [c[0].lower() for c in gpsr_vocab.CATEGORIES]
    cat_p = [c[1].lower() for c in gpsr_vocab.CATEGORIES]
    return {
        "names": names,
        "rooms": rooms,
        "locations": locs,
        "placement_locations": plc,
        "objects": objs,
        "cat_sing": cat_s,
        "cat_plur": cat_p,
        "all_phrases": sorted(set(names + rooms + locs + plc + objs + cat_s + cat_p)),
    }


def suggest_close(phrase: str, candidates: List[str], n: int = 5) -> List[str]:
    phrase = phrase.strip().lower()
    if not phrase:
        return []
    # difflib works better with similar lengths; use cutoff to reduce noise
    return difflib.get_close_matches(phrase, candidates, n=n, cutoff=0.72)


def extract_phrases_for_debug(text: str) -> Dict[str, List[str]]:
    """
    Heuristic: extract likely slot phrases from surface text, regardless of parse success.
    We only use this for "unknown phrase" reporting and correction suggestions.

    Returns dict with keys: rooms, places, objects, names, person_filters
    """
    t = normalize_text(text)
    out: Dict[str, List[str]] = collections.defaultdict(list)

    # rooms: "in the <X>" or "to the <X>" where X might be room or location
    for m in re.finditer(r"\bin the\s+([a-z0-9\- ]+)$", t):
        out["rooms"].append(m.group(1).strip())
    for m in re.finditer(r"\bto the\s+([a-z0-9\- ]+)$", t):
        out["rooms_or_places"].append(m.group(1).strip())

    # from/to places: "from the <X> to the <Y>"
    m = re.search(r"\bfrom the\s+(.+?)\s+to the\s+(.+)$", t)
    if m:
        out["places"].append(m.group(1).strip())
        out["places"].append(m.group(2).strip())

    # "from the <X>"
    for m in re.finditer(r"\bfrom the\s+(.+?)(?:$|\s+then\s+|\s+and\s+)", t):
        out["places"].append(m.group(1).strip())

    # "on the <X>" / "in the <X>" (placement)
    for m in re.finditer(r"\b(?:on|in)\s+the\s+(.+)$", t):
        out["places"].append(m.group(1).strip())

    # names: after guide/lead/escort/follow/find <NAME> ... (rough)
    m = re.search(r"^(?:guide|lead|escort|follow)\s+([a-z]+)\b", t)
    if m:
        out["names"].append(m.group(1).strip())

    # objects: "a/an/the <OBJ>" (very rough; exclude common pronouns)
    m = re.search(r"\b(?:a|an|the)\s+([a-z0-9\- ]+?)(?:\s+from the|\s+in the|\s+on the|\s+to the|$)", t)
    if m:
        candidate = m.group(1).strip()
        if candidate not in ("person", "persons", "people"):
            out["objects_or_categories"].append(candidate)

    # person filter in quiz/deliver: "the <FILTER> in the <ROOM>"
    m = re.search(r"\bto the\s+(.+?)\s+in the\s+(.+)$", t)
    if m:
        out["person_filters"].append(m.group(1).strip())

    return out


def build_parser() -> GpsrParser:
    v = build_vocab_lists()
    return GpsrParser(
        person_names=[x for x in gpsr_vocab.NAMES],
        location_names=[x for x in gpsr_vocab.LOCATIONS],
        placement_location_names=[x for x in gpsr_vocab.PLACEMENT_LOCATIONS],
        room_names=[x for x in gpsr_vocab.ROOMS],
        object_names=[x for x in gpsr_vocab.OBJECTS],
        object_categories_plural=[c[1] for c in gpsr_vocab.CATEGORIES],
        object_categories_singular=[c[0] for c in gpsr_vocab.CATEGORIES],
    )


# -------------------------
# Main evaluation
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", required=True, type=str, help="Path to command list text file (one command per line).")
    ap.add_argument("--out", default="runs", type=str, help="Output root directory.")
    ap.add_argument("--golden", default="", type=str, help="Optional golden YAML file.")
    ap.add_argument("--suggest", action="store_true", help="Write suggested corrections YAML (unknown->best guess).")
    args = ap.parse_args()

    cmd_path = Path(args.commands).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = ensure_run_dir(out_root)

    commands = read_commands(cmd_path)

    parser = build_parser()
    vocab_lists = build_vocab_lists()

    golden: Dict[str, Dict[str, Any]] = {}
    if args.golden:
        golden = load_golden(Path(args.golden).expanduser().resolve())

    # stats
    total = 0
    ok_parse = 0
    ok_validate = 0
    golden_ok = 0

    intent_counter = collections.Counter()
    kind_counter = collections.Counter()
    failure_reasons = collections.Counter()
    missing_fields = collections.Counter()
    unknown_phrases = collections.Counter()
    suggestion_map: Dict[str, str] = {}

    # output files
    jsonl_path = run_dir / "results.jsonl"
    md_path = run_dir / "summary.md"
    bad_path = run_dir / "failures.txt"
    suggest_path = run_dir / "corrections_suggested.yaml"

    with jsonl_path.open("w", encoding="utf-8") as f_jsonl, bad_path.open("w", encoding="utf-8") as f_bad:
        for idx, text in enumerate(commands):
            total += 1
            cmd = text.strip()
            parsed = parser.parse(cmd)
            record: Dict[str, Any] = {
                "i": idx,
                "text": cmd,
                "normalized": normalize_text(cmd),
                "parsed": None,
                "parse_ok": False,
                "validate_ok": False,
                "validation_errors": [],
                "golden_checked": False,
                "golden_ok": None,
                "golden_reasons": [],
            }

            if parsed is None:
                failure_reasons["parse_none"] += 1
                f_bad.write(f"[{idx}] PARSE_NONE: {cmd}\n")
            else:
                pd = parsed.to_dict()
                record["parsed"] = pd
                record["parse_ok"] = bool(pd.get("ok"))

                if record["parse_ok"]:
                    ok_parse += 1
                    intent_counter[pd.get("intent_type", "unknown")] += 1
                    kind_counter[pd.get("command_kind", "unknown")] += 1

                # validation
                val_errs: List[str] = []
                if pd.get("steps"):
                    for s in pd["steps"]:
                        val_errs.extend(validate_step(s))
                else:
                    val_errs.append("no_steps")

                record["validation_errors"] = val_errs
                record["validate_ok"] = (len(val_errs) == 0)

                if record["validate_ok"] and record["parse_ok"]:
                    ok_validate += 1
                else:
                    if not record["parse_ok"]:
                        failure_reasons["parse_failed"] += 1
                    for e in val_errs:
                        missing_fields[e] += 1
                    f_bad.write(f"[{idx}] FAIL: {cmd}\n")
                    if pd.get("steps"):
                        f_bad.write(f"      steps={pd.get('steps')}\n")
                    if val_errs:
                        f_bad.write(f"      errors={val_errs}\n")

            # golden compare
            if golden:
                exp = golden.get(cmd)
                if exp is not None and record["parsed"] is not None:
                    record["golden_checked"] = True
                    ok, reasons = shallow_match(record["parsed"], exp)
                    record["golden_ok"] = ok
                    record["golden_reasons"] = reasons
                    if ok:
                        golden_ok += 1
                    else:
                        failure_reasons["golden_mismatch"] += 1

            # unknown phrase reporting + suggestions (surface-heuristic)
            # Only run if parse failed OR validation failed, to reduce noise.
            if not record["parse_ok"] or not record["validate_ok"]:
                phrases = extract_phrases_for_debug(cmd)

                # rooms
                for ph in phrases.get("rooms", []):
                    if ph.lower() not in vocab_lists["rooms"]:
                        unknown_phrases[("room:" + ph.lower())] += 1
                        sugg = suggest_close(ph, vocab_lists["rooms"], n=3)
                        if sugg:
                            suggestion_map[ph.lower()] = sugg[0]

                # rooms_or_places
                for ph in phrases.get("rooms_or_places", []):
                    if ph.lower() not in vocab_lists["rooms"] and ph.lower() not in vocab_lists["locations"]:
                        unknown_phrases[("dest:" + ph.lower())] += 1
                        sugg = suggest_close(ph, vocab_lists["rooms"] + vocab_lists["locations"], n=3)
                        if sugg:
                            suggestion_map[ph.lower()] = sugg[0]

                # places
                for ph in phrases.get("places", []):
                    if ph.lower() not in vocab_lists["locations"] and ph.lower() not in vocab_lists["placement_locations"]:
                        unknown_phrases[("place:" + ph.lower())] += 1
                        sugg = suggest_close(ph, vocab_lists["locations"] + vocab_lists["placement_locations"], n=3)
                        if sugg:
                            suggestion_map[ph.lower()] = sugg[0]

                # objects/categories
                for ph in phrases.get("objects_or_categories", []):
                    all_obj = vocab_lists["objects"] + vocab_lists["cat_sing"] + vocab_lists["cat_plur"]
                    if ph.lower() not in all_obj:
                        unknown_phrases[("obj:" + ph.lower())] += 1
                        sugg = suggest_close(ph, all_obj, n=3)
                        if sugg:
                            suggestion_map[ph.lower()] = sugg[0]

                # names
                for ph in phrases.get("names", []):
                    if ph.lower() not in vocab_lists["names"]:
                        unknown_phrases[("name:" + ph.lower())] += 1
                        sugg = suggest_close(ph, vocab_lists["names"], n=3)
                        if sugg:
                            suggestion_map[ph.lower()] = sugg[0]

            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

    # write markdown summary
    def top_items(counter: collections.Counter, k: int = 15) -> List[Tuple[Any, int]]:
        return counter.most_common(k)

    lines: List[str] = []
    lines.append(f"# GPSR Parser Eval Summary")
    lines.append("")
    lines.append(f"- commands: **{total}**")
    lines.append(f"- parse_ok: **{ok_parse}** ({(ok_parse/total*100 if total else 0):.1f}%)")
    lines.append(f"- validate_ok (and parse_ok): **{ok_validate}** ({(ok_validate/total*100 if total else 0):.1f}%)")
    if golden:
        lines.append(f"- golden checked: **{len(golden)}** (only exact text matches are checked)")
        lines.append(f"- golden_ok: **{golden_ok}** ({(golden_ok/len(golden)*100 if golden else 0):.1f}%)")
    lines.append("")
    lines.append("## Intent type distribution (parse_ok)")
    for k, v in top_items(intent_counter, 30):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Command kind distribution (parse_ok)")
    for k, v in top_items(kind_counter, 30):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Failure reasons")
    for k, v in top_items(failure_reasons, 30):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Validation / missing fields (top)")
    for k, v in top_items(missing_fields, 30):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Unknown phrases (heuristic, top)")
    for k, v in top_items(unknown_phrases, 40):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- results: `{jsonl_path.name}`")
    lines.append(f"- failures: `{bad_path.name}`")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # optional suggested corrections yaml
    if args.suggest and suggestion_map:
        try:
            import yaml  # pyyaml
        except Exception as e:
            raise SystemExit("PyYAML is required for --suggest. pip install pyyaml") from e
        # Output as a simple mapping (lowercase)
        # NOTE: you likely want to review before merging into your real corrections.yaml
        doc = {"suggested_corrections": dict(sorted(suggestion_map.items(), key=lambda x: x[0]))}
        suggest_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("OK")
    print("run_dir:", run_dir)
    print("results:", jsonl_path)
    print("summary:", md_path)
    print("failures:", bad_path)
    if args.suggest and suggestion_map:
        print("suggestions:", suggest_path)


if __name__ == "__main__":
    main()
'''
gpsr_eval.write_text(content, encoding="utf-8")
os.chmod(gpsr_eval, 0o755)

golden_example = tools_dir / "golden_example.yaml"
golden_example.write_text(textwrap.dedent("""\
- text: "Tell me how many standing persons are in the kitchen"
  expect:
    intent_type: "answer"
    steps:
      - action: "count_persons_in_room"
        args:
          room: "kitchen"
          person_filter_plural: "standing persons"

- text: "Guide Jules from the sink to the sofa"
  expect:
    intent_type: "guide"
    steps:
      - action: "guide_named_person_from_place_to_place"
        args:
          name: "Jules"
          from_place: "sink"
          to_place: "sofa"

- text: "Bring me a red wine from the bedside table"
  expect:
    intent_type: "bring"
    steps:
      - action: "bring_object_to_operator"
        args:
          source_place: "bedside table"
          object: "red wine"
"""), encoding="utf-8")

readme = tools_dir / "README_gpsr_eval.md"
readme.write_text(textwrap.dedent("""\
# GPSR Parser Eval Tool (ASR-bypass)

このツールは、`command.txt` のように大量に列挙された GPSR コマンド文を使って、
`gpsr_parser.py` の解析部分だけを自動で回帰テストするための CLI です。

## 使い方

### 1) 解析だけ回す（goldenなし）
```bash
python3 tools/gpsr_eval.py --commands /mnt/data/command.txt --out /mnt/data/runs
