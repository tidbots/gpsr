#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple


def parse_md_table_single_col(md: str) -> List[str]:
    """| Names | のような1列テーブルから値を抽出"""
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    out = []
    for ln in lines:
        if not ln.startswith("|"):
            continue
        # header/separator skip
        if re.search(r"\|\s*-+\s*\|", ln):
            continue
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) < 1:
            continue
        v = cols[0]
        # header skip
        if v.lower() in ("names", "name"):
            continue
        out.append(v)
    return [x for x in out if x]


def parse_locations(md: str) -> List[Dict[str, Any]]:
    """
    location_names.md:
    | 1 | bed (p) |
    | 3 | shelf (p) | cleaning supplies |
    """
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    locs = []
    for ln in lines:
        if not ln.startswith("|"):
            continue
        if re.search(r"\|\s*-+\s*\|", ln):
            continue
        cols = [c.strip() for c in ln.strip("|").split("|")]
        # expecting: Number | Name | (optional category)
        if len(cols) < 2:
            continue
        if cols[0].lower() in ("number", "#"):
            continue

        name_raw = cols[1]
        placement = False
        # "(p)" mark
        m = re.search(r"\(p\)", name_raw)
        if m:
            placement = True
            name_raw = re.sub(r"\(p\)", "", name_raw).strip()

        entry = {"name": name_raw, "placement": placement}
        if len(cols) >= 3 and cols[2] and cols[2].lower() != "object category":
            entry["category_hint"] = cols[2]
        locs.append(entry)
    return locs


def parse_objects(md: str) -> List[Dict[str, Any]]:
    """
    objects.md:
      # Class drinks (drink)
      | juice_pack | ... |
    を classブロックごとに拾う
    """
    cats = []
    cur_key = None
    cur_singular = None
    cur_objects = []

    for ln in md.splitlines():
        ln = ln.strip()

        # class header: "# Class drinks (drink)"
        m = re.match(r"^#\s*Class\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)", ln)
        if m:
            # flush previous
            if cur_key:
                cats.append({
                    "key": cur_key,
                    "singular": cur_singular,
                    # pluralは “key” をそのまま英語複数形として使う（必要なら後で手修正）
                    "plural": cur_key,
                    "objects": cur_objects
                })
            cur_key = m.group(1)
            cur_singular = m.group(2)
            cur_objects = []
            continue

        # table row: "| juice_pack | ![](...) |"
        if ln.startswith("|") and cur_key:
            cols = [c.strip() for c in ln.strip("|").split("|")]
            if not cols:
                continue
            obj = cols[0]
            # header row skip
            if obj.lower() in ("objectname", "object", "name"):
                continue
            if re.fullmatch(r":-+", obj):  # alignment row
                continue
            if obj:
                cur_objects.append(obj)

    # last flush
    if cur_key:
        cats.append({
            "key": cur_key,
            "singular": cur_singular,
            "plural": cur_key,
            "objects": cur_objects
        })

    return cats


def to_yaml(data: Dict[str, Any]) -> str:
    """最小YAML出力（依存なしの手書き整形）"""
    def esc(s: str) -> str:
        # simple: quote only if needed
        if re.search(r"[:\[\]\{\},#&\*\!\|>\'\"\s]", s):
            return '"' + s.replace('"', '\\"') + '"'
        return s

    out = []
    out.append("schema: gpsr_vocab_v1\n")

    out.append("names:\n")
    for n in data["names"]:
        out.append(f"  - {esc(n)}\n")

    out.append("\nrooms:\n")
    for r in data["rooms"]:
        out.append(f"  - {esc(r)}\n")

    out.append("\nlocations:\n")
    for loc in data["locations"]:
        out.append(f"  - name: {esc(loc['name'])}\n")
        out.append(f"    placement: {str(bool(loc.get('placement', False))).lower()}\n")
        if "category_hint" in loc:
            out.append(f"    category_hint: {esc(loc['category_hint'])}\n")

    out.append("\ncategories:\n")
    for c in data["categories"]:
        out.append(f"  - key: {esc(c['key'])}\n")
        out.append(f"    singular: {esc(c['singular'])}\n")
        out.append(f"    plural: {esc(c['plural'])}\n")
        objs = ", ".join([esc(o) for o in c["objects"]])
        out.append(f"    objects: [{objs}]\n")

    return "".join(out)


def main():
    if len(sys.argv) != 6:
        print("Usage: make_vocab_yaml.py names.md room_names.md location_names.md objects.md out.yaml", file=sys.stderr)
        sys.exit(2)

    names_md = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
    rooms_md = Path(sys.argv[2]).read_text(encoding="utf-8", errors="ignore")
    loc_md = Path(sys.argv[3]).read_text(encoding="utf-8", errors="ignore")
    obj_md = Path(sys.argv[4]).read_text(encoding="utf-8", errors="ignore")
    out_path = Path(sys.argv[5])

    data = {
        "names": parse_md_table_single_col(names_md),
        "rooms": parse_md_table_single_col(rooms_md),
        "locations": parse_locations(loc_md),
        "categories": parse_objects(obj_md),
    }

    out_path.write_text(to_yaml(data), encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
