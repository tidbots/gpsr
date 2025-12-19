#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gen_corrections.py

ASR ログ (例: `rostopic echo /asr/text > asr.log`) を解析して、
GPSR 語彙に対する「誤認識 → 正しい単語」候補を自動生成するツール。

使い方例:

    python gen_corrections.py asr.log > corrections_candidates.py

出力は Python の dict として貼り付け可能な形:

    CORRECTIONS = {
        "livin room": "living room",
        "livingroom": "living room",
        ...
    }
"""

import sys
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

import gpsr_vocab


def extract_text_from_line(line: str) -> str:
    """
    rostopic echo /asr/text の行かもしれない文字列から生テキストを取り出す。

    例:
        data: "Find a sponge in the living room then get it and bring it to me"
        → Find a sponge in the living room then get it and bring it to me

    それ以外の形式なら、行全体をテキストとして扱う。
    """
    line = line.strip()

    # パターン1: data: "...."
    m = re.search(r'data:\s*"(.*)"', line)
    if m:
        return m.group(1)

    # パターン2: data: '....'
    m = re.search(r"data:\s*'(.*)'", line)
    if m:
        return m.group(1)

    # その他はそのまま
    return line


def tokenize(text: str) -> List[str]:
    """
    簡易トークナイザ: 記号を削ってスペース区切り。
    """
    text = text.lower()
    # 句読点などをスペースに
    text = re.sub(r"[.,!?;:]", " ", text)
    # 連続スペースを1個に
    text = re.sub(r"\s+", " ", text)
    return [t for t in text.split(" ") if t]


def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    トークン列から n-gram (1〜3語の連結) を生成。
    例: ["find", "a", "sponge", "in", "the", "living", "room"]
      n=2 → "find a", "a sponge", ..., "living room"
    """
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_ngram_counts(lines: List[str]) -> Counter:
    """
    ASR ログ行のリストから n-gram 出現頻度を数える。
    - 1-gram, 2-gram, 3-gram をすべてカウント
    """
    counter = Counter()
    for line in lines:
        text = extract_text_from_line(line)
        tokens = tokenize(text)
        if not tokens:
            continue

        # 1-gram ~ 3-gram を集計
        for n in (1, 2, 3):
            for ng in generate_ngrams(tokens, n):
                counter[ng] += 1

    return counter


def best_vocab_match(
    phrase: str,
    vocab: List[str],
) -> Tuple[str, float]:
    """
    与えられた phrase に最も近い語彙エントリを difflib.SequenceMatcher で探す。
    戻り値: (best_match_phrase, score[0-100])
    """
    best_phrase = ""
    best_score = 0.0
    for v in vocab:
        # 長すぎる or 短すぎるのは無視（効率 & ノイズ削減のため）
        if len(v) < 3 or len(phrase) < 3:
            continue

        score = SequenceMatcher(None, phrase, v).ratio() * 100.0
        if score > best_score:
            best_score = score
            best_phrase = v

    return best_phrase, best_score


def infer_corrections(
    ngram_counter: Counter,
    vocab_set: set,
    min_count: int = 2,
    min_score: float = 80.0,
) -> Dict[str, Dict]:
    """
    n-gram 頻度と語彙から「誤認識 → correction」候補を推定する。

    戻り値は:
      {
        "wrong_phrase": {
            "correct": "living room",
            "score": 91.2,
            "count": 5,
        },
        ...
      }
    という dict。
    """
    vocab_list = list(vocab_set)

    # まず「GPSR 語彙そのもの」は候補から除外
    candidates = {
        phrase: count
        for phrase, count in ngram_counter.items()
        if phrase not in vocab_set
    }

    corrections: Dict[str, Dict] = {}

    for phrase, count in candidates.items():
        if count < min_count:
            # 1回しか出てこないようなものはノイズが多いので無視
            continue

        best_phrase, best_score = best_vocab_match(phrase, vocab_list)
        if best_score >= min_score and best_phrase:
            # 既に登録済みの場合は、スコア・件数がより大きい方を採用
            if phrase not in corrections:
                corrections[phrase] = {
                    "correct": best_phrase,
                    "score": best_score,
                    "count": count,
                }
            else:
                prev = corrections[phrase]
                if best_score > prev["score"] or count > prev["count"]:
                    corrections[phrase] = {
                        "correct": best_phrase,
                        "score": best_score,
                        "count": count,
                    }

    return corrections


def print_corrections_as_python_dict(corrections: Dict[str, Dict]):
    """
    apply_gpsr_corrections() に貼り付けやすいように、
    Python の dict として出力。
    """
    print("# -*- coding: utf-8 -*-")
    print("# 自動生成した correction 候補")
    print("CORRECTIONS = {")
    for wrong, info in sorted(
        corrections.items(),
        key=lambda kv: (-kv[1]["count"], -kv[1]["score"]),
    ):
        correct = info["correct"]
        score = info["score"]
        count = info["count"]
        print(f'    "{wrong}": "{correct}",  # score={score:.1f}, count={count}')
    print("}")


def print_summary_table(corrections: Dict[str, Dict]):
    """
    人間がざっと目視する用のサマリ表。
    """
    print("## Detected correction candidates (sorted by freq/score)")
    print("wrong_phrase\t->\tcorrect\t(score, count)")
    for wrong, info in sorted(
        corrections.items(),
        key=lambda kv: (-kv[1]["count"], -kv[1]["score"]),
    ):
        print(
            f"{wrong}\t->\t{info['correct']}\t"
            f"(score={info['score']:.1f}, count={info['count']})"
        )


def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_corrections.py <asr_log_file> "
              "[min_count] [min_score]")
        print("Example: python gen_corrections.py asr.log 2 80")
        sys.exit(1)

    log_path = sys.argv[1]
    min_count = int(sys.argv[2]) if len(sys.argv) >= 3 else 2
    min_score = float(sys.argv[3]) if len(sys.argv) >= 4 else 80.0

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # GPSR 語彙セット
    vocab_set = gpsr_vocab.get_full_vocab(include_single_words=True)

    # n-gram 出現頻度
    ngram_counter = build_ngram_counts(lines)

    # 誤認識 → 語彙 の候補推定
    corrections = infer_corrections(
        ngram_counter,
        vocab_set,
        min_count=min_count,
        min_score=min_score,
    )

    # まずサマリ表を STDERR に
    print_summary_table(corrections)

    # そして Python dict を STDOUT に（リダイレクトして使う想定）
    # 実運用では: python gen_corrections.py asr.log > corrections_candidates.py
    print("\n\n# --- Python dict snippet ---\n")
    print_corrections_as_python_dict(corrections)


if __name__ == "__main__":
    main()
