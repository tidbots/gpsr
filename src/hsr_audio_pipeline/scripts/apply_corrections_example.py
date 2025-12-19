# apply_corrections_example.py
# gen_corrections.py の出力をどう使うかのサンプル。

from typing import Dict

# これは自動生成された corrections_candidates.py からコピペしてくる想定
CORRECTIONS = {
    "livin room": "living room",  # score=92.3, count=7
    "livingroom": "living room",  # score=90.1, count=3
    # ...
}


def apply_corrections(text: str, corrections: Dict[str, str]) -> str:
    """ASR テキストに誤認識補正を適用する簡易関数."""
    fixed = text
    for wrong, right in corrections.items():
        fixed = fixed.replace(wrong, right)
        # 最初の文字だけ大文字のパターンもついでに補正
        w_cap = wrong.capitalize()
        r_cap = right.capitalize()
        fixed = fixed.replace(w_cap, r_cap)
    return fixed


if __name__ == "__main__":
    s = "Tell me how many persons are in the livin room"
    print("before:", s)
    print("after :", apply_corrections(s, CORRECTIONS))
