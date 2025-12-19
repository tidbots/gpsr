## チューニング
### 誤認識ログから correction 辞書を自動生成する
## 入力
```
1. /asr/text ログ
または
ASR出力ログファイル（例：asr.log）

2. GPSR 正解語彙リスト（names, locations, objects, categories）

## 出力
以下のような辞書：
```
{
  "livingroom": "living room",
  "livin room": "living room",
  "bath room": "bathroom",
  "corn flakes": "cornflakes",
}
```
## How?
1. ASRログから単語頻度を抽出
例：
```
tell me how many persons raising their right arm are in the livin room
```
→ 分割して
- livin
- room
- livingroom
- etc.

2.語彙リストと距離計算
例えば：
- Levenshtein距離
- Jaro-Winkler
- Token-based fuzzy matching
を用いると、
```
"livin room" → "living room"   (距離極小)
"bath room"  → "bathroom"      (距離極小)
```
を自動推定できます。

3. 閾値ルール
- 距離 < 2 なら補正候補
- 距離 < 4 なら人工判断必要

## ツール
```
from fuzzywuzzy import fuzz
from collections import Counter

def extract_errors(log_file, gpsr_vocab, threshold=80):
    counter = Counter()
    for line in open(log_file):
        for word in tokenize(line):
            if word.lower() not in gpsr_vocab:
                counter[word] += 1

    corrections = {}

    for wrong, _ in counter.most_common():
        best = None
        best_score = 0
        for correct in gpsr_vocab:
            score = fuzz.ratio(wrong, correct)
            if score > best_score:
                best = correct
                best_score = score

        if best_score >= threshold:
            corrections[wrong] = best

    return corrections
```
出力例：
```
{
 "livin room": "living room",
 "book shelve": "bookshelf",
 "corn flakes": "cornflakes",
}
```
これをそのまま：
```
apply_gpsr_corrections()
```
に追加すればOK。

## 運用方法
### Step 1 — ログ収集
```
rostopic echo /asr/text > asr.log
```

大会前になれば1–2時間で1万行ぐらい集まる。

### Step 2 — ツール実行
```
python gen_corrections.py asr.log
```
→ correction 辞書候補生成

### Step 3 — 人間が確認
- 明らかに誤り
- false positive でないか

### Step 4 — 修正辞書追加 & 再学習

## なぜ “自動生成”が重要か
- 長期運用時
  - 環境ノイズ差
  - 話者アクセント
  - マイク違い
→ 誤認識傾向が変わる

- ロボカップ現場
  - 英語アクセント多様
  - 語彙が限定的
→ correction辞書が強い

- 人手で気づけない誤りも拾える

つまり、
- 耳を鍛えるのではなく
- 誤識を吸収する辞書を鍛える
というアプローチ。

## 拡張案
- 自動教師あり学習との組合せ
  - wrong → correct の統計が蓄積されると
  - correction辞書が自動強化

- 特定話者モデル
  - 話者特徴に応じた correction 優先順位調整

## 実装
誤認識ログから correction 辞書を自動生成するツール一式
- gpsr_vocab.py … GPSR の語彙をひとまとめにしたモジュール
- gen_corrections.py … ASRログから correction 候補を自動生成するツール本体
- （おまけ）apply_corrections_example.py … 生成した辞書をどう使うかのサンプル
全部スタンドアロンな Python スクリプトで、ROS に依存しないので，Docker コンテナの中でそのまま動かせる

## 使い方
1. ログを集める
``@
rostopic echo /asr/text > asr.log
```

2. 辞書候補を生成する
```
python3 gen_corrections.py asr.log > corrections_candidates.py
```

3. corrections_candidates.py を開いて
- 「これは明らかに正しい」「これは怪しい」を人間が確認
- OKなものだけ CORRECTIONS として採用

4. faster_whisper_asr_node.py の apply_gpsr_corrections() に統合
5. 再度ログを取りながら、定期的に 1〜4 を回す

ゴールのイメージ

faster_whisper_asr_node.py には現在こういう関数があります：

def apply_gpsr_corrections(self, text: str) -> str:
    corrections = {
        "livingroom": "living room",
        "livin room": "living room",
        "bath room": "bathroom",
        ...
    }

    fixed = text
    for wrong, right in corrections.items():
        fixed = fixed.replace(wrong, right)
        fixed = fixed.replace(wrong.capitalize(), right)
    return fixed


💡 ここに、人間が書いた correction 辞書ではなく、
自動生成ツールで作った辞書を取り込む
というのが「統合」です。

🔧 STEP 1 — 自動生成ツールで correction 候補を作る

例えば：

python gen_corrections.py asr.log > corrections_candidates.py


この corrections_candidates.py はこういう内容になります：

# corrections_candidates.py
CORRECTIONS = {
    "livin room": "living room",  # score=92.3, count=7
    "livingroom": "living room",  # score=90.1, count=3
    "corn flakes": "cornflakes",  # score=88.1, count=4
}

🔧 STEP 2 — faster_whisper_asr_node.py に取り込む
2-1. ファイルを import する

faster_whisper_asr_node.py の上部に：

from corrections_candidates import CORRECTIONS


と1行追加します。

faster_whisper_asr_node.py
corrections_candidates.py
gpsr_vocab.py


が同じフォルダにある必要があります。

2-2. apply_gpsr_corrections() をこう変える
before
def apply_gpsr_corrections(self, text: str) -> str:
    corrections = {
        "livingroom": "living room",
        "livin room": "living room",
        "bath room": "bathroom",
        ...
    }

    fixed = text
    for wrong, right in corrections.items():
        fixed = fixed.replace(wrong, right)
    return fixed

after（統合版）
from corrections_candidates import CORRECTIONS  # 自動生成辞書を import

def apply_gpsr_corrections(self, text: str) -> str:
    fixed = text

    # 自動生成辞書を適用
    for wrong, right in CORRECTIONS.items():
        fixed = fixed.replace(wrong, right)
        fixed = fixed.replace(wrong.capitalize(), right.capitalize())

    return fixed


つまり：

前は：手書きの小さな辞書

今後は：gen_corrections.py が作る CORRECTIONS をそのまま利用

になる、ということです。

「apply_gpsr_corrections() に統合」とは
→ apply_gpsr_corrections 関数は correction 辞書を参照するだけにし、
correction辞書は外部ファイルから供給する

という意味。


運用のメリット
👍 faster_whisper_asr_node.py を編集しなくていい

誤認識辞書を育てるときは：

asr.log を収集

gen_corrections.py 実行

corrections_candidates.py を差し替え

するだけで済む。

👍 correction 辞書が肥大しても整理しやすい

apply_gpsr_corrections() 自体は変わらないので
ロジックはシンプルなまま。

👍 ロボカップ現場で高速チューニングできる

会場ノイズ

審判のアクセント

マイク特性

に応じて correction だけ更新できる。
