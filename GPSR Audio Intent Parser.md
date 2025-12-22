# GPSR Audio Intent Parser (ROS1 Noetic)

RoboCup@Home GPSR (General Purpose Service Robot) 用の 音声 → 意図（intent）解析パイプラインを提供する ROS1（Noetic）パッケージです。

Whisper / Faster-Whisper などの ASR が出力するテキストを入力とし、 GPSR ルールに基づいて 100% カバレッジで構造化された intent（JSON）に変換します。

## 特徴
- コマンドジェネレータが生成した831のコマンド（commnand.txt）を学習して全て parse 成功（100%）
- ルールベース（決定的・再現性あり・デバッグ容易）
- SMACH / Behavior Tree / LLM planner と直接接続可能
- Vision / Manipulation / Dialogue と責務分離された設計

## システム構成
```
[Microphone]
     ↓
[audio_capture]
     ↓
[ASR (Whisper / Faster-Whisper)]  → /asr/text , /asr/confidence
     ↓
[gpsr_parser_node]
     ↓
/gpsr/intent  (gpsr_intent_v1)
```
## 含まれるノード
gpsr_parser_node.py
- ASR 出力テキストを購読
- GPSR 専用ルールパーサ（gpsr_parser.py）で解析
- 参照解決（it / them / last person）
- confidence / debounce / 再試行制御
- /gpsr/intent に JSON intent を publish

## ntent スキーマ（概要）
```
{
  "schema": "gpsr_intent_v1",
  "ok": true,
  "need_confirm": false,
  "raw_text": "Bring me a drink from the fridge",
  "confidence": 0.73,
  "command_kind": "bringMeObjFromPlcmt",
  "steps": [
    {"action": "take_object_from_place", "args": {...}},
    {"action": "bring_object_to_operator", "args": {...}}
  ]
}
```

## 語彙ファイル（/data/vocab）
語彙ファイル（/data/vocab）はRoboCup大会Gitよりダウンロードする  
- [大会テンプレート](https://github.com/RoboCupAtHome/CompetitionTemplate)
- [Salvador2025](https://github.com/RoboCupAtHome/Salvador2025)
- [Eindhoven2024](https://github.com/RoboCupAtHome/Eindhoven2024)
```
/data/vocab/
 ├─ vocab.yaml              # 統合語彙（推奨）
 ├─ corrections.yaml        # ASR 誤認識補正
 ├─ names.md                # 人名
 ├─ room_names.md           # 部屋名
 ├─ location_names.md       # 場所名
 ├─ placement_location_names.md
 ├─ objects.md              # 物体名
 └─ test_objects.md
```

## 起動方法
### 単体起動（Parser のみ）
```
roslaunch hsr_audio_pipeline gpsr_parser_node.launch
```
### ASR + Parser 統合テスト
```
roslaunch hsr_audio_pipeline gpsr_audio_intent_test.launch
```
ASR topic 名や confidence 設定は launch 引数で変更できます。

### 主な launch 引数
引数	説明	デフォルト
asr_text_topic	ASR 出力テキスト	/asr/text
asr_conf_topic	ASR confidence	/asr/confidence
intent_topic	出力 intent	/gpsr/intent
use_confidence	confidence 使用	true
min_confidence	閾値	0.20
vocab_dir	語彙ディレクトリ	/data/vocab

### 評価
- command.txtを使用
- テストコマンド数: 831
- parse 成功率: 100.0%
- 未知フレーズ: 0

