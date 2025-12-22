# General Purpose Service Robot 汎用サービスロボット
ロボットは、幅広く異なる能力を必要とする命令を理解し実行することが求められる。




## ToDo
### 21 Dec.2025
~~Whisperのモデルを起動時に毎回ダウンロードしているので無駄<br>~~
~~Silero VADでtorch.hubでmaster.zipを毎回ダウンロードしている<br>~~
->Docker側で/modelを永続化する

### 20 Dec. 2025
パラメータ他を自動生成
```
[vocab.yaml]  ←←← 年1回更新
     │
     ├─ gpsr_parser_node
     │    └─ 厳密な構文・意味解釈（正解判定）
     │
     └─ faster_whisper_asr_node
          ├─ hotwords 生成
          ├─ correction 辞書生成
          └─ 正規化テキスト出力
```

### 20 Dec.2025
**未テスト**
テスト（ASR→intentまで）：
```
roslaunch hsr_audio_pipeline gpsr_audio_intent_test.launch
```

実行（SMACHまで）：
```
roslaunch hsr_audio_pipeline gpsr_run.launch
```

確認：
```
rostopic echo /gpsr/asr/text
rostopic echo /gpsr/asr/utterance_end
rostopic echo /gpsr/intent
rostopic echo /gpsr/task_state
rostopic echo /gpsr/result
```

### 20 Dec.2025
~~**パラメータ他をハードコーティングしている**~~
->ファイルから読むようにした

~~**initial_prompt に入れておくと良い単語の微調整**~~
->hot word

~~**誤認識ログの収集と登録**~~
->ツールを作成した
```
python3 tools/make_vocab_yaml.py \
  names/names.md \
  maps/room_names.md \
  maps/location_names.md \
  objects/objects.md \
  hsr_audio_pipeline/config/vocab.yaml
```

## インストールとDockerコンテナの実行
```
cd ~/
git clone https://github.com/tidbots/gpsr.git
cd ~/gpsr
export PULSE_SERVER=unix:/run/user/1000/pulse/native
docker compose build
docker compose up -d
```

別のターミナルから
```
cd ~/gpsr
export PULSE_SERVER=unix:/run/user/1000/pulse/native
docker compose exec noetic-audio bash
roslaunch hsr_audio_pipeline gpsr_audio_intent_test.launch
```

## 認識した文字列を表示
別のターミナルから
```
cd ~/gpsr
export PULSE_SERVER=unix:/run/user/1000/pulse/native
docker compose exec noetic-audio bash
rostopic echo /asr/text
```
文字化けしないバージョン
```
rosrun hsr_audio_pipeline asr_plain_echo.py
```

## コマンドを解析した文字列を表示
別のターミナルから
```
cd ~/gpsr
export PULSE_SERVER=unix:/run/user/1000/pulse/native
docker compose exec noetic-audio bash
rostopic echo  /gpsr/intent_json
```
文字化けしないバージョン
```
rosrun hsr_audio_pipeline gpsr_intent_echo.py
```


## マイクに向かって発話する
命令文はCommandGeneratorで作る<br>
[CommandGenerator](https://github.com/RoboCupAtHome/CommandGenerator)

命令のサンプル (command.txt)
```
Tell me how many people in the bathroom are wearing white sweaters
Tell the gesture of the person at the bedside table to the person at the dishwasher
Tell me what is the biggest cleaning supply on the shelf Bring me a spam from the bed
Tell me how many people in the kitchen are wearing white shirts
Navigate to the desk lamp then look for a tuna and grasp it and give it to the lying person in the bathroom
Tell the name of the person at the entrance to the person at the sofa
Fetch a food from the sofa and deliver it to me
Tell me how many drinks there are on the dishwasher
Tell me what is the lightest cleaning supply on the sofa
Tell me how many lying persons are in the bathroom
Tell me how many food there are on the sink
...
...
```

## 結果
「テーブルの上のペットボトルを持ってきて」 と



## 誤認識ログを使って品質を改良する
ロガー　gpsr_asr_logger.py
```
rosrun hsr_audio_pipeline gpsr_asr_logger.py _our_dir:=/home/roboworks/logs/gpsr
```
utterace_end=Trueのたびに
```
{"ts":"...","text":"bring me a plum from the tv stand.","confidence":0.43}
```




## デバッグ
命令文を直接配信する
```
rostopic pub /asr/text std_msgs/String \
"data: 'Find a cleaning supply in the bedroom then get it and put it on the refrigerator'" -1
```

別のターミナルで
gpsr_parser_node は utterance_end がトリガなので、テスト時はセットで叩く：
```
rostopic pub -1 /gpsr/asr/text std_msgs/String "data: '...'"
rostopic pub -1 /gpsr/asr/utterance_end std_msgs/Bool "data: true"
rostopic echo -n 1 /gpsr/intent_json
```

シンプル ASR 動作確認
VADをスキップ
コンテナ内で：
```
cd /hsr_ws
source devel/setup.bash
roslaunch hsr_audio_pipeline audio_asr_simple_test.launch
```


## ファイル構成
```
/Users/roboworks/gpsr
├─ compose.yaml
├─ Dockerfile
├─ LICENSE
├─ README.md
├─ README_コマンド.md
├─ RuleBook.md
├─ UPDATE.md
├─ doc.md
├─ ERROR_CORRECTIONS.md
├─ command.txt # コマンドジェネレータで生成したコマンド
│
├─ gpsr_data/ # Docker volume 永続化領域
│  ├─ logs/
│  │  └─ gpsr/
│  │     └─ dummy
│  ├─ models/
│  │  ├─ hf/
│  │  │  └─ dummy
│  │  └─ torch/
│  │     └─ dummy
│  └─ vocab/ # ★ ASR補正 + 語彙（永続化の核）
│     ├─ vocab.yaml          # vocab.yaml is optional and can be generated from individual vocabulary files.
│     ├─ corrections.yaml
│     ├─ names.md
│     ├─ room_names.md
│     ├─ location_names.md
│     ├─ objects.md
│     └─ test_objects.md
│
└─ src/
   └─ hsr_audio_pipeline/                   # ROS1 Noetic package
      ├─ package.xml
      ├─ CMakeLists.txt
      │
      ├─ launch/
      │  ├─ audio_pipeline.launch
      │  ├─ audio_test.launch
      │  ├─ audio_vad_test.launch
      │  ├─ audio_asr_simple_test.launch
      │  ├─ audio_vad_asr_test.launch
      │  ├─ gpsr_audio_intent_test.launch
      │  ├─ gpsr_audio_intent_test_updated.launch
      │  ├─ gpsr_parser_node_updated.launch
      │  ├─ gpsr_run.launch
      │  └─ gpsr_smach_test.launch
      │
      ├─ scripts/
      │  ├─ faster_whisper_asr_node.py
      │  ├─ silero_vad_node.py
      │  ├─ audio_rms_monitor.py
      │  ├─ asr_plain_echo.py
      │  ├─ gpsr_asr_logger.py
      │  ├─ gpsr_intent_echo.py
      │  ├─ gpsr_intent_node.py
      │  ├─ gpsr_smach_node.py
      │  ├─ gpsr_vocab.py
      │  ├─ gpsr_parser.py                  # ★ 100% パーサ本体
      │  ├─ gpsr_parser.py-bk
      │  ├─ gpsr_parser_node.py             # ★ パーサROSノード（本命）
      │  └─ gpsr_parser_node2.py
      │
      └─ tools/
         ├─ gpsr_eval.py
         ├─ make_vocab_yaml.py
         ├─ command.txt
         ├─ jsonl_to_corrections.py
         ├─ aaa/                            # eval 出力（タイムスタンプ別）
         │  ├─ 20251222-084603/
         │  │  ├─ results.jsonl
         │  │  ├─ failures.txt
         │  │  ├─ summary.md
         │  │  └─ corrections_suggested.yaml
         │  ├─ 20251222-092011/
         │  └─ ...
         └─ aaa                              # (他の補助物がある場合)

```

## 1. 全体パイプライン
```
[Microphone]
     ↓
[audio_capture]
     ↓  /audio/audio
[ audio_rms_monitor ]   [ silero_vad_node ]（任意）
        │                   ↓ /vad/segments
        └──────────────→ [ faster_whisper_asr_node ]
                                ↓ /asr/text (+ confidence)
                         [ gpsr_parser_node ]
                                ↓ /gpsr/intent (JSON)
                         [ gpsr_smach_node ]
                                ↓
          ┌──────── navigation ─────────┐
          │                               │
[ hsr_nav_client_node ]     [ hsr_task_executor_node ]
          │                               │
     move_base                         HSR操作系
```

## 2. 各ノードの役割
### Audio / ASR 系
1. audio_capture ノード（既存）
- 役割
  - マイク入力を ROS トピック /audio/audio に配信
- 出力
  - /audio/audio (audio_common_msgs/AudioData)
- 備考
  - Docker 環境では ALSA / PulseAudio をホストとブリッジ
  - サンプリングレート・フォーマットは launch で制御

2. audio_rms_monitor / silero_vad_node（任意）
audio_rms_monitor
- 役割
  - RMS / PEAK をログ出力
  - 入力音量・マイク設定のデバッグ用
- 本番必須ではない

silero_vad_node（任意）
- 入力
  - /audio/audio

- 出力
  - /vad/segments

- 役割
  - 発話区間（speech / non-speech）の検出
  - Whisper に渡す音声区間のトリガ

- 設計方針
  - 静かな環境では 無効化しても可
  - ノイズ環境・長時間運用では有効

3. faster_whisper_asr_node
- 入力
  - /audio/audio
  - /vad/segments（任意）

- 出力
  - /asr/text (std_msgs/String)
  - /asr/confidence（任意）
  - /asr/partial_text（任意）

- 役割
  - Whisper（tiny.en など）による 英語ロングセンテンス認識
  - GPSR 用に連続音声を安定して文字起こし

- 設計ポイント
  - 文法制約は行わない（自由認識）
  - 誤認識は後段の corrections / parser で吸収

### GPSR コマンド理解系
4. gpsr_parser_node（= gpsr_command_parser_node）
- 入力
  - /asr/text
  - /asr/confidence（任意）

- 出力
  - /gpsr/intent（JSON, schema: gpsr_intent_v1）

- 役割
  - ASR 出力を GPSR ルールベースで 100% 解析
  - 正規化・参照解決（it / them / last person）
  - 不明瞭時は need_confirm=true を付与

- 内部
  - gpsr_parser.py（100% カバレッジのルールパーサ）
  - gpsr_vocab.py（語彙・補正管理）

⚠️ 重要
ここで intent + steps が完全に確定し，下流は自然言語を一切扱わない

### SMACH ベースのタスク制御
5. gpsr_smach_node
- 入力
  - /gpsr/intent
  - /gpsr/nav_status
  - /gpsr/exec_status

- 出力
  - /gpsr/nav_goal
  - /gpsr/exec_command
  - /gpsr/state

- 役割
  - GPSR タスク全体の状態遷移管理

- 典型フロー
```
WAIT
 → LISTEN
 → PARSE
 → PLAN
 → MOVE
 → SEARCH
 → GRASP
 → DELIVER
 → REPORT
 → WAIT
```

- 特徴
  - intent / steps に従って deterministic に遷移
  - 失敗時は聞き返し・再探索へ分岐可能

### ナビゲーション系
6. hsr_nav_client_node
- 入力
  - /gpsr/nav_goal（部屋名・場所名）

- 出力
  - /gpsr/nav_status

- 内部
  - move_base アクション

- 役割
  -  シンボル（kitchen 等）→ map 座標変換
  -  Nav Stack との I/F を一本化

### HSR 操作系
7. hsr_task_executor_node
- 入力
  - /gpsr/exec_command

- 出力
  - /gpsr/exec_status

- 役割
  - 抽象コマンドを HSR の低レベル操作に分解

- 内部で利用
  - 視線制御
  - 物体検出
  - 把持
  - 音声合成 など

### Bringup / Launch 構成
8. bringup launch（想定）

- gpsr_system.launch
  - audio_capture
  - (audio_rms_monitor)
  - (silero_vad_node)
  - faster_whisper_asr_node
  - gpsr_parser_node
  - gpsr_smach_node
  - hsr_nav_client_node
  - hsr_task_executor_node
- gpsr_debug.launch
  - ASR + Parser のみ
  - Parser 単体
  - 評価・デバッグ用

### 設計思想（アップデート要点）
- 音声認識は自由、理解は厳密
- NLP は parser で完結
- SMACH 以降は自然言語を見ない
- Vision / Nav / Manipulation と完全分離






## 全体アーキテクチャ
```
audio_capture
   |
   v
silero_vad_node
   |  (speech_start / speech_end)
   v
faster_whisper_asr_node
   |  /gpsr/asr/raw_text
   |  /gpsr/asr/text (corrected)
   |  /gpsr/asr/confidence
   |  /gpsr/asr/utterance_end
   v
gpsr_parser_node
   |  /gpsr/intent (JSON)
   v
gpsr_smach_node
```


```

