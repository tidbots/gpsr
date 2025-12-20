# General Purpose Service Robot 汎用サービスロボット
ロボットは、幅広く異なる能力を必要とする命令を理解し実行することが求められる。




## ToDo
### 21 Dec.2025
Whisperのモデルを起動時に毎回ダウンロードしているので無駄<br>
Silero VADでtorch.hubでmaster.zipを毎回ダウンロードしている<br>
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
```
python3 tools/make_vocab_yaml.py \
  names/names.md \
  maps/room_names.md \
  maps/location_names.md \
  objects/objects.md \
  hsr_audio_pipeline/config/vocab.yaml
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

~~**initial_prompt に入れておくと良い単語の微調整**~~
->hot word

~~**誤認識ログの収集と登録**~~
->ツールを作成した


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
rostopic echo /gpsr/intent
```
文字化けしないバージョン
```
rosrun hsr_audio_pipeline gpsr__echo.py
```


## マイクに向かって発話する
命令文はCommandGeneratorで作る<br>
[CommandGenerator](https://github.com/RoboCupAtHome/CommandGenerator)

命令のサンプル
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
「テーブルの上のペットボトルを持ってきて」 と発話すると、

/asr/text に文字起こしされた日本語文が

/gpsr/intent に JSON 例：
```
{"raw_text": "テーブルの上のペットボトルを持ってきて", 
 "intent_type": "bring",
 "slots": {"object": "テーブルの上のペットボトル", "source": "", "destination": "", "person": ""}}
```

## デバッグ
命令文を直接配信する
```
rostopic pub /asr/text std_msgs/String \
"data: 'Find a sponge in the living room then get it and bring it to me'" -1
```
別のターミナルで
```
rostopic echo /gpsr/intent
```

シンプル ASR 動作確認
VADをスキップ
コンテナ内で：
```
cd /hsr_ws
source devel/setup.bash
roslaunch hsr_audio_pipeline audio_asr_simple_test.launch
```


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
┌───────────────────────────────────────────────────────────────┐
│                   Host PC / Docker コンテナ                  │
│                     (ROS Noetic, hsr_ws)                     │
└───────────────────────────────────────────────────────────────┘

      [物理マイク / OS のデフォルト録音デバイス]
                               │
                               │  ALSA / PulseAudio
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                        Audio 入力レイヤ                       │
│                                                               │
│  [audio_capture]  (audio_capture/audio_capture)               │
│      - 入力:  OS のデフォルト録音デバイス                    │
│      - 出力: /audio/audio (audio_common_msgs/AudioData)      │
│              /audio/audio_info (AudioInfo)                    │
└───────────────────────────────────────────────────────────────┘
                               │
                               │ /audio/audio
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                        モニタ・VAD レイヤ                    │
│                                                               │
│  [audio_rms_monitor] (hsr_audio_pipeline/audio_rms_monitor.py)│
│      - 入力: /audio/audio                                     │
│      - ログ出力: RMS, PEAK                                   │
│                                                               │
│  （オプション）[silero_vad_node]                             │
│      - 入力: /audio/audio                                     │
│      - 出力: /vad/is_speech (std_msgs/Bool)                  │
└───────────────────────────────────────────────────────────────┘
                               │
                               │ /audio/audio
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                    音声認識 (ASR) レイヤ                     │
│                                                               │
│  [faster_whisper_asr_simple_node]                             │
│      - 入力: /audio/audio (AudioData)                         │
│      - パラメータ例:                                          │
│          language=en, model_size=tiny.en, segment_sec=8.0     │
│      - 出力: /asr/text (std_msgs/String, 英文)                │
└───────────────────────────────────────────────────────────────┘
                               │
                               │ /asr/text
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                  コマンド解釈 (Intent) レイヤ                │
│                                                               │
│  [gpsr_intent_node]                                           │
│      - 入力: /asr/text (英語コマンド文字列)                  │
│      - 内部:                                                  │
│          - 正規化（全角→半角、空白整理、誤認識補正）        │
│          - ルールベースパース                                 │
│              intent_type: bring / navigate / tell / count…    │
│              slots: {object, source, destination,             │
│                      person, room, attribute}                 │
│      - 出力: /gpsr/intent (std_msgs/String, JSON)            │
│          例: {"raw_text": "...",                              │
│                "normalized_text": "...",                      │
│                "intent_type": "bring",                        │
│                "slots": {...}}                                │
└───────────────────────────────────────────────────────────────┘
                               │
                               │ /gpsr/intent
                               ▼
┌───────────────────────────────────────────────────────────────┐
│                 タスク実行制御 (SMACH) レイヤ                │
│                                                               │
│  [gpsr_smach_node]                                            │
│      - 入力: /gpsr/intent                                     │
│      - SMACH state machine:                                   │
│          WAIT_INTENT → PARSE_INTENT → PLAN_TASK → EXECUTE…    │
│      - 現状: Intent を受けてプランニング部まで               │
│               （ナビゲーション・マニピュレーションは今後）   │
└───────────────────────────────────────────────────────────────┘

```

1. マイク → audio_capture → /audio/audio
生の音声ストリームを ROS トピック化。

2. audio_rms_monitor / silero_vad_node
- RMS/PEAK ログで入力レベルを確認
- Silero VAD で「しゃべっている区間」を検出（必要に応じて）

3. faster_whisper_asr_simple_node
- /audio/audio を連続的に読み取り
- Whisper (tiny.en など) で 英語ロングセンテンス をテキスト化 → /asr/text

4. gpsr_intent_node
- /asr/text を英語コマンドとして正規化＆パース
- ルールベースで intent_type と slots を埋めて /gpsr/intent に JSON で出力

5. gpsr_smach_node
- /gpsr/intent を受けて SMACH の状態遷移
- **今後、HSR のナビゲーション・物体把持・対話などをここから呼び出す予定**





## 1. ROS パッケージ構成
```
gpsr/          （メタパッケージ or リポジトリルート）
├── Dockerfile
├── compose.yaml
└── src
     ├── launch/
          ├── audio_asr_simple_test.launch
          ├── audio_pipeline.launch
          ├── audio_test.launch
          ├── audio_vad_asr_test.launch
          ├── audio_vad_test.launch
          ├── gpsr_audio_intent_test.launch
          └── gpsr_smach_test.launch
     ├── scripts/
          ├── apply_corrections_example.py
          ├── asr_plain_echo.py
          ├── audio_rms_monitor.py
          ├── corrections_candidates.py
          ├── faster_whisper_asr_node.py
          ├── faster_whisper_asr_simple_node.py
          ├── gen_corrections.py
          ├── gpsr_intent_echo.py
          ├── gpsr_intent_node.py
          ├── gpsr_parser.py
          ├── gpsr_parser_node.py
          ├── gpsr_smach_node.py
          ├── gpsr_vocab.py
          └── silero_vad_node.py
     ├── CMakeLists.txt
     └── package.xml

```
## 2. 各ノードの役割
### audio 系
#### 1. audio_capture ノード（既存パッケージ）
- 役割: マイク入力 → /audio (AudioData) に配信
- パラメータでサンプリングレート・フォーマットなどを設定
- Docker コンテナ内の ALSA/PulseAudio とホストをブリッジ

#### 2. silero_vad_node
- 入力: /audio (audio_common_msgs/AudioData)
- 出力: /vad/segments（発話区間の開始・終了、VADフラグ 等）
- 役割:
  - Silero VAD で「今しゃべっているかどうか」を検出
  - Whisper に渡す音声区間を切り出すトリガを出す
  - ノイズの多い環境での余分な音声入力を抑える

#### 3. faster_whisper_asr_node
- 入力:
  - /audio（音声ストリーム）
  - /vad/segments（どの区間を認識するか）
- 出力:
  - /asr/text（最終認識結果の文字列）
  - /asr/partial_text（オプション：部分認識結果）
  - /asr/confidence（オプション：信頼度）
  - /asr/result（文字列＋信頼度＋タイムスタンプ等をまとめた構造体）
- 役割:
  - VADで決まった区間の音声を Whisper で文字起こし
  - GPSR用に扱いやすい認識結果をトピックで配信

### GPSR コマンド理解・計画系
#### 4. gpsr_command_parser_node
- 入力:
  - /asr/text or /asr/result
- 出力:
  - /gpsr/intent（意図ラベル: “bring_object”, “go_to”, “answer_question” など）
  - /gpsr/slots（対象物・場所・人などのスロット情報。geometryやID）
  - /gpsr/plan（オプション: サブタスク列の簡易プラン）
- 役割:
  - 認識された文章をルール or ML-based で解析
  - GPSRルールに沿ったタスク表現（意図＋引数）に変換
  - 不明瞭なときは「聞き返しフラグ」を出すのも可

### smach ベースのタスク制御系
#### 5.gpsr_smach_node（hsr_gpsr_smach 内）
- 入力:
  - /gpsr/intent, /gpsr/slots, /gpsr/plan
  - ナビゲーション状態 (/move_base/status など)
  - HSRの状態（腕・グリッパ・音声合成結果など）
- 出力:
  - ナビゲーション用アクションゴール（/move_base 等）
  - 操作・対話コマンド（後述ノードへサービス/アクション）
  - /gpsr/state（現在のステート、デバッグ用）
- 役割:
  - GPSRタスク全体の状態遷移管理
    - 待機 → 音声認識 → 意図解釈 → プランニング → 移動 → 探索 → 把持 → 配達 → 報告 → 待機
  - 各サブタスクを nav_wrapper / task_executor に振り分け
  - エラー時のリカバリ（聞き返し、再探索など）

### ナビゲーション系ラッパ
#### 6. hsr_nav_client_node（hsr_nav_wrapper 内）
- 入力:
  - /gpsr/nav_goal（GPSR側からの抽象ゴール：“kitchen”, “living_room” など）
- 出力:
  - move_base アクション (/move_base/goal, /move_base/result, /move_base/feedback)
  - /gpsr/nav_status（ナビゲーションの成功/失敗/実行中）
- 役割:
  - 「部屋名」などのシンボル → 実座標 (map 座標) への変換
  - Nav Stack へのアクション送信、結果とエラーのラップ
  - 必要なら costmap・local planner 設定もまとめて管理

### HSR の操作系（例示）
#### 7. hsr_task_executor_node（hsr_task_executor 内）
- 入力:
  - /gpsr/exec_command（「Xを拾ってYへ運べ」などの抽象コマンド）
- 内部で利用:
  - HSR固有のトピック/サービス（腕、頭の向き、グリッパ開閉、音声合成など）
- 出力:
  - /gpsr/exec_status（成功/失敗/途中）
- 役割:
  - 抽象コマンドを HSR の低レベルインタフェースに分解
  - 例: 視線誘導 → 物体検出 → 位置合わせ → 把持 → ナビゲーション呼び出し → 置く

### bringup・ユーティリティ系
#### 8.hsr_gpsr_bringup 内の launch 構成
- gpsr_system.launch（全部を一気に立ち上げる）
  - audio_capture
  - silero_vad_node
  - faster_whisper_asr_node
  - gpsr_command_parser_node
  - gpsr_smach_node
  - hsr_nav_client_node
  - hsr_task_executor_node
  - gpsr_debug.launch（ASR/NLPだけなど部分起動用）

## 3. 使用するトピック・サービス・アクション（例）
### 音声・VAD・ASR
- /audio
  - audio_common_msgs/AudioData
  - publisher: audio_capture
  - subscriber: silero_vad_node, faster_whisper_asr_node（必要に応じて）
- /vad/segments
  - 型（自作）：hsr_msgs/VADSegmentArray など
    - 含まれる情報：segment_id, start_time, end_time, is_speech, energy …
  - pub: silero_vad_node
  - sub: faster_whisper_asr_node, gpsr_smach_node（オプション）
- /asr/text
  - std_msgs/String
  - pub: faster_whisper_asr_node
  - sub: gpsr_command_parser_node, デバッグノード
- /asr/partial_text（任意）
  - std_msgs/String
  - リアルタイム表示・インタラクション用
- /asr/confidence（任意）
  - std_msgs/Float32 など
  - 低信頼度なら聞き返しを促すなど
- /asr/result
  - hsr_msgs/ASRResult（文字列＋信頼度＋タイムスタンプ＋言語情報など）
  - 「NLP側はこのトピックだけ見ればよい」ように抽象化

### GPSR 意図・プラン
- /gpsr/intent
  - hsr_msgs/GPSRIntent（intent_type: enum, raw_text 等）
  - pub: gpsr_command_parser_node
  - sub: gpsr_smach_node
- /gpsr/slots
  - hsr_msgs/GPSRSlots
  - 例: target_object, source_location, destination_location, person など
- /gpsr/plan（任意）
  - hsr_msgs/GPSRPlan（サブタスク列）
- /gpsr/state
  - std_msgs/String or hsr_msgs/GPSRState
  - smachの現在ステート名／ステータス可視化
- /gpsr/nav_goal
  - hsr_msgs/NavGoal（場所ID or 座標）
  - pub: gpsr_smach_node
  - sub: hsr_nav_client_node
- /gpsr/nav_status
  - hsr_msgs/NavStatus
  - pub: hsr_nav_client_node
  - sub: gpsr_smach_node
- /gpsr/exec_command
  - hsr_msgs/ExecCommand（操作コマンド）
  - pub: gpsr_smach_node
  - sub: hsr_task_executor_node
- /gpsr/exec_status
  - hsr_msgs/ExecStatus
  - pub: hsr_task_executor_node
  - sub: gpsr_smach_node

### nav stack（例: move_base）
- アクション：/move_base（move_base_msgs/MoveBaseAction）
  - goal: 目的地の geometry_msgs/PoseStamped
  - feedback/result: ナビゲーション状態
  - クライアント: hsr_nav_client_node
  - サーバ: 既存の nav stack ノード
- トピック：
  - /amcl_pose, /map, /tf などは既存 HSR/nav stack に依存

### HSR操作系（例）
ここは HSR 公式インタフェースに依存しますが、イメージとして：
- サービス：
  - /hsr/head_look_at
  - /hsr/gripper_control
  - /hsr/speak など
- これらを hsr_task_executor_node から呼び出し
- gpsr_smach_node は基本的に高レベルコマンドを投げるだけにする

4. 全体アーキテクチャの図（テキスト図）
```
[マイク]
   |
   v
[audio_capture]  ----->  /audio  -----------------------------+
                                                              |
                                                   +----------+----------+
                                                   |                     |
                                          [silero_vad_node]      [faster_whisper_asr_node]
                                                   |                     |
                                         /vad/segments           /asr/text, /asr/result
                                                   \                     /
                                                    \                   /
                                                     v                 v
                                                 [gpsr_command_parser_node]
                                                     |
                                /gpsr/intent, /gpsr/slots, /gpsr/plan
                                                     |
                                                     v
                                                 [gpsr_smach_node]
                                  +------------------+-------------------+
                                  |                                      |
                           /gpsr/nav_goal                         /gpsr/exec_command
                                  v                                      v
                        [hsr_nav_client_node]                    [hsr_task_executor_node]
                                  |                                      |
                               /move_base action                  HSR固有API/トピック
                                  |                                      |
                             (ナビゲーション)                  (把持・運搬・報告 etc.)


smach ノードが「脳（タスクマネージャ）」で、
audio_pipeline（VAD+Whisper）が「耳」、
nav_wrapper が「足」、
task_executor が「手・口」という分担です。
```










