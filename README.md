# gpsr

## 1. ROS パッケージ構成（例）
```
hsr_gpsr_system/          （メタパッケージ or リポジトリルート）
├── hsr_gpsr_bringup/     … 全体起動用 launch・設定
├── hsr_audio_pipeline/   … audio_capture + Silero VAD + Whisper(faster-whisper)
├── hsr_gpsr_nlp/         … GPSRコマンド解析（NLU）
├── hsr_gpsr_smach/       … GPSRタスク全体の状態遷移(smach)
├── hsr_nav_wrapper/      … nav stack のラッパ（ゴール指示、状態取得）
├── hsr_task_executor/    … 物体操作・対話などの実行インタフェース
└── hsr_msgs/             … 本システム専用のメッセージ/サービス定義（必要なら）
```
## 2. 各ノードの役割
### audio 系
#### 1. audio_capture ノード（既存パッケージ）
- 役割: マイク入力 → /audio (AudioData) に配信
- パラメータでサンプリングレート・フォーマットなどを設定
- Docker コンテナ内の ALSA/PulseAudio とホストをブリッジ

#### 2. silero_vad_node（hsr_audio_pipeline 内）
- 入力: /audio (audio_common_msgs/AudioData)
- 出力: /vad/segments（発話区間の開始・終了、VADフラグ 等）
- 役割:
  - Silero VAD で「今しゃべっているかどうか」を検出
  - Whisper に渡す音声区間を切り出すトリガを出す
  - ノイズの多い環境での余分な音声入力を抑える

#### 3. faster_whisper_asr_node（hsr_audio_pipeline 内）
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
#### 4. gpsr_command_parser_node（hsr_gpsr_nlp 内）
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


```
export PULSE_SERVER=unix:/run/user/1000/pulse/native
docker compose up -d --build
```

```
docker compose exec noetic-audio bash
roslaunch hsr_audio_pipeline gpsr_audio_intent_test.launch
```
```
docker compose exec noetic-audio bash
rostopic echo /gpsr/intent
```
```
docker compose exec noetic-audio bash
rostopic echo /asr/text
```
rosrun hsr_audio_pipeline asr_plain_echo.py

「テーブルの上のペットボトルを持ってきて」

と 2〜3 秒くらい話して、少し黙る、を数回やると、

/asr/text に文字起こしされた日本語文

/gpsr/intent に JSON 例：

{"raw_text": "テーブルの上のペットボトルを持ってきて", 
 "intent_type": "bring",
 "slots": {"object": "テーブルの上のペットボトル", "source": "", "destination": "", "person": ""}}

