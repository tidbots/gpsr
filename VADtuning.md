# 実環境におけるVADの安定化
- 長文でも途中で切れにくい
- 必ず最後に区切れて finalize される
- 環境音で誤トリガしにくい
- 二重発火しない

## システム構成
- Silero VAD が /vad/is_speech を出す
- ASR は
  - False→True で録音開始（pre_roll を混ぜる）
  - True→False で finalize を予約（post_roll 後に確定）
  - max_segment_sec 超えでも強制 finalize

つまり、競技で事故る典型は：
1. ずっと True（環境音・閾値・hangoverで切れない）
2. 細切れ連発（閾値が厳しくて True/False がチカチカ）
3. 二重 finalize（短時間に2回 True→False が起きる）
4. 長文で max_segment に到達して尻切れ（ただしこれは回避可能）

## Silero VAD の “競技用プリセット”
launchファイルでsilero_vad パラメータを設定

### 屋内・ファンノイズあり
- speech_threshold: 0.55
- silence_threshold: 0.35
- hangover_ms: 250
- chunk_size: 512

意味
- speech_threshold を少し下げて「喋り始め」を取りこぼしにくく
- silence_threshold を少し上げて「喋ってない」を明確に
- hangover を短くして **“必ず切れる”**方向へ
```
<node pkg="hsr_audio_pipeline" type="silero_vad_node.py" name="silero_vad" output="screen">
  <param name="audio_topic" value="/audio"/>
  <param name="vad_topic" value="/vad/is_speech"/>
  <param name="sample_rate" value="16000"/>
  <param name="chunk_size" value="512"/>

  <param name="speech_threshold" value="0.55"/>
  <param name="silence_threshold" value="0.35"/>
  <param name="hangover_ms" value="250"/>
</node>
```

## ASR側で「絶対区切れる」ための2つの保険
VADが多少暴れても、ASR側で事故を減らす

### 強制 finalize の挙動を競技向けに（max_segment_sec）
長文GPSRでは 18秒がギリの場合があるので、まず：
- max_segment_sec: 22.0（推奨）
- post_roll: 0.45（推奨）
- pre_roll: 0.35（推奨）
```
<node pkg="hsr_audio_pipeline" type="faster_whisper_asr_node.py" name="faster_whisper_asr_node" output="screen">
  <param name="audio_topic" value="/audio"/>
  <param name="vad_topic" value="/vad/is_speech"/>

  <param name="pre_roll" value="0.35"/>
  <param name="post_roll" value="0.45"/>
  <param name="max_segment_sec" value="22.0"/>
</node>
```

### finalize デバウンス（2重発火防止）
ASRノードに “finalize後N秒は次のfinalizeを無視” を入れる。

これは競技で効きます（細切れやFalse連打のときに intent が連発しなくなる）。

ASRノードに以下を追加する
- param：~finalize_cooldown_sec（例 0.8）
- publish前に「直近finalize時刻」をチェック

実装済み

### VADが“切れない”ときの決定打（ハード・タイムアウト）
競技では「いつまでも True」が一番怖いので、ASR側にもう1個保険を入れます。

「speech が N 秒続いたら、一度切って出す」

これは既に max_segment_sec でほぼ実現できていますが、
VADがTrueのままで post_roll が機能しない場合があるので、max到達時は：
- self._in_speech = False 相当の処理
- self._pending_finalize = True を確実に立てる

  
