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
