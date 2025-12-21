# Version.2
改良版

## 全体アーキテクチャ
### 入力から意図まで
1.Silero VAD（音声区間抽出）
2.faster-whisper（文字起こし）
3.GPSR正規化（語彙置換・ゆらぎ吸収・同義語→正規形）
4.GPSRテンプレパース（generator由来のテンプレ集合でregexマッチ）
5. intent publish（JSONでも独自msgでもOK）

### 音声認識への“強制”のかけ方（ASR側）
- faster-whisper の initial_prompt に 語彙（names/rooms/locations/objects/categories + 典型フレーズ）を列挙
- 文字起こし後に 正規化（verb/前置詞/複数形/服装表現など） をかけてテンプレに寄せる

コマンドジェネレータには verb/prep の同義語集合が明記されているので、それを 逆引き辞書にして canonical へ潰します（例：navigate→go、fetch→get、look for→find など）。

### 語彙は md から読む（gpsr_vocab.py）
Competition Templateを参照

- gpsr_vocab.py（md を読み込んで正規化済み語彙を返す）

### ジェネレータ由来テンプレで“有限パース”する（gpsr_parser.py）
CommandGenerator.generate_command_start() が出し得るコマンド種別は、コード上ほぼ固定です（goToLoc, takeObjFromPlcmt, findPrsInRoom, …）。

なので parser は「汎用NLP」ではなく、このテンプレに100%合わせた正規表現の集合にします。

ここは最初は「例文に強い」状態でOKです。次に CommandGenerator の各 command_string を全部網羅するよう、templateを増やしていきます（有限なので必ず収束します？？？）

### ROSノード化
ASRテキスト（例：/asr/text）を購読して、/gpsr/intent に JSON で出す

### faster_whisper_asr_node.py 側で“テンプレ誘導”する具体策
狙い（「認識すべき文法（テンプレート）をもう少ししっかり処理」）は、次の3点を入れると一気に効く

#### initial_prompt を “語彙 + 典型フレーズ” で固定
- 語彙は md からロード（上の GpsrVocab.all_terms_for_prompt()）
- さらに generator の定型句（tell me how many, answer the quiz of, from the, to the など）を混ぜる
（generatorで実際に出るので根拠がある）

例：
- Find ... in the ... then ...
- Tell me how many ... are in the ...
- Answer the question of the ... in the ...
- Guide NAME from the ... to the ... など

#### ASR後の正規化を “parserと共通化”
- normalize_text() を ASR側でも使う（同義語潰し・persons→people等）
- 誤認識 correction は 語彙に基づく（例：past room→ rooms/locationsにないので bathroom 近傍？…のような“推測”は危険。まずは 辞書置換を厚くする）

#### “語彙外トークン” を落とす（軽い強制）
- rooms/names/locations/objects/categories に一致しないトークンが少数なら削除・置換する
- とくに past room のような roomsに存在しない語が出たら、room スロットは 空にして parser 側で unparsed に落とす方が安全（誤動作よりマシ）

rooms はこの5つしかない、など“有限集合”が明確です。

### 次の一手（ここから先の“収束”のさせ方）
1. gpsr_commands.py（ジェネレータ）にある command type を全部列挙
2. それぞれに対して「正規表現テンプレ」を追加
3. 1000文くらいジェネレータで生成→ASR風ノイズ（小さなゆらぎ）を入れる→parse成功率を測る
4. 足りないテンプレだけ追加

ジェネレータは verb / prep も有限集合で管理されているので、ここも “正規化” で確実に潰せます。


