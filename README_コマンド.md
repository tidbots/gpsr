コマンドジェネレータで生成されるであろうすべてのコマンドを逆変換する

「すべてのコマンドタイプ」をカバーする逆変換パーサは、今の gpsr_commands.py のソースから十分設計できます 
（ただし、実装はそれなりに大きくなります）

1. 何を「逆引き」するのか整理
gpsr_commands.py の generate_command_start() を見ると、
全コマンドは次の 23 種類に分類されています。

People 系
goToLoc
findPrsInRoom
meetPrsAtBeac
countPrsInRoom
tellPrsInfoInLoc
talkInfoToGestPrsInRoom
answerToGestPrsInRoom
followNameFromBeacToRoom
guideNameFromBeacToBeac
guidePrsFromBeacToBeac
guideClothPrsFromBeacToBeac
greetClothDscInRm
greetNameInRm
meetNameAtLocThenFindInRm
countClothPrsInRoom
tellPrsInfoAtLocToPrsAtLoc
followPrsAtLoc

Object 系
goToLoc
takeObjFromPlcmt
findObjInRoom
countObjOnPlcmt
tellObjPropOnPlcmt
bringMeObjFromPlcmt
tellCatPropOnPlcmt


各 command に対して、ソースではこういうテンプレートが書いてあります（例）:

if command == "goToLoc":
    command_string = (
        "{goVerb} {toLocPrep} the {loc_room} then "
        + self.generate_command_followup("atLoc", cmd_category, difficulty)
    )

elif command == "findObjInRoom":
    command_string = (
        "{findVerb} {art} {obj_singCat} {inLocPrep} the {room} then "
        + self.generate_command_followup("foundObj", cmd_category, difficulty)
    )

elif command == "countObjOnPlcmt":
    command_string = (
        "{countVerb} {plurCat} there are {onLocPrep} the {plcmtLoc}"
    )
# などなど…


パーサでやりたいことは、このテンプレートから「構造」を逆算することです。
2. 内部表現（AST）を決める

まず「パース結果をどう表すか」を決めます。
ここが決まれば、あとは 各テンプレート → その構造 を詰めていくだけです。

たとえば、共通の構造体（Python 的には dict）をこんな感じに決める：

@dataclass
class GpsrCommand:
    kind: str              # "goToLoc", "findObjInRoom", ...
    steps: list[dict]      # マルチステップ（then/and で分かれる）をリストで
    raw_text: str


steps の中身はコマンド毎に違いますが、典型的には：

goToLoc

{
    "action": "go_to_location",
    "location": "sofa",      # or room
    "room": "living room",   # loc_room の場合
}


findObjInRoom + followup(takeObj→deliverObjToMe)

[
    {
        "action": "find_object",
        "object_category": "cleaning supply",
        "room": "bathroom",
    },
    {
        "action": "pick_up_object",
        "target": "same_as_previous_object",
    },
    {
        "action": "deliver_object",
        "destination": "operator",
    },
]


などのイメージです。

3. テンプレートを「逆引き」する一般的な設計
3.1 事前に持っておく辞書

ジェネレータと同じもの：

ROOMS, LOCATIONS, PLACEMENTS

NAMES

OBJECTS

CAT_SING, CAT_PLUR

gesture_person_list, pose_person_list, など

さらに、プレースホルダごとの “許される表現パターン” を用意：

VERB_PATTERNS = {
    "goVerb":    r"(go|move|navigate)",
    "takeVerb":  r"(take|get|grasp|pick up)",
    "findVerb":  r"(find|locate|look for|search for)",
    "meetVerb":  r"(meet|greet)",
    "countVerb": r"(count|tell me how many)",
    "tellVerb":  r"(tell me|say|describe)",
    "deliverVerb": r"(bring|take|deliver|hand)",
    "talkVerb":  r"(say|tell|talk about|describe)",
    # など、ジェネレータ側の verb_dict に合わせて拡張
}

PREP_PATTERNS = {
    "toLocPrep":    r"(to)",
    "fromLocPrep":  r"(from)",
    "inLocPrep":    r"(in)",
    "onLocPrep":    r"(on)",
    "atLocPrep":    r"(at)",
    "deliverPrep":  r"(to)",
    "talkPrep":     r"(to)",
    "ofPrsPrep":    r"(of|from)",
}


場所や部屋は、リストを使って大きな正規表現にします：

ROOM_PATTERN      = r"(" + "|".join(map(re.escape, ROOMS)) + r")"
LOC_PATTERN       = r"(" + "|".join(map(re.escape, LOCATIONS)) + r")"
PLCMT_PATTERN     = r"(" + "|".join(map(re.escape, PLACEMENTS)) + r")"
NAME_PATTERN      = r"(" + "|".join(map(re.escape, NAMES)) + r")"
OBJ_PATTERN       = r"(" + "|".join(map(re.escape, OBJECTS)) + r")"
CAT_SING_PATTERN  = r"(" + "|".join(map(re.escape, CAT_SING)) + r")"
CAT_PLUR_PATTERN  = r"(" + "|".join(map(re.escape, CAT_PLUR)) + r")"
GESTPERS_PATTERN  = r"(" + "|".join(map(re.escape, gesture_person_list)) + r")"
GESTPERSPL_PATTERN= r"(" + "|".join(map(re.escape, gesture_person_plural_list)) + r")"
POSE_PATTERN      = r"(" + "|".join(map(re.escape, pose_person_list)) + r")"

3.2 テンプレート → 正規表現

考え方：

テンプレート文字列をそのままコピー
例: "{findVerb} {art} {obj_singCat} {inLocPrep} the {room} then "

{findVerb} → (?P<findVerb>VERB_PATTERNS["findVerb"])

{obj_singCat} → (?P<object_category>CAT_SING_PATTERN)

{room} → (?P<room>ROOM_PATTERN)

{art} は「a/an/the」なので無視、または (?:a|an|the)? に変換

固定文字列（"then", "the", 空白）はそのまま re.escape して入れる

こうして「1つのテンプレートから1つの正規表現」を自動生成します。

実装イメージ：

def template_to_regex(template: str) -> str:
    # 波括弧をそのまま pattern にするのではなく、
    # 各プレースホルダを対応するサブパターンに変換
    pattern = re.escape(template)
    # ただし {} だけは escape 解除しておく
    pattern = pattern.replace(r"\{", "{").replace(r"\}", "}")

    # a/an の扱いをゆるめる
    pattern = pattern.replace("{art}", r"(?:a|an|the)?")

    # 動詞プレースホルダ
    for ph, verb_pat in VERB_PATTERNS.items():
        pattern = pattern.replace(
            "{" + ph + "}",
            rf"(?P<{ph}>{verb_pat})",
        )

    # 前置詞
    for ph, prep_pat in PREP_PATTERNS.items():
        pattern = pattern.replace(
            "{" + ph + "}",
            rf"(?P<{ph}>{prep_pat})",
        )

    # 場所系
    pattern = pattern.replace("{room}",     rf"(?P<room>{ROOM_PATTERN})")
    pattern = pattern.replace("{loc_room}", rf"(?P<loc_room>{LOC_PATTERN}|{ROOM_PATTERN})")
    pattern = pattern.replace("{loc}",      rf"(?P<loc>{LOC_PATTERN})")
    pattern = pattern.replace("{plcmtLoc}", rf"(?P<plcmtLoc>{PLCMT_PATTERN})")

    # 人・物・カテゴリ
    pattern = pattern.replace("{name}",           rf"(?P<name>{NAME_PATTERN})")
    pattern = pattern.replace("{obj}",            rf"(?P<object>{OBJ_PATTERN})")
    pattern = pattern.replace("{obj_singCat}",    rf"(?P<object_category>{CAT_SING_PATTERN}|{OBJ_PATTERN})")
    pattern = pattern.replace("{plurCat}",        rf"(?P<object_category_plural>{CAT_PLUR_PATTERN})")
    pattern = pattern.replace("{singCat}",        rf"(?P<object_category_sing>{CAT_SING_PATTERN})")
    pattern = pattern.replace("{gestPers}",       rf"(?P<gesture_person>{GESTPERS_PATTERN})")
    pattern = pattern.replace("{gestPersPlur_posePersPlur}", rf"(?P<person_filter_plural>{GESTPERSPL_PATTERN}|{POSE_PATTERN}s)")
    pattern = pattern.replace("{gestPers_posePers}", rf"(?P<person_filter>{GESTPERS_PATTERN}|{POSE_PATTERN})")

    # ... 他のプレースホルダも同様

    # 空白を緩める
    pattern = re.sub(r"\s+", r"\s+", pattern)

    # 文頭/文末アンカー
    return r"^" + pattern.strip() + r"$"


※ 実際には {inRoom_atLoc} など複合プレースホルダもありますが、
その場合は『部屋 or 置き場所』を一括でマッチさせるなど簡略化して OK です。

4. すべてのコマンドタイプをカバーする疑似コード

上の「テンプレート→正規表現変換」を使えば、
全 23 種類のコマンドを 1 つのテーブルで扱えます。

4.1 コマンド定義テーブル
COMMAND_TEMPLATES = {
    # ---- People/HRI commands ----
    "goToLoc": {
        "template": "{goVerb} {toLocPrep} the {loc_room} then {atLoc_followup}",
        "slots": ["loc_room"],
    },
    "findPrsInRoom": {
        "template": "{findVerb} a {gestPers_posePers} {inLocPrep} the {room} and {foundPers_followup}",
        "slots": ["room", "person_filter"],
    },
    "meetPrsAtBeac": {
        "template": "{meetVerb} {name} {inLocPrep} the {room} and {foundPers_followup}",
        "slots": ["name", "room"],
    },
    "countPrsInRoom": {
        "template": "{countVerb} {gestPersPlur_posePersPlur} are {inLocPrep} the {room}",
        "slots": ["room", "person_filter_plural"],
    },
    "tellPrsInfoInLoc": {
        "template": "{tellVerb} me the {persInfo} of the person {inRoom_atLoc}",
        "slots": ["persInfo", "room_or_loc"],
    },
    "talkInfoToGestPrsInRoom": {
        "template": "{talkVerb} {talk} {talkPrep} the {gestPers} {inLocPrep} the {room}",
        "slots": ["talk", "gesture_person", "room"],
    },
    "answerToGestPrsInRoom": {
        "template": "{answerVerb} the {question} {ofPrsPrep} the {gestPers} {inLocPrep} the {room}",
        "slots": ["question", "gesture_person", "room"],
    },
    "followNameFromBeacToRoom": {
        "template": "{followVerb} {name} {fromLocPrep} the {loc} {toLocPrep} the {room}",
        "slots": ["name", "loc", "room"],
    },
    "guideNameFromBeacToBeac": {
        "template": "{guideVerb} {name} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
        "slots": ["name", "loc", "loc_room"],
    },
    "guidePrsFromBeacToBeac": {
        "template": "{guideVerb} the {gestPers_posePers} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
        "slots": ["person_filter", "loc", "loc_room"],
    },
    "guideClothPrsFromBeacToBeac": {
        "template": "{guideVerb} the person wearing a {colorClothe} {fromLocPrep} the {loc} {toLocPrep} the {loc_room}",
        "slots": ["colorClothe", "loc", "loc_room"],
    },
    "greetClothDscInRm": {
        "template": "{greetVerb} the person wearing {art} {colorClothe} {inLocPrep} the {room} and {foundPers_followup}",
        "slots": ["colorClothe", "room"],
    },
    "greetNameInRm": {
        "template": "{greetVerb} {name} {inLocPrep} the {room} and {foundPers_followup}",
        "slots": ["name", "room"],
    },
    "meetNameAtLocThenFindInRm": {
        "template": "{meetVerb} {name} {atLocPrep} the {loc} then {findVerb} them {inLocPrep} the {room}",
        "slots": ["name", "loc", "room"],
    },
    "countClothPrsInRoom": {
        "template": "{countVerb} people {inLocPrep} the {room} are wearing {colorClothes}",
        "slots": ["room", "colorClothes"],
    },
    "tellPrsInfoAtLocToPrsAtLoc": {
        "template": "{tellVerb} the {persInfo} of the person {atLocPrep} the {loc} to the person {atLocPrep} the {loc2}",
        "slots": ["persInfo", "loc", "loc2"],
    },
    "followPrsAtLoc": {
        "template": "{followVerb} the {gestPers_posePers} {inRoom_atLoc}",
        "slots": ["person_filter", "room_or_loc"],
    },

    # ---- Object manipulation commands ----
    "takeObjFromPlcmt": {
        "template": "{takeVerb} {art} {obj_singCat} {fromLocPrep} the {plcmtLoc} and {hasObj_followup}",
        "slots": ["object_category", "plcmtLoc"],
    },
    "findObjInRoom": {
        "template": "{findVerb} {art} {obj_singCat} {inLocPrep} the {room} then {foundObj_followup}",
        "slots": ["object_category", "room"],
    },
    "countObjOnPlcmt": {
        "template": "{countVerb} {plurCat} there are {onLocPrep} the {plcmtLoc}",
        "slots": ["object_category_plural", "plcmtLoc"],
    },
    "tellObjPropOnPlcmt": {
        "template": "{tellVerb} me what is the {objComp} object {onLocPrep} the {plcmtLoc}",
        "slots": ["objComp", "plcmtLoc"],
    },
    "bringMeObjFromPlcmt": {
        "template": "{bringVerb} me {art} {obj} {fromLocPrep} the {plcmtLoc}",
        "slots": ["object", "plcmtLoc"],
    },
    "tellCatPropOnPlcmt": {
        "template": "{tellVerb} me what is the {objComp} {singCat} {onLocPrep} the {plcmtLoc}",
        "slots": ["objComp", "object_category_sing", "plcmtLoc"],
    },
}


ここで、
{atLoc_followup}, {foundObj_followup}, {hasObj_followup} などは
generate_command_followup() 側のテンプレート（takeObj, deliverObjToMe など）に対応します。

パーサ側では：

"then" や "and" で文を分割

先頭部分を上記 COMMAND_TEMPLATES でマッチ

後半部分については、別の小さなテーブル FOLLOWUP_TEMPLATES で解析

とすれば、全コマンド＋フォローアップを一貫した仕組みで扱えます。

5. 実際のパーサの疑似コード（全タイプ対応）
class GpsrParser:
    def __init__(self, vocab):
        self.vocab = vocab
        self.command_patterns = {
            name: re.compile(template_to_regex(spec["template"]), re.IGNORECASE)
            for name, spec in COMMAND_TEMPLATES.items()
        }
        self.followup_patterns = {
            name: re.compile(template_to_regex(spec["template"]), re.IGNORECASE)
            for name, spec in FOLLOWUP_TEMPLATES.items()
        }

    def parse(self, text: str) -> GpsrCommand | None:
        norm = normalize_text(text)

        # 1) "then" / "and" で緩く分割
        parts = split_into_clauses(norm)  # ["find ... in the bathroom", "get it and bring it to me"]

        steps = []

        # 2) 最初の節：トップレベルコマンドの決定
        first = parts[0]
        top_cmd = self._match_top_level(first)
        if top_cmd is None:
            return None
        steps.append(top_cmd["step"])

        # 3) 残りの節：followup コマンドに割り当て
        for clause in parts[1:]:
            follow = self._match_followup(clause, previous_step=steps[-1])
            if follow is not None:
                steps.extend(follow["steps"])

        return GpsrCommand(kind=top_cmd["kind"], steps=steps, raw_text=text)

    def _match_top_level(self, clause: str) -> dict | None:
        for name, pattern in self.command_patterns.items():
            m = pattern.match(clause)
            if not m:
                continue

            slots = extract_slots_from_match(name, m)
            step_struct = build_step_from_slots(name, slots)
            return {"kind": name, "step": step_struct}
        return None

    def _match_followup(self, clause: str, previous_step: dict) -> dict | None:
        for name, pattern in self.followup_patterns.items():
            m = pattern.match(clause)
            if not m:
                continue
            slots = extract_followup_slots(name, m, previous_step)
            steps = build_followup_steps(name, slots, previous_step)
            return {"kind": name, "steps": steps}
        return None


ここで

extract_slots_from_match は COMMAND_TEMPLATES[name]["slots"] と match.groupdict() を使って
{"room": ..., "object_category": ...} のような dict を作る関数

build_step_from_slots は kind ごとに
「どの slot をどういうフィールド名で返すか」を定義するだけです。

たとえば findObjInRoom の場合：

def build_step_from_slots(kind: str, slots: dict) -> dict:
    if kind == "findObjInRoom":
        return {
            "action": "find_object_in_room",
            "room": slots["room"],
            "object_category": slots["object_category"],
        }
    elif kind == "countObjOnPlcmt":
        return {
            "action": "count_objects_on_place",
            "place": slots["plcmtLoc"],
            "object_category": slots["object_category_plural"],
        }
    # ... 他の kind も同様に

6. まとめ：「全部いける？」への答え

gpsr_commands.py のソースから、

全 23 種類のトップレベルコマンド

フォローアップ用のサブコマンド（takeObj, deliverObjToMe, …）
をすべて把握できるので、

上のような「テンプレート → 正規表現」の変換関数と

COMMAND_TEMPLATES / FOLLOWUP_TEMPLATES のテーブル

を用意すれば、

全タイプの GPSR コマンドを逆引きするパーサ（Whisper 出力 → 構造化コマンド）を設計・実装できます。

実コード化すると数百行クラスになりそうですが、
方針としては 「テンプレートと同じ構造を持つ正規表現を生成して match する」 だけなので、

メンテナンスもしやすく
ジェネレータ側と仕様がズレにくいという良い形になります。



