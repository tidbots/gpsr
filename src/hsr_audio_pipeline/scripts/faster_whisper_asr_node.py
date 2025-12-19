#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faster-Whisper ASR ROS1 node with competition-specific initial_prompt.
- Forces English transcription for GPSR.
- Supplies a compact initial_prompt built from the fixed vocabulary.
- Emits plaintext transcript on /asr/text.
"""
import numpy as np
import rospy
from std_msgs.msg import String
try:
    from audio_common_msgs.msg import AudioData
except Exception:
    # Placeholder to avoid import errors when linting without audio_common_msgs
    class AudioData(object):
        _type = "audio_common_msgs/AudioData"
        __slots__ = ["data"]
        def __init__(self): self.data = b""

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

NODE_NAME = "faster_whisper_asr_node"

# === Fixed GPSR vocabulary (from user) ===
NAMES = ['Adel','Angel','Axel','Charlie','Jane','Jules','Morgan','Paris','Robin','Simone']
LOCATIONS = ['bed','bedside table','shelf','trashbin','dishwasher','potted plant','kitchen table','chairs','pantry','refrigerator','sink','cabinet','coatrack','desk','armchair','desk lamp','waste basket','tv stand','storage rack','lamp','side tables','sofa','bookshelf','entrance','exit']
ROOMS = ['bedroom','kitchen','office','living room','bathroom']
OBJECTS = ['juice pack','cola','milk','orange juice','tropical juice','red wine','iced tea','tennis ball','rubiks cube','baseball','soccer ball','dice','orange','pear','peach','strawberry','apple','lemon','banana','plum','cornflakes','pringles','cheezit','cup','bowl','fork','plate','knife','spoon','chocolate jello','coffee grounds','mustard','tomato soup','tuna','strawberry jello','spam','sugar','cleanser','sponge']
CATEGORIES = [('drink','drinks'), ('toy','toys'), ('fruit','fruits'), ('snack','snacks'), ('dish','dishes'), ('food','food'), ('cleaning supply','cleaning supplies')]

def build_initial_prompt():
    names = ", ".join(NAMES)
    locations = ", ".join(LOCATIONS)
    rooms = ", ".join(ROOMS)
    objects = ", ".join(OBJECTS)
    categories_sg = ", ".join(s for s, _ in CATEGORIES)
    categories_pl = ", ".join(p for _, p in CATEGORIES)
    templates = (
        "bring <object> from <source> to me; "
        "fetch <category>; "
        "tell me how many <category> are on <location>; "
        "tell the gesture of the person at <src> to the person at <dst>; "
        "navigate to <location> then look for <object> and give it to the lying person in the bathroom"
    )
    return (
        "Home-robot GPSR commands only. Use these words and spelling. "
        f"Names: {names}. Locations: {locations}. Rooms: {rooms}. "
        f"Objects: {objects}. Categories: {categories_sg}; {categories_pl}. "
        f"Templates: {templates}."
    )

class FasterWhisperASRNode(object):
    def __init__(self):
        rospy.init_node(NODE_NAME)
        self.pub_text = rospy.Publisher("/asr/text", String, queue_size=10)

        # Parameters
        model_size   = rospy.get_param("~model_size", "small")
        device       = rospy.get_param("~device", "auto")              # "cuda" / "cpu" / "auto"
        compute_type = rospy.get_param("~compute_type", "auto")        # "int8" / "float16" / etc.
        self.language    = rospy.get_param("~language", "en")          # GPSRは英語
        self.sample_rate = rospy.get_param("~sample_rate", 16000)
        # ★ ここを /audio/audio に
        self.audio_topic = rospy.get_param("~audio_topic", "/audio/audio")

        self.enable       = rospy.get_param("~enable", True)
        self.vad_filter   = rospy.get_param("~vad_filter", False)
        self.beam_size    = rospy.get_param("~beam_size", 5)
        self.temperature  = rospy.get_param("~temperature", 0.0)
        self.initial_prompt = build_initial_prompt()

        if WhisperModel is None:
            rospy.logwarn("[%s] faster_whisper not installed; node will run but not transcribe.", NODE_NAME)
            self.model = None
        else:
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                rospy.loginfo("[%s] Loaded faster-whisper model=%s device=%s compute_type=%s",
                              NODE_NAME, model_size, device, compute_type)
            except Exception as e:
                rospy.logerr("[%s] Failed to load WhisperModel: %s", NODE_NAME, repr(e))
                self.model = None

        rospy.Subscriber(self.audio_topic, AudioData, self.on_audio, queue_size=10)
        rospy.loginfo("[%s] Ready. Subscribed to %s, publishing to /asr/text", NODE_NAME, self.audio_topic)

    def on_audio(self, msg: AudioData):
        if not self.enable or self.model is None:
            return
        try:
            # audio_common_msgs/AudioData: mono 16-bit PCM bytes at sample_rate
            pcm = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
            if pcm.size == 0:
                return
            segments, info = self.model.transcribe(
                pcm,
                language=self.language,
                beam_size=self.beam_size,
                temperature=self.temperature,
                initial_prompt=self.initial_prompt,
                vad_filter=self.vad_filter,
                without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", "").strip())
            if text:
                self.pub_text.publish(String(data=text))
        except Exception as e:
            rospy.logwarn("[%s] Transcribe failed: %s", NODE_NAME, repr(e))

if __name__ == "__main__":
    try:
        node = FasterWhisperASRNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
