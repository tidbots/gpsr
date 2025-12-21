#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
from dataclasses import dataclass
from typing import Optional

import rospy
from std_msgs.msg import String, Bool, Float32


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class Latest:
    raw_text: str = ""
    text: str = ""
    conf: Optional[float] = None
    t_wall_text: float = 0.0
    t_wall_raw: float = 0.0


class GpsrAsrLogger:
    def __init__(self):
        rospy.init_node("gpsr_asr_logger")

        # topics
        self.raw_text_topic = rospy.get_param("~raw_text_topic", "/gpsr/asr/raw_text")
        self.text_topic = rospy.get_param("~text_topic", "/gpsr/asr/text")
        self.conf_topic = rospy.get_param("~confidence_topic", "/gpsr/asr/confidence")
        self.utt_end_topic = rospy.get_param("~utterance_end_topic", "/gpsr/asr/utterance_end")

        # output
        self.out_dir = rospy.get_param("~out_dir", os.path.expanduser("~/logs/gpsr"))
        self.max_age_sec = float(rospy.get_param("~max_age_sec", 30.0))
        self.flush_every = int(rospy.get_param("~flush_every", 1))

        os.makedirs(self.out_dir, exist_ok=True)
        self.out_path = os.path.join(self.out_dir, f"asr_log_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self._fh = open(self.out_path, "a", encoding="utf-8")

        self.latest = Latest()
        self._count = 0

        # subs
        rospy.Subscriber(self.raw_text_topic, String, self._on_raw_text, queue_size=50)
        rospy.Subscriber(self.text_topic, String, self._on_text, queue_size=50)
        rospy.Subscriber(self.conf_topic, Float32, self._on_conf, queue_size=50)
        rospy.Subscriber(self.utt_end_topic, Bool, self._on_utt_end, queue_size=50)

        rospy.loginfo(
            "gpsr_asr_logger ready: raw=%s text=%s conf=%s end=%s -> %s",
            self.raw_text_topic, self.text_topic, self.conf_topic, self.utt_end_topic, self.out_path
        )

    def _on_raw_text(self, msg: String):
        self.latest.raw_text = (msg.data or "").strip()
        self.latest.t_wall_raw = time.time()

    def _on_text(self, msg: String):
        self.latest.text = (msg.data or "").strip()
        self.latest.t_wall_text = time.time()

    def _on_conf(self, msg: Float32):
        self.latest.conf = float(msg.data)

    def _on_utt_end(self, msg: Bool):
        if not msg.data:
            return

        now = time.time()
        age_text = now - (self.latest.t_wall_text or 0.0)
        age_raw = now - (self.latest.t_wall_raw or 0.0)

        if age_text > self.max_age_sec or not self.latest.text:
            rospy.logwarn(
                "gpsr_asr_logger: utterance_end but no fresh text (age=%.3f, text='%s')",
                age_text, self.latest.text
            )
            return

        rec = {
            "ts": now_iso(),
            "t_wall": now,
            "raw_text": self.latest.raw_text,   # ★追加
            "text": self.latest.text,
            "confidence": self.latest.conf,
            "age_text": age_text,
            "age_raw": age_raw,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._count += 1
        if (self._count % self.flush_every) == 0:
            self._fh.flush()

    def close(self):
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


def main():
    node = GpsrAsrLogger()
    try:
        rospy.spin()
    finally:
        node.close()


if __name__ == "__main__":
    main()
