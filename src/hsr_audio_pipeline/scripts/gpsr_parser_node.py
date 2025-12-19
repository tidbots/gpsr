#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import json
from std_msgs.msg import String

from hsr_audio_pipeline.gpsr_parser import GpsrParser, GpsrCommand


class GpsrParserNode:
    def __init__(self):
        # 語彙は gpsr_commands.py と同じにする
        person_names = [
            "Adel", "Angel", "Axel", "Charlie", "Jane",
            "Jules", "Morgan", "Paris", "Robin", "Simone",
        ]
        locations = [
            "bed", "bedside table", "shelf", "trashbin", "dishwasher",
            "potted plant", "kitchen table", "chairs", "pantry",
            "refrigerator", "sink", "cabinet", "coatrack", "desk",
            "armchair", "desk lamp", "waste basket", "tv stand",
            "storage rack", "lamp", "side tables", "sofa", "bookshelf",
            "entrance", "exit",
        ]
        placements = [
            "bed", "bedside table", "shelf", "dishwasher",
            "kitchen table", "pantry", "refrigerator", "sink",
            "cabinet", "desk", "tv stand", "storage rack",
            "side tables", "sofa", "bookshelf",
        ]
        rooms = ["bedroom", "kitchen", "office", "living room", "bathroom"]
        objects = [
            "juice pack", "cola", "milk", "orange juice", "tropical juice",
            "red wine", "iced tea", "tennis ball", "rubiks cube", "baseball",
            "soccer ball", "dice", "orange", "pear", "peach", "strawberry",
            "apple", "lemon", "banana", "plum", "cornflakes", "pringles",
            "cheezit", "cup", "bowl", "fork", "plate", "knife", "spoon",
            "chocolate jello", "coffee grounds", "mustard", "tomato soup",
            "tuna", "strawberry jello", "spam", "sugar", "cleanser", "sponge",
        ]
        categories = [
            ("drink", "drinks"),
            ("toy", "toys"),
            ("fruit", "fruits"),
            ("snack", "snacks"),
            ("dish", "dishes"),
            ("food", "food"),
            ("cleaning supply", "cleaning supplies"),
        ]
        cat_sing = [c[0] for c in categories]
        cat_plur = [c[1] for c in categories]

        self.parser = GpsrParser(
            person_names=person_names,
            location_names=locations,
            placement_location_names=placements,
            room_names=rooms,
            object_names=objects,
            object_categories_plural=cat_plur,
            object_categories_singular=cat_sing,
        )

        self.pub_intent = rospy.Publisher("/gpsr/intent", String, queue_size=10)
        self.sub_asr = rospy.Subscriber("/asr/text", String, self.asr_callback)

    def asr_callback(self, msg: String):
        text = msg.data
        rospy.loginfo("gpsr_parser_node: received text: %s", text)

        command = self.parser.parse(text)
        if command is None:
            rospy.logwarn("gpsr_parser_node: failed to parse command")
            payload = {
                "ok": False,
                "error": "parse_failed",
                "raw_text": text,
            }
        else:
            payload = {
                "ok": True,
                "command": json.loads(command.to_json()),
            }

        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.pub_intent.publish(out)


def main():
        rospy.init_node("gpsr_parser_node")
        node = GpsrParserNode()
        rospy.loginfo("gpsr_parser_node started, waiting for /asr/text")
        rospy.spin()


if __name__ == "__main__":
        main()
