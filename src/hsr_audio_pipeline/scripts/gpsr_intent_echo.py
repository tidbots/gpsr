#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

def callback(msg):
    # ここでそのまま日本語を標準出力に出す
    text = msg.data
    print(u"[ASR] {}".format(text))

def main():
    rospy.init_node("gpsr_intent_echo")
    rospy.Subscriber("/gpsr/intent_json", String, callback)
    rospy.loginfo("gpsr_intent_echo: subscribed to /gpsr/intent_json")
    rospy.spin()

if __name__ == "__main__":
    main()
