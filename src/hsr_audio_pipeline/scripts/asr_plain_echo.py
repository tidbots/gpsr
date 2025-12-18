#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

def callback(msg):
    # ここでそのまま日本語を標準出力に出す
    text = msg.data
    print(u"[ASR] {}".format(text))

def main():
    rospy.init_node("asr_plain_echo")
    rospy.Subscriber("/asr/text", String, callback)
    rospy.loginfo("asr_plain_echo: subscribed to /asr/text")
    rospy.spin()

if __name__ == "__main__":
    main()
