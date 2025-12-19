#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import rospy
import smach
import smach_ros
from std_msgs.msg import String

# ---- SMACH States ----

class WaitForIntent(smach.State):
    """
    /gpsr/intent から新しいインテントが来るまで待つ状態
    """
    def __init__(self, intent_topic):
        smach.State.__init__(
            self,
            outcomes=['got_intent', 'preempted'],
            output_keys=['intent_json']
        )
        self.intent_topic = intent_topic

    def execute(self, userdata):
        rospy.loginfo("WAIT_INTENT: waiting for intent on %s", self.intent_topic)
        if rospy.is_shutdown():
            return 'preempted'

        try:
            msg = rospy.wait_for_message(self.intent_topic, String, timeout=None)
        except rospy.ROSInterruptException:
            return 'preempted'

        userdata.intent_json = msg.data
        rospy.loginfo("WAIT_INTENT: received intent: %s", msg.data)
        return 'got_intent'


class ParseIntent(smach.State):
    """
    JSON 文字列を Python dict に変換する
    """
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['parsed', 'parse_error'],
            input_keys=['intent_json'],
            output_keys=['intent_dict', 'error_msg']
        )

    def execute(self, userdata):
        try:
            intent = json.loads(userdata.intent_json)
            userdata.intent_dict = intent
            userdata.error_msg = ""
            rospy.loginfo("PARSE_INTENT: parsed intent: %s", intent)
            return 'parsed'
        except Exception as e:
            err = "failed to parse intent JSON: %s" % e
            rospy.logerr("PARSE_INTENT: %s", err)
            userdata.intent_dict = {}
            userdata.error_msg = err
            return 'parse_error'


class PlanTask(smach.State):
    """
    intent_type と slots から「何をするか」の説明テキストを作る
    （この段階ではまだ「実際には動かさない」）
    """
    def __init__(self, task_state_pub):
        smach.State.__init__(
            self,
            outcomes=['planned', 'unsupported'],
            input_keys=['intent_dict', 'error_msg'], 
            output_keys=['task_description', 'error_msg']
        )
        self.task_state_pub = task_state_pub

    def execute(self, userdata):
        intent = userdata.intent_dict or {}
        intent_type = intent.get("intent_type", "other")
        slots = intent.get("slots", {})
        object_ = slots.get("object", "")
        dest = slots.get("destination", "")
        person = slots.get("person", "")

        desc = ""
        if intent_type == "bring":
            # 例: 「テーブルの上のペットボトルを持ってきて」
            desc = "『%s』を探して持ってくるタスクを計画します。" % (object_ or "指定物体")
        elif intent_type == "guide":
            desc = "『%s』へ案内するタスクを計画します。" % (dest or "指定場所")
        elif intent_type == "answer":
            desc = "『%s』という質問に答えるタスクを計画します。" % intent.get("raw_text", "")
        else:
            userdata.error_msg = "unsupported intent_type: %s" % intent_type
            rospy.logwarn("PLAN_TASK: %s", userdata.error_msg)
            userdata.task_description = ""
            # 状態としては「対応していないインテント」
            self.task_state_pub.publish(String(data="UNSUPPORTED: %s" % intent.get("raw_text", "")))
            return 'unsupported'

        userdata.task_description = desc
        userdata.error_msg = ""
        msg = "PLANNED: %s" % desc
        rospy.loginfo("PLAN_TASK: %s", msg)
        self.task_state_pub.publish(String(data=msg))
        return 'planned'


class ExecuteTask(smach.State):
    """
    実行フェーズ（今はダミーでログを出すだけ）
    将来的にここで move_base や HSR のナビ／マニピュレーションを呼ぶ。
    """
    def __init__(self, task_state_pub):
        smach.State.__init__(
            self,
            outcomes=['done', 'failed'],
            input_keys=['intent_dict', 'task_description'],
            output_keys=['execution_result']
        )
        self.task_state_pub = task_state_pub

    def execute(self, userdata):
        intent = userdata.intent_dict or {}
        intent_type = intent.get("intent_type", "other")
        raw_text = intent.get("raw_text", "")

        rospy.loginfo("EXECUTE_TASK: start (%s)", userdata.task_description)
        self.task_state_pub.publish(String(data="EXECUTING: %s" % userdata.task_description))

        # ここで本当はナビやアームのアクションを呼ぶ。
        # 今は「やったつもり」で少し待ってから成功とする。
        rospy.sleep(2.0)

        result_text = "タスク '%s' を仮想的に実行しました。（intent_type=%s, raw_text='%s')" % (
            userdata.task_description, intent_type, raw_text
        )
        rospy.loginfo("EXECUTE_TASK: %s", result_text)
        self.task_state_pub.publish(String(data="DONE: %s" % userdata.task_description))

        userdata.execution_result = result_text
        return 'done'


class ReportResult(smach.State):
    """
    実行結果を /gpsr/task_result に流し、次のインテント待ちへ戻る
    """
    def __init__(self, task_result_pub):
        smach.State.__init__(
            self,
            outcomes=['reported'],
            input_keys=['execution_result', 'error_msg']
        )
        self.task_result_pub = task_result_pub

    def execute(self, userdata):
        if userdata.error_msg:
            text = "ERROR: %s" % userdata.error_msg
        else:
            text = userdata.execution_result or "no execution result"

        rospy.loginfo("REPORT_RESULT: %s", text)
        self.task_result_pub.publish(String(data=text))
        return 'reported'


def main():
    rospy.init_node('gpsr_smach_node')

    intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")

    # 状態表示用トピック
    task_state_pub = rospy.Publisher("/gpsr/task_state", String, queue_size=10)
    task_result_pub = rospy.Publisher("/gpsr/task_result", String, queue_size=10)

    # StateMachine 定義
    sm = smach.StateMachine(outcomes=['preempted'])
    sm.userdata.intent_json = ""
    sm.userdata.intent_dict = {}
    sm.userdata.task_description = ""
    sm.userdata.execution_result = ""
    sm.userdata.error_msg = ""

    with sm:
        smach.StateMachine.add(
            'WAIT_INTENT',
            WaitForIntent(intent_topic),
            transitions={
                'got_intent': 'PARSE_INTENT',
                'preempted': 'preempted'
            }
        )

        smach.StateMachine.add(
            'PARSE_INTENT',
            ParseIntent(),
            transitions={
                'parsed': 'PLAN_TASK',
                'parse_error': 'REPORT_RESULT'
            }
        )

        smach.StateMachine.add(
            'PLAN_TASK',
            PlanTask(task_state_pub),
            transitions={
                'planned': 'EXECUTE_TASK',
                'unsupported': 'REPORT_RESULT'
            }
        )

        smach.StateMachine.add(
            'EXECUTE_TASK',
            ExecuteTask(task_state_pub),
            transitions={
                'done': 'REPORT_RESULT',
                'failed': 'REPORT_RESULT'
            }
        )

        smach.StateMachine.add(
            'REPORT_RESULT',
            ReportResult(task_result_pub),
            transitions={
                'reported': 'WAIT_INTENT'
            }
        )

    # introspection（smach_viewer 用）
    sis = smach_ros.IntrospectionServer('gpsr_smach_introspection', sm, '/GPSR_SM')
    sis.start()

    rospy.loginfo("gpsr_smach_node: started. Waiting for intents on %s", intent_topic)
    outcome = sm.execute()
    rospy.loginfo("gpsr_smach_node: state machine finished with outcome: %s", outcome)
    sis.stop()


if __name__ == '__main__':
    main()
