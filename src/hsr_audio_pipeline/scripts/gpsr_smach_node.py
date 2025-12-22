#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_smach_node.py (DUMMY executor)

Subscribes:
  - /gpsr/intent_json (std_msgs/String)  # gpsr_intent_v1 JSON string

Publishes:
  - /gpsr/task/status (std_msgs/String)  # JSON status
  - /gpsr/task/event  (std_msgs/String)  # one-line events

Goal:
  End-to-end "ASR -> parser -> executor" wiring using SMACH,
  without robot dependencies (nav/manip are dummy).
"""

import json
import time
import threading
from typing import Any, Dict, List, Optional

import rospy
import smach
import smach_ros

from std_msgs.msg import String


def now_ms() -> int:
    return int(time.time() * 1000)


def jdump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


class TaskStatusPub:
    def __init__(self, topic_status: str, topic_event: str):
        self.pub_status = rospy.Publisher(topic_status, String, queue_size=10)
        self.pub_event = rospy.Publisher(topic_event, String, queue_size=10)

    def status(self, state: str, task_id: str, **kw):
        payload = {"ts_ms": now_ms(), "state": state, "task_id": task_id}
        payload.update(kw)
        self.pub_status.publish(String(jdump(payload)))

    def event(self, msg: str):
        self.pub_event.publish(String(msg))


class DummyStepState(smach.State):
    """
    Executes one step from intent["steps"][i] in a dummy way.
    """
    def __init__(self, status_pub: TaskStatusPub, step_timeout: float):
        super().__init__(
            outcomes=["succeeded", "failed", "timeout"],
            input_keys=["intent", "step_index", "task_id"],
            output_keys=["step_index"],
        )
        self.status_pub = status_pub
        self.step_timeout = step_timeout

    def execute(self, ud):
        intent: Dict[str, Any] = ud.intent
        idx: int = int(ud.step_index)
        task_id: str = str(ud.task_id)

        steps: List[Dict[str, Any]] = intent.get("steps") or []
        if idx >= len(steps):
            return "succeeded"

        step = steps[idx] or {}
        action = step.get("action", "")
        args = step.get("args", {}) or {}

        # --- Dummy behavior controls ---
        # You can force failure by putting:
        #   {"action":"...", "args":{...}, "dummy":{"result":"failed"}}
        dummy = step.get("dummy", {}) or {}
        forced = dummy.get("result", "")  # "succeeded" | "failed" | "timeout"
        sleep_sec = float(dummy.get("sleep_sec", 0.2))

        self.status_pub.status(
            "STEP_START",
            task_id,
            step_index=idx,
            action=action,
            args=args,
        )
        self.status_pub.event(f"[DUMMY] step[{idx}] action={action} args={args}")

        t0 = time.time()
        # pretend doing work
        if sleep_sec > 0:
            time.sleep(min(sleep_sec, self.step_timeout))

        # timeout check
        if (time.time() - t0) > self.step_timeout:
            self.status_pub.status(
                "STEP_TIMEOUT",
                task_id,
                step_index=idx,
                action=action,
            )
            return "timeout"

        # forced result
        if forced == "failed":
            self.status_pub.status("STEP_FAILED", task_id, step_index=idx, action=action)
            return "failed"
        if forced == "timeout":
            self.status_pub.status("STEP_TIMEOUT", task_id, step_index=idx, action=action)
            return "timeout"

        # default: succeed
        self.status_pub.status("STEP_DONE", task_id, step_index=idx, action=action)
        ud.step_index = idx + 1
        return "succeeded"


class DummyExecutorNode:
    def __init__(self):
        # ---- params ----
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent_json")

        self.status_topic = rospy.get_param("~status_topic", "/gpsr/task/status")
        self.event_topic = rospy.get_param("~event_topic", "/gpsr/task/event")

        self.auto_run = bool(rospy.get_param("~auto_run", True))  # if False, ignore intents
        self.drop_if_busy = bool(rospy.get_param("~drop_if_busy", True))
        self.step_timeout = float(rospy.get_param("~step_timeout", 5.0))

        # optional: execute only if ok==true and need_confirm==false
        self.require_ok = bool(rospy.get_param("~require_ok", True))
        self.require_no_confirm = bool(rospy.get_param("~require_no_confirm", True))

        self.status_pub = TaskStatusPub(self.status_topic, self.event_topic)

        self._lock = threading.Lock()
        self._busy = False
        self._last_task_id = 0

        rospy.Subscriber(self.intent_topic, String, self._on_intent, queue_size=10)

        rospy.loginfo(
            "gpsr_smach_node (DUMMY) ready: intent=%s status=%s event=%s auto_run=%s drop_if_busy=%s",
            self.intent_topic, self.status_topic, self.event_topic,
            self.auto_run, self.drop_if_busy
        )

    def _next_task_id(self) -> str:
        self._last_task_id += 1
        return f"task-{self._last_task_id:04d}"

    def _on_intent(self, msg: String):
        if not self.auto_run:
            return

        try:
            intent = json.loads(msg.data)
        except Exception as e:
            self.status_pub.event(f"[DUMMY] invalid JSON on {self.intent_topic}: {e}")
            return

        if self.require_ok and not bool(intent.get("ok", False)):
            self.status_pub.event("[DUMMY] ignore intent: ok=false")
            return
        if self.require_no_confirm and bool(intent.get("need_confirm", False)):
            self.status_pub.event("[DUMMY] ignore intent: need_confirm=true")
            return

        with self._lock:
            if self._busy:
                if self.drop_if_busy:
                    self.status_pub.event("[DUMMY] drop intent: busy")
                    return
                # else: ignore new intents until done (still drop)
                self.status_pub.event("[DUMMY] ignore intent: busy")
                return
            self._busy = True

        task_id = self._next_task_id()
        threading.Thread(target=self._run_task, args=(task_id, intent), daemon=True).start()

    def _run_task(self, task_id: str, intent: Dict[str, Any]):
        try:
            self.status_pub.status("TASK_START", task_id, intent_type=intent.get("intent_type"), raw_text=intent.get("raw_text"))

            # Build a SMACH state machine: loop DummyStepState until done
            sm = smach.StateMachine(outcomes=["SUCCEEDED", "FAILED", "TIMEOUT"])
            sm.userdata.intent = intent
            sm.userdata.step_index = 0
            sm.userdata.task_id = task_id

            with sm:
                smach.StateMachine.add(
                    "STEP",
                    DummyStepState(self.status_pub, self.step_timeout),
                    transitions={
                        "succeeded": "STEP",
                        "failed": "FAILED",
                        "timeout": "TIMEOUT",
                    },
                )

            # Introspection (optional; keep false in competition)
            enable_viewer = bool(rospy.get_param("~introspection", False))
            sis = None
            if enable_viewer:
                sis = smach_ros.IntrospectionServer("gpsr_smach_introspection", sm, "/GPSR_SM")
                sis.start()

            outcome = sm.execute()

            if enable_viewer and sis is not None:
                sis.stop()

            if outcome == "FAILED":
                self.status_pub.status("TASK_FAILED", task_id)
            elif outcome == "TIMEOUT":
                self.status_pub.status("TASK_TIMEOUT", task_id)
            else:
                self.status_pub.status("TASK_SUCCEEDED", task_id)

        except Exception as e:
            self.status_pub.status("TASK_EXCEPTION", task_id, error=str(e))
        finally:
            with self._lock:
                self._busy = False
            self.status_pub.status("TASK_END", task_id)


def main():
    rospy.init_node("gpsr_smach_node")
    _ = DummyExecutorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
