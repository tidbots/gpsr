#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_smach_node.py (DUMMY executor with QUEUE + COALESCE)

Subscribes:
  - /gpsr/intent_json (std_msgs/String)  # gpsr_intent_v1 JSON string

Publishes:
  - /gpsr/task/status (std_msgs/String)  # JSON status
  - /gpsr/task/event  (std_msgs/String)  # one-line events

Goal:
  End-to-end "ASR -> parser -> executor" wiring using SMACH,
  without robot dependencies (nav/manip are dummy).
  Adds:
    - task queue (no drop when busy)
    - coalesce same text within a window (avoid duplicate intents)
"""

import json
import time
import threading
import queue
from typing import Any, Dict, List

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

    You can force behavior per step by adding:
      step["dummy"] = {"result":"failed"|"timeout"|"succeeded", "sleep_sec": 0.2}
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

        dummy = step.get("dummy", {}) or {}
        forced = str(dummy.get("result", "")).strip().lower()  # succeeded|failed|timeout
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
        if sleep_sec > 0:
            time.sleep(min(sleep_sec, self.step_timeout))

        if (time.time() - t0) > self.step_timeout:
            self.status_pub.status("STEP_TIMEOUT", task_id, step_index=idx, action=action)
            return "timeout"

        if forced == "failed":
            self.status_pub.status("STEP_FAILED", task_id, step_index=idx, action=action)
            return "failed"
        if forced == "timeout":
            self.status_pub.status("STEP_TIMEOUT", task_id, step_index=idx, action=action)
            return "timeout"

        # default success
        self.status_pub.status("STEP_DONE", task_id, step_index=idx, action=action)
        ud.step_index = idx + 1
        return "succeeded"


class DummyExecutorNode:
    def __init__(self):
        # ---- params ----
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent_json")
        self.status_topic = rospy.get_param("~status_topic", "/gpsr/task/status")
        self.event_topic = rospy.get_param("~event_topic", "/gpsr/task/event")

        self.auto_run = bool(rospy.get_param("~auto_run", True))

        # acceptance rules
        self.require_ok = bool(rospy.get_param("~require_ok", True))
        self.require_no_confirm = bool(rospy.get_param("~require_no_confirm", True))

        # dummy exec
        self.step_timeout = float(rospy.get_param("~step_timeout", 5.0))
        self.introspection = bool(rospy.get_param("~introspection", False))

        # queue + coalesce
        self.queue_size = int(rospy.get_param("~queue_size", 10))
        self.coalesce_same_text_sec = float(rospy.get_param("~coalesce_same_text_sec", 1.0))
        self.max_queue_wait_sec = float(rospy.get_param("~max_queue_wait_sec", 120.0))  # drop if too old in queue

        self.status_pub = TaskStatusPub(self.status_topic, self.event_topic)

        self._lock = threading.Lock()
        self._last_task_id = 0

        # queue items: {"intent":..., "enq_time":..., "text_key":...}
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.queue_size)

        self._last_accept_text = ""
        self._last_accept_time = 0.0

        rospy.Subscriber(self.intent_topic, String, self._on_intent, queue_size=50)

        threading.Thread(target=self._worker_loop, daemon=True).start()

        rospy.loginfo(
            "gpsr_smach_node (DUMMY+QUEUE) ready: intent=%s status=%s event=%s queue=%d coalesce=%.2fs",
            self.intent_topic, self.status_topic, self.event_topic,
            self.queue_size, self.coalesce_same_text_sec
        )

    def _next_task_id(self) -> str:
        self._last_task_id += 1
        return f"task-{self._last_task_id:04d}"

    def _intent_text_key(self, intent: Dict[str, Any]) -> str:
        return (intent.get("normalized_text") or intent.get("raw_text") or "").strip().lower()

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

        text_key = self._intent_text_key(intent)
        t = time.time()

        # coalesce duplicates in short time window
        if text_key and text_key == self._last_accept_text and (t - self._last_accept_time) < self.coalesce_same_text_sec:
            self.status_pub.event("[DUMMY] coalesce intent: same text")
            return

        item = {"intent": intent, "enq_time": t, "text_key": text_key}

        try:
            self._q.put_nowait(item)
            self._last_accept_text = text_key
            self._last_accept_time = t
            self.status_pub.event(f"[DUMMY] enqueue intent (qsize={self._q.qsize()}/{self.queue_size})")
        except queue.Full:
            self.status_pub.event("[DUMMY] drop intent: queue full")

    def _worker_loop(self):
        while not rospy.is_shutdown():
            try:
                item = self._q.get(timeout=0.2)
            except Exception:
                continue

            enq_time = float(item.get("enq_time", time.time()))
            if (time.time() - enq_time) > self.max_queue_wait_sec:
                self.status_pub.event("[DUMMY] drop queued intent: too old")
                try:
                    self._q.task_done()
                except Exception:
                    pass
                continue

            intent = item.get("intent") or {}
            task_id = self._next_task_id()

            self._run_task(task_id, intent)

            try:
                self._q.task_done()
            except Exception:
                pass

    def _run_task(self, task_id: str, intent: Dict[str, Any]):
        try:
            self.status_pub.status(
                "TASK_START",
                task_id,
                intent_type=intent.get("intent_type"),
                raw_text=intent.get("raw_text"),
            )

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

            sis = None
            if self.introspection:
                sis = smach_ros.IntrospectionServer("gpsr_smach_introspection", sm, "/GPSR_SM")
                sis.start()

            outcome = sm.execute()

            if self.introspection and sis is not None:
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
            self.status_pub.status("TASK_END", task_id)


def main():
    rospy.init_node("gpsr_smach_node")
    _ = DummyExecutorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
