#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_smach_node.py
Executor with:
  - Queue (no drop when busy)
  - Coalesce same text in a short window
  - RobotAPI injection point (DummyRobotAPI now)
  - need_confirm handling:
      ASK_CONFIRM -> WAIT_ANSWER (yes/no) -> execute or reject

Subscribes:
  - /gpsr/intent_json        (std_msgs/String)  gpsr_intent_v1 JSON string
  - /gpsr/confirm/answer    (std_msgs/String)  yes/no answer (e.g., "yes", "no")

Publishes:
  - /gpsr/task/status        (std_msgs/String)  JSON status
  - /gpsr/task/event         (std_msgs/String)  one-line events
  - /gpsr/confirm/question   (std_msgs/String)  question prompt for user
  - /gpsr/confirm/result     (std_msgs/String)  JSON: accepted/rejected/timeout
"""

import json
import time
import threading
import queue
from typing import Any, Dict, List, Optional

import rospy
import smach
import smach_ros

from std_msgs.msg import String


def now_ms() -> int:
    return int(time.time() * 1000)


def jdump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _s(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x).strip() if x is not None else ""


# =========================
# Status / Event publisher
# =========================
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


# =========================
# Confirm channel
# =========================
class ConfirmIO:
    def __init__(self, topic_question: str, topic_answer: str, topic_result: str):
        self.pub_q = rospy.Publisher(topic_question, String, queue_size=10)
        self.pub_result = rospy.Publisher(topic_result, String, queue_size=10)

        self._lock = threading.Lock()
        self._last_answer: Optional[str] = None
        self._last_answer_ts = 0.0

        rospy.Subscriber(topic_answer, String, self._on_answer, queue_size=20)

    def _on_answer(self, msg: String):
        ans = _s(msg.data).lower()
        with self._lock:
            self._last_answer = ans
            self._last_answer_ts = time.time()

    def ask(self, task_id: str, question: str):
        self.pub_q.publish(String(question))
        self.pub_result.publish(String(jdump({
            "ts_ms": now_ms(),
            "task_id": task_id,
            "state": "ASKED",
            "question": question,
        })))

    def wait_yesno(self, task_id: str, timeout_sec: float) -> str:
        """
        Returns: "yes" | "no" | "timeout" | "unknown"
        """
        t0 = time.time()
        # snapshot current timestamp so we only accept newer answers
        with self._lock:
            base_ts = self._last_answer_ts

        while not rospy.is_shutdown():
            if (time.time() - t0) > timeout_sec:
                self.pub_result.publish(String(jdump({
                    "ts_ms": now_ms(),
                    "task_id": task_id,
                    "state": "TIMEOUT",
                })))
                return "timeout"

            with self._lock:
                if self._last_answer_ts > base_ts and self._last_answer:
                    ans = self._last_answer
                else:
                    ans = None

            if ans is None:
                time.sleep(0.05)
                continue

            # normalize yes/no
            if any(w in ans.split() for w in ["yes", "yeah", "yep", "sure", "affirmative", "ok", "okay"]):
                self.pub_result.publish(String(jdump({
                    "ts_ms": now_ms(),
                    "task_id": task_id,
                    "state": "ACCEPTED",
                    "answer": ans,
                })))
                return "yes"

            if any(w in ans.split() for w in ["no", "nope", "negative", "cancel", "stop"]):
                self.pub_result.publish(String(jdump({
                    "ts_ms": now_ms(),
                    "task_id": task_id,
                    "state": "REJECTED",
                    "answer": ans,
                })))
                return "no"

            # got something but not interpretable
            self.pub_result.publish(String(jdump({
                "ts_ms": now_ms(),
                "task_id": task_id,
                "state": "UNKNOWN",
                "answer": ans,
            })))
            return "unknown"


# =========================
# RobotAPI (injection point)
# =========================
class RobotAPI:
    def go_to_room(self, room: str) -> Dict[str, Any]:
        raise NotImplementedError

    def go_to_place(self, place: str) -> Dict[str, Any]:
        raise NotImplementedError

    def find_object_in_room(self, obj_or_cat: str, room: str) -> Dict[str, Any]:
        raise NotImplementedError

    def take_object(self, obj_or_cat: str) -> Dict[str, Any]:
        raise NotImplementedError

    def place_object_on_place(self, place: str, obj_or_cat: str = "") -> Dict[str, Any]:
        raise NotImplementedError

    def deliver_to_operator(self) -> Dict[str, Any]:
        raise NotImplementedError


class DummyRobotAPI(RobotAPI):
    def __init__(self, sleep_sec: float = 0.3):
        self.sleep_sec = float(max(0.0, sleep_sec))

    def _ok(self, **kw) -> Dict[str, Any]:
        if self.sleep_sec > 0:
            time.sleep(self.sleep_sec)
        out = {"ok": True}
        out.update(kw)
        return out

    def go_to_room(self, room: str) -> Dict[str, Any]:
        return self._ok(api="go_to_room", room=room)

    def go_to_place(self, place: str) -> Dict[str, Any]:
        return self._ok(api="go_to_place", place=place)

    def find_object_in_room(self, obj_or_cat: str, room: str) -> Dict[str, Any]:
        return self._ok(api="find_object_in_room", obj_or_cat=obj_or_cat, room=room)

    def take_object(self, obj_or_cat: str) -> Dict[str, Any]:
        return self._ok(api="take_object", obj_or_cat=obj_or_cat)

    def place_object_on_place(self, place: str, obj_or_cat: str = "") -> Dict[str, Any]:
        return self._ok(api="place_object_on_place", place=place, obj_or_cat=obj_or_cat)

    def deliver_to_operator(self) -> Dict[str, Any]:
        return self._ok(api="deliver_to_operator")


# =========================
# Step dispatch helpers
# =========================
def _obj_key(args: Dict[str, Any]) -> str:
    return _s(args.get("object") or args.get("object_category") or args.get("object_or_category"))


def _room_key(args: Dict[str, Any]) -> str:
    return _s(args.get("room") or args.get("source_room") or args.get("destination_room"))


def _place_key(args: Dict[str, Any]) -> str:
    return _s(args.get("place") or args.get("source_place") or args.get("destination_place"))


# =========================
# SMACH states
# =========================
class AskConfirmState(smach.State):
    def __init__(self, status_pub: TaskStatusPub, confirm: ConfirmIO, question_template: str):
        super().__init__(
            outcomes=["asked"],
            input_keys=["intent", "task_id"],
        )
        self.status_pub = status_pub
        self.confirm = confirm
        self.question_template = question_template

    def execute(self, ud):
        intent = ud.intent
        task_id = str(ud.task_id)

        raw = _s(intent.get("raw_text") or intent.get("normalized_text"))
        q = self.question_template.format(raw_text=raw)

        self.status_pub.status("CONFIRM_ASK", task_id, question=q, raw_text=raw)
        self.status_pub.event(f"[CONFIRM] ask: {q}")
        self.confirm.ask(task_id, q)
        return "asked"


class WaitAnswerState(smach.State):
    def __init__(self, status_pub: TaskStatusPub, confirm: ConfirmIO, timeout_sec: float):
        super().__init__(
            outcomes=["yes", "no", "timeout", "unknown"],
            input_keys=["task_id"],
        )
        self.status_pub = status_pub
        self.confirm = confirm
        self.timeout_sec = float(timeout_sec)

    def execute(self, ud):
        task_id = str(ud.task_id)
        self.status_pub.status("CONFIRM_WAIT", task_id, timeout_sec=self.timeout_sec)
        ans = self.confirm.wait_yesno(task_id, self.timeout_sec)
        self.status_pub.status("CONFIRM_GOT", task_id, answer=ans)
        self.status_pub.event(f"[CONFIRM] answer={ans}")
        return ans


class StepState(smach.State):
    def __init__(self, status_pub: TaskStatusPub, step_timeout: float, robot: RobotAPI):
        super().__init__(
            outcomes=["succeeded", "failed", "timeout"],
            input_keys=["intent", "step_index", "task_id"],
            output_keys=["step_index"],
        )
        self.status_pub = status_pub
        self.step_timeout = float(step_timeout)
        self.robot = robot

    def execute(self, ud):
        intent: Dict[str, Any] = ud.intent
        idx: int = int(ud.step_index)
        task_id: str = str(ud.task_id)

        steps: List[Dict[str, Any]] = intent.get("steps") or []
        if idx >= len(steps):
            return "succeeded"

        step = steps[idx] or {}
        action = _s(step.get("action", ""))
        args = step.get("args", {}) or {}

        self.status_pub.status("STEP_START", task_id, step_index=idx, action=action, args=args)
        self.status_pub.event(f"[EXEC] step[{idx}] action={action} args={args}")

        t0 = time.time()
        res: Dict[str, Any]

        try:
            if action == "navigate_to_room":
                res = self.robot.go_to_room(_room_key(args))

            elif action == "navigate_to_place":
                res = self.robot.go_to_place(_place_key(args))

            elif action == "find_object_in_room":
                res = self.robot.find_object_in_room(_obj_key(args), _room_key(args))

            elif action == "take_object":
                res = self.robot.take_object(_obj_key(args))

            elif action == "place_object_on_place":
                res = self.robot.place_object_on_place(_place_key(args), _obj_key(args))

            elif action == "bring_object_to_operator":
                sp = _s(args.get("source_place"))
                obj = _s(args.get("object") or args.get("object_category"))
                r1 = self.robot.go_to_place(sp)
                if not bool(r1.get("ok", False)):
                    res = {"ok": False, "error": "go_to_place failed", "detail": r1}
                else:
                    r2 = self.robot.take_object(obj)
                    if not bool(r2.get("ok", False)):
                        res = {"ok": False, "error": "take_object failed", "detail": r2}
                    else:
                        res = self.robot.deliver_to_operator()

            else:
                res = {"ok": False, "error": f"unknown action: {action}"}

        except Exception as e:
            res = {"ok": False, "error": f"exception: {e}"}

        elapsed = time.time() - t0
        if elapsed > self.step_timeout:
            self.status_pub.status("STEP_TIMEOUT", task_id, step_index=idx, action=action, elapsed_sec=elapsed, result=res)
            return "timeout"

        if not bool(res.get("ok", False)):
            self.status_pub.status("STEP_FAILED", task_id, step_index=idx, action=action, elapsed_sec=elapsed, result=res)
            return "failed"

        self.status_pub.status("STEP_DONE", task_id, step_index=idx, action=action, elapsed_sec=elapsed, result=res)
        ud.step_index = idx + 1
        return "succeeded"


# =========================
# Executor node
# =========================
class GpsrSmachExecutorNode:
    def __init__(self):
        # topics
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent_json")
        self.status_topic = rospy.get_param("~status_topic", "/gpsr/task/status")
        self.event_topic = rospy.get_param("~event_topic", "/gpsr/task/event")

        # confirm topics
        self.confirm_question_topic = rospy.get_param("~confirm_question_topic", "/gpsr/confirm/question")
        self.confirm_answer_topic = rospy.get_param("~confirm_answer_topic", "/gpsr/confirm/answer")
        self.confirm_result_topic = rospy.get_param("~confirm_result_topic", "/gpsr/confirm/result")
        self.confirm_timeout_sec = float(rospy.get_param("~confirm_timeout_sec", 8.0))
        self.confirm_question_template = rospy.get_param(
            "~confirm_question_template",
            "I heard: '{raw_text}'. Should I execute this command? Please say yes or no."
        )

        # acceptance rules
        self.auto_run = bool(rospy.get_param("~auto_run", True))
        self.require_ok = bool(rospy.get_param("~require_ok", True))
        # IMPORTANT: now we *handle* need_confirm, so default is False (do NOT reject)
        self.reject_need_confirm = bool(rospy.get_param("~reject_need_confirm", False))

        # execution
        self.step_timeout = float(rospy.get_param("~step_timeout", 5.0))
        self.introspection = bool(rospy.get_param("~introspection", False))

        # queue + coalesce
        self.queue_size = int(rospy.get_param("~queue_size", 10))
        self.coalesce_same_text_sec = float(rospy.get_param("~coalesce_same_text_sec", 1.0))
        self.max_queue_wait_sec = float(rospy.get_param("~max_queue_wait_sec", 120.0))

        # robot impl selection
        self.robot_impl = _s(rospy.get_param("~robot_impl", "dummy")).lower()
        self.dummy_sleep_sec = float(rospy.get_param("~dummy_sleep_sec", 0.3))

        self.status_pub = TaskStatusPub(self.status_topic, self.event_topic)
        self.confirm = ConfirmIO(self.confirm_question_topic, self.confirm_answer_topic, self.confirm_result_topic)

        self._last_task_id = 0
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.queue_size)
        self._last_accept_text = ""
        self._last_accept_time = 0.0

        # build robot
        if self.robot_impl == "dummy":
            self.robot: RobotAPI = DummyRobotAPI(self.dummy_sleep_sec)
            self.status_pub.event(f"[EXEC] robot_impl=dummy (sleep={self.dummy_sleep_sec})")
        else:
            self.robot = DummyRobotAPI(self.dummy_sleep_sec)
            self.status_pub.event("[EXEC] robot_impl!=dummy not implemented yet -> fallback dummy")

        rospy.Subscriber(self.intent_topic, String, self._on_intent, queue_size=50)
        threading.Thread(target=self._worker_loop, daemon=True).start()

        rospy.loginfo(
            "gpsr_smach_node ready: intent=%s status=%s event=%s queue=%d confirm_timeout=%.1f",
            self.intent_topic, self.status_topic, self.event_topic,
            self.queue_size, self.confirm_timeout_sec
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
            self.status_pub.event(f"[EXEC] invalid JSON on {self.intent_topic}: {e}")
            return

        if self.require_ok and not bool(intent.get("ok", False)):
            self.status_pub.event("[EXEC] ignore intent: ok=false")
            return
        if self.reject_need_confirm and bool(intent.get("need_confirm", False)):
            self.status_pub.event("[EXEC] ignore intent: need_confirm=true (reject_need_confirm=true)")
            return

        text_key = self._intent_text_key(intent)
        t = time.time()

        # coalesce duplicates in short time window
        if text_key and text_key == self._last_accept_text and (t - self._last_accept_time) < self.coalesce_same_text_sec:
            self.status_pub.event("[EXEC] coalesce intent: same text")
            return

        item = {"intent": intent, "enq_time": t, "text_key": text_key}
        try:
            self._q.put_nowait(item)
            self._last_accept_text = text_key
            self._last_accept_time = t
            self.status_pub.event(f"[EXEC] enqueue intent (qsize={self._q.qsize()}/{self.queue_size})")
        except queue.Full:
            self.status_pub.event("[EXEC] drop intent: queue full")

    def _worker_loop(self):
        while not rospy.is_shutdown():
            try:
                item = self._q.get(timeout=0.2)
            except Exception:
                continue

            enq_time = float(item.get("enq_time", time.time()))
            if (time.time() - enq_time) > self.max_queue_wait_sec:
                self.status_pub.event("[EXEC] drop queued intent: too old")
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
                command_kind=intent.get("command_kind"),
                need_confirm=bool(intent.get("need_confirm", False)),
            )

            sm = smach.StateMachine(outcomes=["SUCCEEDED", "FAILED", "TIMEOUT", "REJECTED"])
            sm.userdata.intent = intent
            sm.userdata.step_index = 0
            sm.userdata.task_id = task_id

            need_confirm = bool(intent.get("need_confirm", False))

            with sm:
                if need_confirm:
                    smach.StateMachine.add(
                        "ASK_CONFIRM",
                        AskConfirmState(self.status_pub, self.confirm, self.confirm_question_template),
                        transitions={"asked": "WAIT_ANSWER"},
                    )
                    smach.StateMachine.add(
                        "WAIT_ANSWER",
                        WaitAnswerState(self.status_pub, self.confirm, self.confirm_timeout_sec),
                        transitions={
                            "yes": "STEP",
                            "no": "REJECTED",
                            "timeout": "TIMEOUT",
                            "unknown": "REJECTED",
                        },
                    )

                smach.StateMachine.add(
                    "STEP",
                    StepState(self.status_pub, self.step_timeout, robot=self.robot),
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

            if outcome == "REJECTED":
                self.status_pub.status("TASK_REJECTED", task_id)
            elif outcome == "FAILED":
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
    _ = GpsrSmachExecutorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
