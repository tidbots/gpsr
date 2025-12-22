#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gpsr_smach_node.py
DUMMY executor with:
  - Queue (no drop when busy)
  - Coalesce same text in a short window (avoid duplicates)
  - RobotAPI "injection point" for easy real robot swap later

Subscribes:
  - /gpsr/intent_json (std_msgs/String)  # gpsr_intent_v1 JSON string

Publishes:
  - /gpsr/task/status (std_msgs/String)  # JSON status
  - /gpsr/task/event  (std_msgs/String)  # one-line events
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


# =========================
# Status publisher
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
# RobotAPI (injection point)
# =========================
class RobotAPI:
    """Interface to be implemented by DummyRobotAPI now, and HSRRobotAPI later."""
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

    # reserved for later (step-2 you requested):
    # def ask_yesno(self, question: str) -> bool:
    # def ask_choice(self, question: str, choices: List[str]) -> str:


class DummyRobotAPI(RobotAPI):
    """Dummy implementation that sleeps and always succeeds."""
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
def _s(x: Any) -> str:
    return (x or "").strip() if isinstance(x, str) else str(x).strip() if x is not None else ""


def _obj_key(args: Dict[str, Any]) -> str:
    return _s(args.get("object") or args.get("object_category") or args.get("object_or_category"))


def _room_key(args: Dict[str, Any]) -> str:
    return _s(args.get("room") or args.get("source_room") or args.get("destination_room"))


def _place_key(args: Dict[str, Any]) -> str:
    return _s(args.get("place") or args.get("source_place") or args.get("destination_place"))


# =========================
# SMACH state
# =========================
class StepState(smach.State):
    """
    Executes one step from intent["steps"][i] by dispatching to RobotAPI.

    You can force behavior per step by adding:
      step["dummy"] = {"result":"failed"|"timeout"|"succeeded", "sleep_sec": 0.2}
    (dummy.* only affects DummyRobotAPI mode, but kept for debugging.)
    """
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

        # Debug controls (optional)
        dummy = step.get("dummy", {}) or {}
        forced = _s(dummy.get("result", "")).lower()  # succeeded|failed|timeout
        dummy_sleep = dummy.get("sleep_sec", None)

        self.status_pub.status(
            "STEP_START",
            task_id,
            step_index=idx,
            action=action,
            args=args,
        )
        self.status_pub.event(f"[EXEC] step[{idx}] action={action} args={args}")

        # Allow per-step dummy sleep override in DummyRobotAPI
        if isinstance(self.robot, DummyRobotAPI) and dummy_sleep is not None:
            self.robot.sleep_sec = float(max(0.0, dummy_sleep))

        # Timeout enforcement (simple wall clock)
        t0 = time.time()

        # Forced result (debug)
        if forced in ("failed", "timeout"):
            if forced == "failed":
                self.status_pub.status("STEP_FAILED", task_id, step_index=idx, action=action, forced=True)
                return "failed"
            self.status_pub.status("STEP_TIMEOUT", task_id, step_index=idx, action=action, forced=True)
            return "timeout"

        # --- Dispatch ---
        res: Dict[str, Any]
        try:
            if action == "navigate_to_room":
                room = _room_key(args)
                res = self.robot.go_to_room(room)

            elif action == "navigate_to_place":
                place = _place_key(args)
                res = self.robot.go_to_place(place)

            elif action == "find_object_in_room":
                obj = _obj_key(args)
                room = _room_key(args)
                res = self.robot.find_object_in_room(obj, room)

            elif action == "take_object":
                obj = _obj_key(args)
                res = self.robot.take_object(obj)

            elif action == "place_object_on_place":
                place = _place_key(args)
                obj = _obj_key(args)
                res = self.robot.place_object_on_place(place, obj)

            elif action == "bring_object_to_operator":
                # Composite inside one step:
                # go_to_place(source_place) -> take_object(object/category) -> deliver_to_operator()
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
                # Unknown action: fail so you notice early
                res = {"ok": False, "error": f"unknown action: {action}"}

        except Exception as e:
            res = {"ok": False, "error": f"exception: {e}"}

        elapsed = time.time() - t0
        if elapsed > self.step_timeout:
            self.status_pub.status(
                "STEP_TIMEOUT",
                task_id,
                step_index=idx,
                action=action,
                elapsed_sec=elapsed,
                result=res,
            )
            return "timeout"

        if not bool(res.get("ok", False)):
            self.status_pub.status(
                "STEP_FAILED",
                task_id,
                step_index=idx,
                action=action,
                elapsed_sec=elapsed,
                result=res,
            )
            return "failed"

        self.status_pub.status(
            "STEP_DONE",
            task_id,
            step_index=idx,
            action=action,
            elapsed_sec=elapsed,
            result=res,
        )
        ud.step_index = idx + 1
        return "succeeded"


# =========================
# Executor node
# =========================
class GpsrSmachExecutorNode:
    def __init__(self):
        # ---- topics ----
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent_json")
        self.status_topic = rospy.get_param("~status_topic", "/gpsr/task/status")
        self.event_topic = rospy.get_param("~event_topic", "/gpsr/task/event")

        # ---- acceptance rules ----
        self.auto_run = bool(rospy.get_param("~auto_run", True))
        self.require_ok = bool(rospy.get_param("~require_ok", True))
        self.require_no_confirm = bool(rospy.get_param("~require_no_confirm", True))

        # ---- execution ----
        self.step_timeout = float(rospy.get_param("~step_timeout", 5.0))
        self.introspection = bool(rospy.get_param("~introspection", False))

        # ---- queue + coalesce ----
        self.queue_size = int(rospy.get_param("~queue_size", 10))
        self.coalesce_same_text_sec = float(rospy.get_param("~coalesce_same_text_sec", 1.0))
        self.max_queue_wait_sec = float(rospy.get_param("~max_queue_wait_sec", 120.0))

        # ---- robot impl selection ----
        self.robot_impl = _s(rospy.get_param("~robot_impl", "dummy")).lower()  # dummy | hsr (later)
        self.dummy_sleep_sec = float(rospy.get_param("~dummy_sleep_sec", 0.3))

        self.status_pub = TaskStatusPub(self.status_topic, self.event_topic)

        self._last_task_id = 0
        self._lock = threading.Lock()

        # queue items: {"intent":..., "enq_time":..., "text_key":...}
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.queue_size)
        self._last_accept_text = ""
        self._last_accept_time = 0.0

        # build robot
        if self.robot_impl == "dummy":
            self.robot: RobotAPI = DummyRobotAPI(self.dummy_sleep_sec)
            self.status_pub.event(f"[EXEC] robot_impl=dummy (sleep={self.dummy_sleep_sec})")
        else:
            # Placeholder for later real robot implementation
            self.robot = DummyRobotAPI(self.dummy_sleep_sec)
            self.status_pub.event("[EXEC] robot_impl!=dummy not implemented yet -> fallback dummy")

        rospy.Subscriber(self.intent_topic, String, self._on_intent, queue_size=50)
        threading.Thread(target=self._worker_loop, daemon=True).start()

        rospy.loginfo(
            "gpsr_smach_node ready: intent=%s status=%s event=%s queue=%d coalesce=%.2fs robot=%s",
            self.intent_topic, self.status_topic, self.event_topic,
            self.queue_size, self.coalesce_same_text_sec, self.robot_impl
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
        if self.require_no_confirm and bool(intent.get("need_confirm", False)):
            self.status_pub.event("[EXEC] ignore intent: need_confirm=true")
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
            )

            sm = smach.StateMachine(outcomes=["SUCCEEDED", "FAILED", "TIMEOUT"])
            sm.userdata.intent = intent
            sm.userdata.step_index = 0
            sm.userdata.task_id = task_id

            with sm:
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
    _ = GpsrSmachExecutorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
