#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPSR SMACH Orchestrator (ROS1 / Noetic)

Input:
  - /gpsr/intent : std_msgs/String (JSON) with schema "gpsr_intent_v1"
    {
      "schema": "gpsr_intent_v1",
      "ok": bool,
      "need_confirm": bool,
      "intent_type": "bring|guide|answer|other",
      "slots": {...},
      "raw_text": "...",
      "confidence": 0..1 or null,
      "source": "parser|intent",
      "steps": [...],           # optional
      "command_kind": "..."     # optional
    }

Output:
  - /gpsr/task_state : std_msgs/String  (human-readable state)
  - /gpsr/result     : std_msgs/String  (JSON summary)

Features:
  - 7-minute global time limit
  - single mode (1-by-1) or multi mode (collect up to 3 commands)
  - interleaving planner (simple heuristic: group by destination/room and keep order)
  - return to Instruction Point (operator) after each run
  - robust handling of need_confirm / parse_failed

Integration points:
  - Navigation via move_base (optional; can run in "dry_run" mode)
  - Task executors: bring/guide/answer hooks (currently stubs)

"""

import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import rospy
import smach
import smach_ros
from std_msgs.msg import String

# Optional navigation (move_base)
try:
    import actionlib
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    from geometry_msgs.msg import Quaternion
    import tf.transformations as tft
except Exception:
    actionlib = None
    MoveBaseAction = None
    MoveBaseGoal = None
    Quaternion = None
    tft = None


@dataclass
class IntentV1:
    ok: bool
    need_confirm: bool
    intent_type: str
    slots: Dict[str, Any]
    raw_text: str = ""
    confidence: Optional[float] = None
    source: str = "unknown"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    command_kind: Optional[str] = None

    @staticmethod
    def from_json(s: str) -> Optional["IntentV1"]:
        try:
            d = json.loads(s)
            if d.get("schema") != "gpsr_intent_v1":
                return None
            return IntentV1(
                ok=bool(d.get("ok", False)),
                need_confirm=bool(d.get("need_confirm", False)),
                intent_type=str(d.get("intent_type", "other")),
                slots=dict(d.get("slots") or {}),
                raw_text=str(d.get("raw_text") or ""),
                confidence=d.get("confidence", None),
                source=str(d.get("source") or "unknown"),
                steps=list(d.get("steps") or []),
                command_kind=d.get("command_kind", None),
            )
        except Exception:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "gpsr_intent_v1",
            "ok": self.ok,
            "need_confirm": self.need_confirm,
            "intent_type": self.intent_type,
            "slots": self.slots,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "source": self.source,
            "steps": self.steps,
            "command_kind": self.command_kind,
        }


class MoveBaseClient:
    """Thin wrapper for move_base. Works in dry_run mode even if move_base is absent."""

    def __init__(self, action_name: str, timeout_sec: float, dry_run: bool):
        self.action_name = action_name
        self.timeout_sec = timeout_sec
        self.dry_run = dry_run

        self._client = None
        if not dry_run and actionlib and MoveBaseAction:
            self._client = actionlib.SimpleActionClient(action_name, MoveBaseAction)
            rospy.loginfo("move_base: waiting for action server '%s' ...", action_name)
            ok = self._client.wait_for_server(rospy.Duration(10.0))
            if not ok:
                rospy.logwarn("move_base action server not available. Switching to dry_run.")
                self.dry_run = True
                self._client = None

    def goto_pose(self, frame_id: str, x: float, y: float, yaw: float) -> bool:
        if self.dry_run or not self._client:
            rospy.loginfo("[dry_run] NAV to (%s) x=%.3f y=%.3f yaw=%.3f", frame_id, x, y, yaw)
            rospy.sleep(0.5)
            return True

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame_id
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        q = self._yaw_to_quat(yaw)
        goal.target_pose.pose.orientation = q

        self._client.send_goal(goal)
        ok = self._client.wait_for_result(rospy.Duration(self.timeout_sec))
        if not ok:
            rospy.logwarn("move_base timeout (%.1fs). Cancelling goal.", self.timeout_sec)
            self._client.cancel_goal()
            return False

        state = self._client.get_state()
        # actionlib goal statuses: 3=SUCCEEDED
        return state == 3

    def _yaw_to_quat(self, yaw: float):
        if not tft or not Quaternion:
            # fallback "no rotation"
            return Quaternion(0.0, 0.0, 0.0, 1.0)
        q = tft.quaternion_from_euler(0.0, 0.0, yaw)
        return Quaternion(q[0], q[1], q[2], q[3])


class IntentBuffer:
    """Thread-safe buffer for incoming intents."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queue: List[IntentV1] = []
        self._last_received = rospy.Time(0)

    def push(self, it: IntentV1):
        with self._lock:
            self._queue.append(it)
            self._last_received = rospy.Time.now()

    def pop_all(self) -> List[IntentV1]:
        with self._lock:
            items = self._queue[:]
            self._queue = []
            return items

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def last_received_age(self) -> float:
        with self._lock:
            if self._last_received == rospy.Time(0):
                return 1e9
            return (rospy.Time.now() - self._last_received).to_sec()


# -----------------------------
# SMACH States
# -----------------------------
class WaitForCommands(smach.State):
    """
    Collect commands either:
      - single mode: first valid intent triggers immediately
      - multi mode: collect up to max_cmd within collect_window_sec
    """

    def __init__(
        self,
        buf: IntentBuffer,
        mode: str,
        max_cmd: int,
        collect_window_sec: float,
        pub_state,
        global_deadline: rospy.Time,
    ):
        smach.State.__init__(self, outcomes=["got", "timeout", "global_timeout"])
        self.buf = buf
        self.mode = mode
        self.max_cmd = max_cmd
        self.collect_window_sec = collect_window_sec
        self.pub_state = pub_state
        self.global_deadline = global_deadline

    def execute(self, userdata):
        self.pub_state.publish(String("WAIT_FOR_COMMANDS"))
        start = rospy.Time.now()

        while not rospy.is_shutdown():
            if rospy.Time.now() >= self.global_deadline:
                self.pub_state.publish(String("GLOBAL_TIMEOUT"))
                return "global_timeout"

            items = self.buf.pop_all()
            valid = []
            for it in items:
                # ignore malformed or non-ok
                if it.need_confirm:
                    rospy.logwarn("WAIT: need_confirm intent ignored. raw='%s'", it.raw_text)
                    self.pub_state.publish(String("NEED_CONFIRM"))
                    continue
                if not it.ok:
                    rospy.logwarn("WAIT: not-ok intent ignored. raw='%s'", it.raw_text)
                    continue
                valid.append(it)

            # stash any valid into userdata list
            if valid:
                if not hasattr(userdata, "intents"):
                    userdata.intents = []
                userdata.intents.extend(valid)

            # single mode: as soon as we have 1, return
            if self.mode == "single":
                if getattr(userdata, "intents", []):
                    self.pub_state.publish(String("GOT_COMMAND_SINGLE"))
                    return "got"

            # multi mode: stop when max reached OR window passed after first arrival
            if self.mode == "multi":
                n = len(getattr(userdata, "intents", []))
                if n >= self.max_cmd:
                    self.pub_state.publish(String(f"GOT_COMMANDS_{n}"))
                    return "got"

                # If we already received at least one, count window
                if n > 0:
                    if (rospy.Time.now() - start).to_sec() >= self.collect_window_sec:
                        self.pub_state.publish(String(f"GOT_COMMANDS_{n}_WINDOW_END"))
                        return "got"

            # If nobody speaks for a while in single mode, allow timeout to hand control back
            if (rospy.Time.now() - start).to_sec() >= max(self.collect_window_sec, 15.0) and not getattr(userdata, "intents", []):
                self.pub_state.publish(String("WAIT_TIMEOUT_NO_COMMAND"))
                return "timeout"

            rospy.sleep(0.05)


class PlanInterleaved(smach.State):
    """
    Interleaving heuristic:
      - use destination room/location if available
      - group by destination key, preserve original order inside group
      - output planned list into userdata.plan
    """

    def __init__(self, pub_state):
        smach.State.__init__(self, outcomes=["planned", "empty", "fail"])
        self.pub_state = pub_state

    def execute(self, userdata):
        self.pub_state.publish(String("PLAN_INTERLEAVED"))
        intents: List[IntentV1] = getattr(userdata, "intents", []) or []
        if not intents:
            return "empty"

        # derive destination keys
        def dest_key(it: IntentV1) -> str:
            s = it.slots or {}
            # prefer explicit destination room/location
            for k in ["destination_room", "destination", "room", "location", "to"]:
                v = s.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip().lower()
            # if parser provided steps, inspect for room/location field
            for st in it.steps or []:
                f = st.get("fields") or {}
                for k in ["destination_room", "destination", "room", "location", "to"]:
                    v = f.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip().lower()
            return "unknown"

        groups: Dict[str, List[Tuple[int, IntentV1]]] = {}
        for idx, it in enumerate(intents):
            dk = dest_key(it)
            groups.setdefault(dk, []).append((idx, it))

        # sort groups: unknown last, otherwise by first appearance
        group_keys = list(groups.keys())
        group_keys.sort(key=lambda k: (k == "unknown", groups[k][0][0]))

        plan: List[IntentV1] = []
        for k in group_keys:
            # preserve original order inside group
            for _, it in sorted(groups[k], key=lambda p: p[0]):
                plan.append(it)

        userdata.plan = plan
        self.pub_state.publish(String(f"PLANNED_{len(plan)}"))
        return "planned"


class ExecutePlan(smach.State):
    """
    Execute each intent using task hooks.
    Optionally navigate to destinations if provided.
    """

    def __init__(
        self,
        nav: MoveBaseClient,
        operator_pose: Dict[str, Any],
        pub_state,
        per_task_nav: bool,
    ):
        smach.State.__init__(self, outcomes=["done", "fail"])
        self.nav = nav
        self.operator_pose = operator_pose
        self.pub_state = pub_state
        self.per_task_nav = per_task_nav

    def execute(self, userdata):
        plan: List[IntentV1] = getattr(userdata, "plan", []) or getattr(userdata, "intents", []) or []
        if not plan:
            self.pub_state.publish(String("EXEC_EMPTY"))
            return "done"

        self.pub_state.publish(String(f"EXEC_START_{len(plan)}"))

        results = []
        all_ok = True

        for i, it in enumerate(plan, start=1):
            self.pub_state.publish(String(f"EXEC_{i}/{len(plan)}_{it.intent_type}"))
            rospy.loginfo("EXEC: #%d intent_type=%s slots=%s raw='%s'", i, it.intent_type, it.slots, it.raw_text)

            ok = self._execute_one(it)
            results.append({"i": i, "ok": ok, "intent": it.to_dict()})
            all_ok = all_ok and ok

        userdata.exec_results = results
        return "done" if all_ok else "fail"

    def _execute_one(self, it: IntentV1) -> bool:
        # Optional per-task navigation: go to destination if slot exists
        if self.per_task_nav:
            dest = self._extract_destination_pose(it)
            if dest is not None:
                frame, x, y, yaw = dest
                nav_ok = self.nav.goto_pose(frame, x, y, yaw)
                if not nav_ok:
                    rospy.logwarn("EXEC: navigation to destination failed.")
                    return False

        # Task hooks
        if it.intent_type == "bring":
            return self._do_bring(it)
        if it.intent_type == "guide":
            return self._do_guide(it)
        if it.intent_type == "answer":
            return self._do_answer(it)

        # other: accept as "unsupported" but not hard fail, depending on your strategy
        rospy.logwarn("EXEC: intent_type='%s' treated as unsupported -> fail", it.intent_type)
        return False

    def _extract_destination_pose(self, it: IntentV1) -> Optional[Tuple[str, float, float, float]]:
        """
        Very simple mapping: if slots contains x,y,yaw or location named,
        this returns a pose. For real arenas, you should map room/location names to poses.
        """
        s = it.slots or {}
        # direct numeric pose
        try:
            if all(k in s for k in ["x", "y"]):
                x = float(s["x"])
                y = float(s["y"])
                yaw = float(s.get("yaw", 0.0))
                frame = str(s.get("frame", "map"))
                return (frame, x, y, yaw)
        except Exception:
            pass

        # named destination -> map via parameter dictionary (room_pose_map)
        # We'll read mapping from ROS param in this state, if present.
        try:
            dest_name = ""
            for k in ["destination", "room", "location", "to", "destination_room"]:
                v = s.get(k)
                if isinstance(v, str) and v.strip():
                    dest_name = v.strip().lower()
                    break
            if not dest_name:
                return None

            room_map = rospy.get_param("~room_pose_map", {})  # { "kitchen": {"x":..,"y":..,"yaw":..,"frame":"map"} }
            if not isinstance(room_map, dict):
                return None
            if dest_name not in room_map:
                return None
            p = room_map[dest_name]
            frame = str(p.get("frame", "map"))
            x = float(p["x"])
            y = float(p["y"])
            yaw = float(p.get("yaw", 0.0))
            return (frame, x, y, yaw)
        except Exception:
            return None

    # -----------------------------
    # Task hooks (stubs)
    # Replace internals with your HSR APIs
    # -----------------------------
    def _do_bring(self, it: IntentV1) -> bool:
        obj = (it.slots or {}).get("object", "")
        dest = (it.slots or {}).get("destination", "")
        rospy.loginfo("TASK bring: object='%s' destination='%s' (stub)", obj, dest)
        rospy.sleep(0.5)
        return True

    def _do_guide(self, it: IntentV1) -> bool:
        person = (it.slots or {}).get("person", "")
        dest = (it.slots or {}).get("destination", "")
        rospy.loginfo("TASK guide: person='%s' destination='%s' (stub)", person, dest)
        rospy.sleep(0.5)
        return True

    def _do_answer(self, it: IntentV1) -> bool:
        rospy.loginfo("TASK answer: raw='%s' (stub)", it.raw_text)
        rospy.sleep(0.3)
        return True


class ReturnToOperator(smach.State):
    """Return to Instruction Point (operator pose)"""

    def __init__(self, nav: MoveBaseClient, operator_pose: Dict[str, Any], pub_state):
        smach.State.__init__(self, outcomes=["returned", "fail"])
        self.nav = nav
        self.operator_pose = operator_pose
        self.pub_state = pub_state

    def execute(self, userdata):
        self.pub_state.publish(String("RETURN_TO_OPERATOR"))
        frame = str(self.operator_pose.get("frame", "map"))
        x = float(self.operator_pose.get("x", 0.0))
        y = float(self.operator_pose.get("y", 0.0))
        yaw = float(self.operator_pose.get("yaw", 0.0))
        ok = self.nav.goto_pose(frame, x, y, yaw)
        return "returned" if ok else "fail"


class SummarizeAndPublish(smach.State):
    """Publish summary JSON on /gpsr/result"""

    def __init__(self, pub_state, pub_result, global_start: rospy.Time, global_deadline: rospy.Time):
        smach.State.__init__(self, outcomes=["done"])
        self.pub_state = pub_state
        self.pub_result = pub_result
        self.global_start = global_start
        self.global_deadline = global_deadline

    def execute(self, userdata):
        self.pub_state.publish(String("SUMMARIZE"))
        now = rospy.Time.now()

        intents = getattr(userdata, "intents", []) or []
        plan = getattr(userdata, "plan", []) or []
        exec_results = getattr(userdata, "exec_results", []) or []
        reason = getattr(userdata, "finish_reason", "completed")

        payload = {
            "schema": "gpsr_result_v1",
            "finish_reason": reason,
            "t_start": self.global_start.to_sec(),
            "t_end": now.to_sec(),
            "time_limit": (self.global_deadline - self.global_start).to_sec(),
            "intents_received": [it.to_dict() for it in intents],
            "plan": [it.to_dict() for it in plan],
            "exec_results": exec_results,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_result.publish(msg)
        return "done"


# -----------------------------
# Main node
# -----------------------------
class GpsrSmachNode:
    def __init__(self):
        # Params
        self.intent_topic = rospy.get_param("~intent_topic", "/gpsr/intent")
        self.task_state_topic = rospy.get_param("~task_state_topic", "/gpsr/task_state")
        self.result_topic = rospy.get_param("~result_topic", "/gpsr/result")

        self.mode = rospy.get_param("~mode", "multi")  # "single" or "multi"
        self.max_commands = int(rospy.get_param("~max_commands", 3))
        self.collect_window_sec = float(rospy.get_param("~collect_window_sec", 12.0))

        # GPSR global time limit (7 minutes default)
        self.global_time_limit_sec = float(rospy.get_param("~global_time_limit_sec", 7.0 * 60.0))

        # Navigation
        self.dry_run_nav = bool(rospy.get_param("~dry_run_nav", True))
        self.move_base_action = rospy.get_param("~move_base_action", "/move_base")
        self.nav_timeout_sec = float(rospy.get_param("~nav_timeout_sec", 40.0))
        self.per_task_nav = bool(rospy.get_param("~per_task_nav", False))

        # Operator / Instruction Point pose
        self.operator_pose = rospy.get_param(
            "~operator_pose",
            {"frame": "map", "x": 0.0, "y": 0.0, "yaw": 0.0},
        )

        # Publishers
        self.pub_state = rospy.Publisher(self.task_state_topic, String, queue_size=10)
        self.pub_result = rospy.Publisher(self.result_topic, String, queue_size=10)

        # Intent buffer
        self.buf = IntentBuffer()
        rospy.Subscriber(self.intent_topic, String, self._on_intent, queue_size=50)

        # Nav client
        self.nav = MoveBaseClient(self.move_base_action, self.nav_timeout_sec, self.dry_run_nav)

        # Timing
        self.global_start = rospy.Time.now()
        self.global_deadline = self.global_start + rospy.Duration(self.global_time_limit_sec)

        rospy.loginfo(
            "gpsr_smach_node: mode=%s max_commands=%d window=%.1fs time_limit=%.1fs intent=%s",
            self.mode, self.max_commands, self.collect_window_sec, self.global_time_limit_sec, self.intent_topic
        )

        # Build SMACH
        self.sm = self._build_state_machine()

        # Introspection (optional)
        enable_introspection = bool(rospy.get_param("~enable_introspection", True))
        self.intro = None
        if enable_introspection:
            self.intro = smach_ros.IntrospectionServer("gpsr_smach_introspection", self.sm, "/GPSR_SMACH")
            self.intro.start()

    def _on_intent(self, msg: String):
        it = IntentV1.from_json(msg.data)
        if it is None:
            rospy.logwarn("SMACH: received non-v1 intent. ignoring.")
            return
        self.buf.push(it)
        rospy.loginfo("SMACH: intent received ok=%s need_confirm=%s type=%s raw='%s'",
                      it.ok, it.need_confirm, it.intent_type, it.raw_text)

    def _build_state_machine(self):
        sm = smach.StateMachine(outcomes=["DONE"])
        sm.userdata.intents = []
        sm.userdata.plan = []
        sm.userdata.exec_results = []
        sm.userdata.finish_reason = "completed"

        with sm:
            smach.StateMachine.add(
                "WAIT",
                WaitForCommands(
                    buf=self.buf,
                    mode=self.mode,
                    max_cmd=self.max_commands,
                    collect_window_sec=self.collect_window_sec,
                    pub_state=self.pub_state,
                    global_deadline=self.global_deadline,
                ),
                transitions={
                    "got": "PLAN",
                    "timeout": "SUMMARIZE",
                    "global_timeout": "SUMMARIZE",
                },
            )

            smach.StateMachine.add(
                "PLAN",
                PlanInterleaved(pub_state=self.pub_state),
                transitions={
                    "planned": "EXEC",
                    "empty": "SUMMARIZE",
                    "fail": "SUMMARIZE",
                },
            )

            smach.StateMachine.add(
                "EXEC",
                ExecutePlan(
                    nav=self.nav,
                    operator_pose=self.operator_pose,
                    pub_state=self.pub_state,
                    per_task_nav=self.per_task_nav,
                ),
                transitions={
                    "done": "RETURN",
                    "fail": "RETURN",  # still return to operator for GPSR flow
                },
            )

            smach.StateMachine.add(
                "RETURN",
                ReturnToOperator(
                    nav=self.nav,
                    operator_pose=self.operator_pose,
                    pub_state=self.pub_state,
                ),
                transitions={
                    "returned": "SUMMARIZE",
                    "fail": "SUMMARIZE",
                },
            )

            smach.StateMachine.add(
                "SUMMARIZE",
                SummarizeAndPublish(
                    pub_state=self.pub_state,
                    pub_result=self.pub_result,
                    global_start=self.global_start,
                    global_deadline=self.global_deadline,
                ),
                transitions={"done": "DONE"},
            )

        return sm

    def run(self):
        self.pub_state.publish(String("SMACH_START"))
        outcome = self.sm.execute()
        rospy.loginfo("gpsr_smach_node finished outcome=%s", outcome)

        if self.intro:
            self.intro.stop()


def main():
    rospy.init_node("gpsr_smach_node")
    node = GpsrSmachNode()
    node.run()


if __name__ == "__main__":
    main()
