#!/usr/bin/env python3
import json
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

from action.action_config import CONFIG
from action.scene_reader import parse_scene_message
from .destination_resolver import resolve_destination
from .place_planner import plan_place_pose
from .shelf_config import default_shelf_approach_pose, default_shelf_id
from .task_context import TaskContext, TaskObject
from .task_parser import parse_official_task_commands, parse_task_command


class TaskState:
    WAIT_TASK = "WAIT_TASK"
    SAVE_HOME_POSE = "SAVE_HOME_POSE"
    GROUND_REFERENCES = "GROUND_REFERENCES"
    RESOLVE_DESTINATION = "RESOLVE_DESTINATION"
    WAIT_PICK_READY = "WAIT_PICK_READY"
    START_PICK = "START_PICK"
    WAIT_PICK_DONE = "WAIT_PICK_DONE"
    NAVIGATE_TO_REFERENCE_AREA = "NAVIGATE_TO_REFERENCE_AREA"
    GROUND_PLACE_REFERENCE = "GROUND_PLACE_REFERENCE"
    NAVIGATE_TO_PLACE = "NAVIGATE_TO_PLACE"
    SELECT_PLACE_POSE = "SELECT_PLACE_POSE"
    START_PLACE = "START_PLACE"
    WAIT_PLACE_DONE = "WAIT_PLACE_DONE"
    RETURN_HOME = "RETURN_HOME"
    RESET_ROBOT = "RESET_ROBOT"
    WAIT_RESET_DONE = "WAIT_RESET_DONE"
    TASK_DONE = "TASK_DONE"
    SAFE_STOP = "SAFE_STOP"


class TaskExecutorNode(Node):
    def __init__(self):
        super().__init__("task_executor_node")
        self.config = CONFIG
        self.task_ctx = None
        self.parsed_task = None
        self.latest_scene = None
        self.latest_ready = None
        self.latest_ready_rx = 0.0
        self.latest_grasp_status = None
        self.latest_nav_status = None
        self.latest_place_status = None
        self.robot_pose = None
        self.pending_nav_goal_id = None
        self.pending_nav_purpose = None
        self.pending_nav_pose = None
        self.last_navigation_goal_time = 0.0
        self.last_command_time = 0.0
        self.last_reference_query_time = 0.0
        self.last_pick_query_time = 0.0
        self.reset_command_time = 0.0
        self.last_official_instruction = None
        self.last_official_instruction_rx = 0.0
        self.official_task_queue = []
        self.official_completed_task_ids = set()
        self.official_batch_signature = None
        self.initial_object_memory = {}
        self.task_done_reported = False

        self.create_subscription(String, self.config.task_command_topic, self.on_task_command, 10)
        self.create_subscription(
            String,
            self.config.official_instruction_topic,
            self.on_official_instruction,
            10,
        )
        self.create_subscription(String, self.config.scene_topic, self.on_scene, 10)
        self.create_subscription(String, self.config.ready_topic, self.on_ready, 10)
        self.create_subscription(String, self.config.grasp_status_topic, self.on_grasp_status, 10)
        self.create_subscription(String, self.config.navigation_status_topic, self.on_navigation_status, 10)
        self.create_subscription(String, self.config.place_status_topic, self.on_place_status, 10)
        self.create_subscription(Odometry, self.config.odom_topic, self.on_odom, 10)

        self.task_status_pub = self.create_publisher(String, self.config.task_status_topic, 10)
        self.grasp_command_pub = self.create_publisher(String, self.config.grasp_command_topic, 10)
        self.navigation_goal_pub = self.create_publisher(String, self.config.navigation_goal_topic, 10)
        self.pick_goal_pub = self.create_publisher(String, self.config.task_pick_goal_topic, 10)
        self.place_command_pub = self.create_publisher(String, self.config.place_command_topic, 10)
        self.reference_query_pub = self.create_publisher(
            String,
            self.config.task_reference_query_topic,
            10,
        )

        self.create_timer(0.2, self.tick)
        self.get_logger().info("task executor started")

    def on_task_command(self, message):
        self.official_task_queue = []
        self.start_task_from_instruction(message.data, source="manual", force=True)

    def on_official_instruction(self, message):
        instruction = str(message.data).strip()
        if not instruction:
            return
        if self.task_is_active():
            self.publish_task_status("official_instruction_ignored_task_running")
            return

        tasks = parse_official_task_commands(instruction)
        signature = self.official_signature(tasks)
        if signature != self.official_batch_signature:
            self.official_batch_signature = signature
            self.official_completed_task_ids = set()
            self.initial_object_memory = {}

        pending = [
            task
            for task in tasks
            if str(task.get("task_id")) not in self.official_completed_task_ids
        ]
        if not pending:
            if self.task_ctx is not None:
                self.publish_task_status("official_instruction_ignored_all_tasks_done")
            return

        now = time.time()
        if (
            instruction == self.last_official_instruction
            and now - self.last_official_instruction_rx < 2.0
            and not self.official_task_queue
        ):
            return
        self.last_official_instruction = instruction
        self.last_official_instruction_rx = now
        self.official_task_queue = list(pending[1:])
        self.start_parsed_task(pending[0], source="official")

    def task_is_active(self):
        if self.task_ctx is None:
            return False
        return self.task_ctx.state in {
            TaskState.SAVE_HOME_POSE,
            TaskState.GROUND_REFERENCES,
            TaskState.RESOLVE_DESTINATION,
            TaskState.WAIT_PICK_READY,
            TaskState.START_PICK,
            TaskState.WAIT_PICK_DONE,
            TaskState.NAVIGATE_TO_REFERENCE_AREA,
            TaskState.GROUND_PLACE_REFERENCE,
            TaskState.NAVIGATE_TO_PLACE,
            TaskState.SELECT_PLACE_POSE,
            TaskState.START_PLACE,
            TaskState.WAIT_PLACE_DONE,
            TaskState.RETURN_HOME,
            TaskState.RESET_ROBOT,
            TaskState.WAIT_RESET_DONE,
            TaskState.TASK_DONE,
            TaskState.SAFE_STOP,
        }

    def official_signature(self, tasks):
        return json.dumps(
            [
                {
                    "task_id": task.get("task_id"),
                    "instruction": task.get("instruction"),
                }
                for task in tasks
            ],
            ensure_ascii=False,
            sort_keys=True,
        )

    def start_task_from_instruction(self, message_data, source, force=False):
        if not force and self.task_ctx is not None and self.task_ctx.state not in (
            TaskState.TASK_DONE,
            TaskState.SAFE_STOP,
        ):
            self.publish_task_status(f"{source}_instruction_ignored_task_running")
            return
        self.parsed_task = parse_task_command(message_data)
        self.start_parsed_task(self.parsed_task, source=source)

    def start_parsed_task(self, parsed_task, source):
        self.parsed_task = parsed_task
        self.task_ctx = TaskContext(
            task_id=self.parsed_task["task_id"],
            raw_instruction=self.parsed_task["instruction"],
            state=TaskState.SAVE_HOME_POSE,
        )
        self.task_ctx.destination.place_relation = self.parsed_task.get("place_relation")
        self.task_ctx.destination.place_type = self.parsed_task.get("place_type")
        self.task_ctx.destination.direction = self.parsed_task.get("direction")
        if self.parsed_task.get("place_world"):
            self.task_ctx.destination.place_pose = self.parsed_task.get("place_world")
        # 新任务不能继承上一轮的 ready / done / arrived 状态。
        self.latest_ready = None
        self.latest_ready_rx = 0.0
        self.latest_grasp_status = None
        self.latest_nav_status = None
        self.latest_place_status = None
        self.pending_nav_goal_id = None
        self.pending_nav_purpose = None
        self.pending_nav_pose = None
        self.last_navigation_goal_time = 0.0
        self.reset_command_time = 0.0
        self.last_pick_query_time = 0.0
        self.last_reference_query_time = 0.0
        self.task_done_reported = False
        self.publish_task_status(f"task_received_from_{source}")

    def on_scene(self, message):
        try:
            self.latest_scene = parse_scene_message(message.data)
            self.remember_initial_objects(self.latest_scene)
        except Exception as exc:
            self.get_logger().warning(f"scene parse failed: {exc}")

    def on_ready(self, message):
        try:
            self.latest_ready = json.loads(message.data)
            self.latest_ready_rx = time.time()
        except json.JSONDecodeError:
            pass

    def on_grasp_status(self, message):
        try:
            self.latest_grasp_status = json.loads(message.data)
        except json.JSONDecodeError:
            pass

    def on_navigation_status(self, message):
        try:
            self.latest_nav_status = json.loads(message.data)
        except json.JSONDecodeError:
            pass

    def on_place_status(self, message):
        try:
            self.latest_place_status = json.loads(message.data)
        except json.JSONDecodeError:
            pass

    def on_odom(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        yaw = Rotation.from_quat(quat).as_euler("xyz")[2]
        self.robot_pose = {
            "x": float(position.x),
            "y": float(position.y),
            "yaw": float(yaw),
        }

    def set_state(self, state, reason=""):
        if self.task_ctx is None:
            return
        if self.task_ctx.state != state:
            self.get_logger().info(f"task state: {self.task_ctx.state} -> {state}")
        self.task_ctx.state = state
        self.task_ctx.reason = reason
        self.publish_task_status(reason or "state_changed")

    def publish_json(self, publisher, payload):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def publish_task_status(self, event):
        if self.task_ctx is None:
            return
        payload = self.task_ctx.to_dict()
        payload["event"] = event
        payload["time"] = time.time()
        payload["pending_nav_goal_id"] = self.pending_nav_goal_id
        payload["latest_navigation_status"] = self.latest_nav_status
        payload["official_queue_remaining"] = len(self.official_task_queue)
        payload["official_completed_task_ids"] = sorted(self.official_completed_task_ids)
        self.publish_json(self.task_status_pub, payload)

    def label_parts(self, label):
        text = str(label or "").lower()
        color = None
        category = None

        for candidate in ("pink", "brown", "yellow", "white"):
            if candidate in text:
                color = candidate
                break

        if any(word in text for word in ("box", "cube", "cuboid", "block")):
            category = "box"
        elif "cylinder" in text:
            category = "cylinder"

        return color, category

    def pick_source_surface(self):
        """Return the surface where the current object must be picked."""
        target = self.task_ctx.pick_target if self.task_ctx is not None else None
        target = target or TaskObject()
        surface = str(
            getattr(target, "support_surface", None)
            or getattr(target, "source_location", None)
            or ""
        ).lower()
        if surface in ("table", "table_front", "table_candidate"):
            return "table"
        if surface in ("shelf", "shelf_front", "shelf_candidate"):
            return "shelf"

        # Official instructions often describe only the object and the
        # destination. Prefer the latest semantic target binding when it has
        # already identified the source surface.
        scene = self.latest_scene or {}
        active_task = (scene.get("task_queue") or [{}])[0] or {}
        scene_target = active_task.get("target") or {}
        scene_surface = str(
            scene_target.get("support_surface")
            or scene_target.get("source_location")
            or ""
        ).lower()
        if scene_surface in ("table", "shelf"):
            return scene_surface

        instruction = str(self.task_ctx.raw_instruction if self.task_ctx else "")
        pickup_clause = instruction.split("放到", 1)[0].lower()
        if "货架" in pickup_clause or "shelf" in pickup_clause:
            return "shelf"
        if "桌" in pickup_clause or "table" in pickup_clause:
            return "table"
        return None

    def pick_grounding_prompt(self, label):
        """Give GroundingDINO contrast classes so a single query word does not
        force every detected box to receive the requested color label.
        """
        color, category = self.label_parts(label)
        if category == "box":
            labels = [
                "pink box",
                "brown box",
                "yellow box",
                "white cuboid",
                "white cube",
            ]
        elif category == "cylinder":
            labels = [
                "white cylinder",
                "pink box",
                "brown box",
                "yellow box",
                "white cuboid",
                "white cube",
            ]
        else:
            labels = [label]

        # Keep the requested target first, while removing duplicates and
        # preserving the stable prompt vocabulary used by the vision node.
        ordered = [label] + labels
        unique = []
        for item in ordered:
            item = str(item or "").strip().lower()
            if item and item not in unique:
                unique.append(item)
        return " . ".join(unique) + " ."

    def label_key(self, label):
        return str(label or "").strip().lower()

    def iter_scene_objects(self, scene):
        if not scene:
            return

        for obj in scene.get("objects") or []:
            yield obj

        for task in scene.get("task_queue") or []:
            target = (task or {}).get("target") or {}
            if target.get("pose_world"):
                yield target
            place_goal = (task or {}).get("place_goal") or {}
            if place_goal.get("reference_pose_world"):
                yield {
                    "object_id": place_goal.get("reference_object_id"),
                    "label": place_goal.get("reference_label"),
                    "pose_world": place_goal.get("reference_pose_world"),
                    "size_3d": place_goal.get("reference_size_3d"),
                    "confidence": place_goal.get("confidence", 0.0),
                }

    def remember_initial_objects(self, scene):
        for obj in self.iter_scene_objects(scene) or []:
            label = self.label_key(
                obj.get("corrected_label")
                or obj.get("label")
                or obj.get("raw_label")
            )
            if not label or not obj.get("pose_world"):
                continue
            if label in ("table", "shelf"):
                continue
            if label not in self.initial_object_memory:
                self.initial_object_memory[label] = dict(obj)

    def find_initial_object(self, label_hint):
        hint = self.label_key(label_hint)
        if not hint:
            return None
        for label, obj in self.initial_object_memory.items():
            if hint in label or label in hint:
                return obj
        return None

    def publish_pick_goal(self, event):
        if self.task_ctx is None:
            return

        label = (
            self.task_ctx.pick_target.label
            or self.parsed_task.get("pick_label")
        )
        color, category = self.label_parts(label)
        source_surface = self.pick_source_surface()
        self.publish_json(
            self.pick_goal_pub,
            {
                "event": event,
                "task_id": self.task_ctx.task_id,
                "target_label": label,
                "target_color": color,
                "target_category": category,
                "expected_source_surface": source_surface,
                "grounding_prompt": self.pick_grounding_prompt(label),
                "query_role": "pick_target_only",
                "reference_label": self.parsed_task.get("reference_label"),
                "raw_instruction": self.task_ctx.raw_instruction,
                "source": "task_executor",
                "time": time.time(),
            },
        )

    def publish_pick_query(self):
        now = time.time()
        period = float(self.config.task_reference_query_period_sec)
        if now - self.last_pick_query_time < period:
            return

        label = (
            self.task_ctx.pick_target.label
            or self.parsed_task.get("pick_label")
            or "box"
        )

        msg = String()
        source_surface = self.pick_source_surface()
        msg.data = json.dumps(
            {
                "grounding_prompt": self.pick_grounding_prompt(label),
                "target_label": label,
                "target_color": self.label_parts(label)[0],
                "target_category": self.label_parts(label)[1],
                "expected_source_surface": source_surface,
                "query_role": "pick_target_only",
                "source": "task_executor_pick_query",
                "task_id": self.task_ctx.task_id,
            },
            ensure_ascii=False,
        )
        self.reference_query_pub.publish(msg)
        self.last_pick_query_time = now
        self.publish_task_status("pick_query_sent")

    def find_object(self, label_hint, role=None):
        if not self.latest_scene:
            return None

        # 兼容旧 scene_reader 规范化后的 task_queue target。
        task_queue = self.latest_scene.get("task_queue") or []
        if role == "pick" and task_queue:
            target = (task_queue[0] or {}).get("target") or {}
            target_label = str(target.get("label") or "").lower()
            hint = str(label_hint or "").lower()
            label_matches = (
                not hint
                or not target_label
                or hint in target_label
                or target_label in hint
            )
            if target.get("pose_world") and label_matches:
                return target

        objects = self.latest_scene.get("objects") or []
        label_hint = str(label_hint or "").lower()
        for obj in objects:
            label = str(obj.get("label") or "").lower()
            semantic_role = str(obj.get("semantic_role") or "").lower()
            label_matches = (
                not label_hint
                or not label
                or label_hint in label
                or label in label_hint
            )
            if role and role in semantic_role and obj.get("pose_world") and label_matches:
                return obj
            if label_hint and label_hint in label and obj.get("pose_world"):
                return obj
        return None

    def find_place_reference(self):
        reference = self.find_object(
            self.parsed_task.get("reference_label"),
            role="place_reference",
        )

        # 兼容扁平 scene 中的 reference 字段。
        if reference is None and self.latest_scene:
            task = (self.latest_scene.get("task_queue") or [{}])[0]
            place_goal = task.get("place_goal") or {}
            ref_pose = place_goal.get("reference_pose_world")
            ref_label = str(place_goal.get("reference_label") or "").lower()
            hint = str(self.parsed_task.get("reference_label") or "").lower()
            label_matches = (
                not hint
                or not ref_label
                or hint in ref_label
                or ref_label in hint
            )
            if ref_pose and label_matches:
                reference = {
                    "object_id": place_goal.get("reference_object_id"),
                    "label": place_goal.get("reference_label"),
                    "pose_world": ref_pose,
                }

        return reference

    def ground_references(self):
        pick = self.find_object(self.parsed_task.get("pick_label"), role="pick")
        relation = str(self.parsed_task.get("place_relation") or "").lower()
        temporal_reference = str(
            self.parsed_task.get("temporal_reference") or ""
        ).lower()

        # "原位置"必须绑定任务批次开始时保存的历史物体，不能被
        # 当前画面里同名物体（例如任务一已放到货架上的粉色箱子）覆盖。
        reference = None
        if relation == "original_position_of" or temporal_reference == "initial":
            reference = self.find_initial_object(
                self.parsed_task.get("reference_label")
            )

        # 历史记忆缺失时才允许使用当前语义场景作为兜底。
        if reference is None:
            reference = self.find_place_reference()

        if pick is None:
            # 起始视角可能看不到抓取目标。先保留语义目标，
            # 让导航节点进入搜索/重观察，之后用 ready_for_grasp 回填真实目标。
            self.task_ctx.pick_target = TaskObject(
                label=self.parsed_task.get("pick_label"),
            )
            if reference is not None:
                self.task_ctx.place_reference = TaskObject(
                    object_id=reference.get("object_id"),
                    label=reference.get("label"),
                    pose_world=reference.get("pose_world"),
                    size_3d=reference.get("size_3d"),
                    confidence=float(reference.get("confidence") or reference.get("dino_score") or 0.0),
                )
            elif not self.task_ctx.destination.place_pose:
                self.task_ctx.destination.type = "shelf_level_pending_reference"
                self.task_ctx.destination.shelf_id = default_shelf_id()
                self.task_ctx.destination.approach_pose = default_shelf_approach_pose()
            return True, "pick_target_deferred_wait_navigation_ready"

        self.task_ctx.pick_target = TaskObject(
            object_id=pick.get("object_id"),
            label=pick.get("label"),
            pose_world=pick.get("pose_world"),
            size_3d=pick.get("size_3d"),
            confidence=float(pick.get("confidence") or pick.get("dino_score") or 0.0),
        )

        if reference is None and not self.task_ctx.destination.place_pose:
            self.task_ctx.destination.type = "shelf_level_pending_reference"
            self.task_ctx.destination.shelf_id = default_shelf_id()
            self.task_ctx.destination.approach_pose = default_shelf_approach_pose()
            return True, "pick_grounded_place_reference_deferred"

        if reference is not None:
            self.task_ctx.place_reference = TaskObject(
                object_id=reference.get("object_id"),
                label=reference.get("label"),
                pose_world=reference.get("pose_world"),
                size_3d=reference.get("size_3d"),
                confidence=float(reference.get("confidence") or reference.get("dino_score") or 0.0),
            )
        return True, "references_grounded"

    def ground_place_reference_at_shelf(self):
        reference = self.find_place_reference()
        if reference is None:
            return False, "waiting_place_reference_at_shelf"

        self.task_ctx.place_reference = TaskObject(
            object_id=reference.get("object_id"),
            label=reference.get("label"),
            pose_world=reference.get("pose_world"),
            size_3d=reference.get("size_3d"),
            confidence=float(reference.get("confidence") or reference.get("dino_score") or 0.0),
        )
        return True, "place_reference_grounded_at_shelf"

    def send_grasp_command(self, command):
        # grasp_executor_node 当前接收纯文本命令，例如 "start"。
        msg = String()
        msg.data = str(command)
        self.grasp_command_pub.publish(msg)

    def send_navigation_goal(self, purpose, pose):
        goal_id = f"{self.task_ctx.task_id}_{purpose}_{int(time.time())}"
        self.pending_nav_goal_id = goal_id
        self.pending_nav_purpose = purpose
        self.pending_nav_pose = dict(pose)
        self.last_navigation_goal_time = time.time()
        self.publish_json(
            self.navigation_goal_pub,
            {
                "goal_id": goal_id,
                "purpose": purpose,
                "type": "pose2d",
                "pose": pose,
            },
        )

    def resend_pending_navigation_goal(self):
        if (
            not self.pending_nav_goal_id
            or not self.pending_nav_purpose
            or not self.pending_nav_pose
        ):
            return
        now = time.time()
        if now - self.last_navigation_goal_time < 2.0:
            return
        self.last_navigation_goal_time = now
        self.publish_json(
            self.navigation_goal_pub,
            {
                "goal_id": self.pending_nav_goal_id,
                "purpose": self.pending_nav_purpose,
                "type": "pose2d",
                "pose": self.pending_nav_pose,
            },
        )

    def send_place_command(self, command):
        self.publish_json(
            self.place_command_pub,
            {
                "command": command,
                "task_id": self.task_ctx.task_id,
                "held_object_id": self.task_ctx.held_object_id,
                "destination": self.task_ctx.destination.__dict__,
                "place_pose": self.task_ctx.destination.place_pose,
            },
        )

    def publish_reference_query(self):
        now = time.time()
        period = float(self.config.task_reference_query_period_sec)
        if now - self.last_reference_query_time < period:
            return

        label = self.parsed_task.get("reference_label") or "white cylinder"
        msg = String()
        msg.data = json.dumps(
            {
                "grounding_prompt": f"{label} . shelf .",
                "target_label": label,
                "query_role": "place_reference_search",
                "expected_source_surface": "shelf",
                "task_id": self.task_ctx.task_id if self.task_ctx else None,
                "source": "task_executor_reference_query",
            },
            ensure_ascii=False,
        )
        self.reference_query_pub.publish(msg)
        self.last_reference_query_time = now
        self.publish_task_status("reference_query_sent")

    def navigation_arrived(self, purpose):
        status = self.latest_nav_status or {}
        if status.get("state") != "ARRIVED":
            return False
        if status.get("purpose") != purpose:
            return False

        goal_id = status.get("goal_id")
        if goal_id == self.pending_nav_goal_id:
            return True

        # 调试/实机救场时允许人工放行，但必须显式写 manual_ 前缀，
        # 避免旧的 ARRIVED 状态误触发新的导航阶段。
        reason = str(status.get("reason") or "")
        if reason.startswith("manual_"):
            self.get_logger().warning(
                f"manual navigation arrival accepted for {purpose}: "
                f"expected goal_id={self.pending_nav_goal_id}, got={goal_id}"
            )
            return True

        return False

    def pick_ready(self):
        ready = self.latest_ready or {}
        if not ready.get("ready_for_grasp"):
            self.publish_task_status("waiting_pick_ready")
            return False

        max_age = float(getattr(self.config, "grasp_ready_max_age_sec", 3.0))
        ready_age = time.time() - self.latest_ready_rx
        if ready_age > max_age:
            self.publish_task_status("waiting_fresh_pick_ready")
            return False

        if ready.get("phase") != "ready_for_grasp":
            self.publish_task_status("waiting_navigation_ready_phase")
            return False

        reason = str(ready.get("reason") or "")
        allowed_reasons = {
            "pregrasp_head_spine_adjusted_target_visible",
            "pregrasp_close_range_locked_ready",
            "pregrasp_timeout_close_range_locked_ready",
            "ready_latched_waiting_grasp",
            "target_aligned_for_grasp_distance_valid_and_stable",
        }
        if reason not in allowed_reasons:
            self.publish_task_status(f"waiting_stable_pick_ready:{reason}")
            return False

        pick_label = str(self.task_ctx.pick_target.label or "").lower()
        ready_label = str(ready.get("target_label") or "").lower()
        if (
            pick_label
            and ready_label
            and pick_label not in ready_label
            and ready_label not in pick_label
        ):
            self.publish_task_status("waiting_pick_ready_target_match")
            return False

        self.task_ctx.pick_target = TaskObject(
            object_id=ready.get("target_object_id")
            or self.task_ctx.pick_target.object_id,
            label=ready.get("target_label")
            or self.task_ctx.pick_target.label,
            pose_world=ready.get("target_pose_world")
            or self.task_ctx.pick_target.pose_world,
            size_3d=ready.get("target_size_3d")
            or self.task_ctx.pick_target.size_3d,
            confidence=float(
                ready.get("confidence")
                or self.task_ctx.pick_target.confidence
                or 0.0
            ),
        )

        return True

    def grasp_done(self):
        status = self.latest_grasp_status or {}
        event = str(status.get("event") or "").lower()
        state = str(status.get("state") or "").upper()
        return event in ("lift_done_waiting_retreat", "grasp_done", "done") or state in ("DONE", "LIFT_DONE")

    def place_done(self):
        status = self.latest_place_status or {}
        return status.get("state") == "PLACE_DONE" or status.get("event") == "place_done"

    def reset_done(self):
        status = self.latest_grasp_status or {}
        if float(status.get("time") or 0.0) < self.reset_command_time:
            return False
        event = str(status.get("event") or "").lower()
        return event == "reset_done"

    def tick(self):
        if self.task_ctx is None:
            return

        state = self.task_ctx.state

        if state == TaskState.SAVE_HOME_POSE:
            if bool(getattr(self.config, "task_use_fixed_home_pose", True)):
                self.task_ctx.home_pose = {
                    "x": float(getattr(self.config, "task_home_x", -0.70)),
                    "y": float(getattr(self.config, "task_home_y", 0.55)),
                    "yaw": float(
                        getattr(
                            self.config,
                            "task_home_yaw",
                            1.5707963267948966,
                        )
                    ),
                }
                self.set_state(TaskState.GROUND_REFERENCES, "fixed_home_pose_saved")
                return
            if self.robot_pose is None:
                self.publish_task_status("waiting_odom_for_home_pose")
                return
            self.task_ctx.home_pose = dict(self.robot_pose)
            self.set_state(TaskState.GROUND_REFERENCES, "home_pose_saved")
            return

        if state == TaskState.GROUND_REFERENCES:
            ok, reason = self.ground_references()
            if not ok:
                self.publish_task_status(reason)
                return
            if self.task_ctx.place_reference.pose_world or self.task_ctx.destination.place_pose:
                self.set_state(TaskState.RESOLVE_DESTINATION, reason)
            else:
                self.set_state(TaskState.WAIT_PICK_READY, reason)
            return

        if state == TaskState.RESOLVE_DESTINATION:
            ok, reason = resolve_destination(self.task_ctx, self.config)
            if not ok:
                if (
                    "pick" in self.task_ctx.completed_steps
                    and reason == "reference_object_not_inside_configured_shelf"
                ):
                    self.task_ctx.place_reference.pose_world = None
                    self.publish_task_status(f"retry_place_reference:{reason}")
                    self.set_state(TaskState.GROUND_PLACE_REFERENCE, reason)
                    return
                self.set_state(TaskState.SAFE_STOP, reason)
                return
            if "pick" in self.task_ctx.completed_steps:
                self.set_state(TaskState.NAVIGATE_TO_PLACE, reason)
            else:
                self.set_state(TaskState.WAIT_PICK_READY, reason)
            return

        if state == TaskState.WAIT_PICK_READY:
            self.publish_pick_goal("wait_pick_ready")
            self.publish_pick_query()
            if self.pick_ready():
                self.set_state(TaskState.START_PICK, "pick_ready_received")
            return

        if state == TaskState.START_PICK:
            self.send_grasp_command("start")
            self.task_ctx.held_object_id = self.task_ctx.pick_target.object_id
            self.set_state(TaskState.WAIT_PICK_DONE, "grasp_start_sent")
            return

        if state == TaskState.WAIT_PICK_DONE:
            if self.grasp_done():
                self.task_ctx.completed_steps.append("pick")
                if self.task_ctx.place_reference.pose_world or self.task_ctx.destination.place_pose:
                    self.set_state(TaskState.NAVIGATE_TO_PLACE, "pick_done")
                else:
                    self.set_state(
                        TaskState.NAVIGATE_TO_REFERENCE_AREA,
                        "pick_done_reference_deferred",
                    )
            else:
                self.publish_task_status("waiting_pick_done")
            return

        if state == TaskState.NAVIGATE_TO_REFERENCE_AREA:
            approach_pose = (
                self.task_ctx.destination.approach_pose
                or default_shelf_approach_pose()
            )
            if approach_pose is None:
                self.set_state(TaskState.SAFE_STOP, "default_shelf_approach_missing")
                return
            self.task_ctx.destination.shelf_id = (
                self.task_ctx.destination.shelf_id
                or default_shelf_id()
            )
            self.task_ctx.destination.approach_pose = approach_pose
            self.send_navigation_goal("reference_area", approach_pose)
            self.set_state(
                TaskState.GROUND_PLACE_REFERENCE,
                "reference_area_navigation_goal_sent",
            )
            return

        if state == TaskState.GROUND_PLACE_REFERENCE:
            if not self.navigation_arrived("reference_area"):
                self.resend_pending_navigation_goal()
                self.publish_task_status("waiting_reference_area_arrival")
                return

            self.publish_reference_query()
            ok, reason = self.ground_place_reference_at_shelf()
            if not ok:
                self.publish_task_status(reason)
                return

            self.set_state(TaskState.RESOLVE_DESTINATION, reason)
            return

        if state == TaskState.NAVIGATE_TO_PLACE:
            self.send_navigation_goal(
                "place",
                self.task_ctx.destination.approach_pose,
            )
            self.set_state(TaskState.SELECT_PLACE_POSE, "place_navigation_goal_sent")
            return

        if state == TaskState.SELECT_PLACE_POSE:
            if not self.navigation_arrived("place"):
                self.resend_pending_navigation_goal()
                self.publish_task_status("waiting_place_navigation_arrival")
                return
            place_pose, reason = plan_place_pose(self.task_ctx, self.latest_scene, self.config)
            if place_pose is None:
                self.set_state(TaskState.SAFE_STOP, reason)
                return
            self.task_ctx.destination.place_pose = place_pose
            self.set_state(TaskState.START_PLACE, reason)
            return

        if state == TaskState.START_PLACE:
            self.send_place_command("start")
            self.set_state(TaskState.WAIT_PLACE_DONE, "place_start_sent")
            return

        if state == TaskState.WAIT_PLACE_DONE:
            if self.place_done():
                self.task_ctx.completed_steps.append("place")
                self.task_ctx.held_object_id = None
                self.set_state(TaskState.RETURN_HOME, "place_done")
            else:
                self.publish_task_status("waiting_place_done")
            return

        if state == TaskState.RETURN_HOME:
            self.send_navigation_goal("home", self.task_ctx.home_pose)
            self.set_state(TaskState.RESET_ROBOT, "return_home_goal_sent")
            return

        if state == TaskState.RESET_ROBOT:
            if self.navigation_arrived("home"):
                if "return_home" not in self.task_ctx.completed_steps:
                    self.task_ctx.completed_steps.append("return_home")
                self.reset_command_time = time.time()
                self.send_grasp_command("reset")
                self.set_state(TaskState.WAIT_RESET_DONE, "home_arrived_reset_sent")
            else:
                self.resend_pending_navigation_goal()
                self.publish_task_status("waiting_home_arrival")
            return

        if state == TaskState.WAIT_RESET_DONE:
            if self.reset_done():
                if "reset" not in self.task_ctx.completed_steps:
                    self.task_ctx.completed_steps.append("reset")
                self.official_completed_task_ids.add(str(self.task_ctx.task_id))
                self.set_state(TaskState.TASK_DONE, "robot_reset_done")
            else:
                self.publish_task_status("waiting_robot_reset")
            return

        if state == TaskState.TASK_DONE:
            if not self.task_done_reported:
                self.task_done_reported = True
                self.official_completed_task_ids.add(str(self.task_ctx.task_id))
                self.publish_task_status("task_done")
            if self.official_task_queue:
                next_task = self.official_task_queue.pop(0)
                self.start_parsed_task(next_task, source="official_queue")
            return

        if state == TaskState.SAFE_STOP:
            self.publish_task_status(self.task_ctx.reason or "safe_stop")


def main():
    rclpy.init()
    node = TaskExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
