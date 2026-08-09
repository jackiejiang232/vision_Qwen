#!/usr/bin/env python3
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
from .active_observer import ActiveObserver
from .motion_astar_nav import make_dock_controller

from .action_config import CONFIG
from .goal_builder import (
    Pose2D,
    build_approach_goal_from_target,
    build_search_goal,
    pose2d_from_motion_handoff,
)
from .motion_handoff_adapter import build_motion_handoff
from .scene_reader import (
    bind_task_target_from_objects,
    choose_search_areas,
    get_active_task,
    parse_scene_message,
    target_is_detected,
    target_is_visible_for_servo,
)
from .simple_nav import SimpleCmdVelNavigator
from .target_lock import SearchTargetLock
from .visual_servo import ServoState, VisualServo


class Phase:
    WAIT_SCENE = "wait_scene"
    SELECT_TASK = "select_task"
    PLAN = "plan"
    NAVIGATE = "navigate"
    SEARCH_ROTATE = "search_rotate"
    SCAN_AND_REOBSERVE = "scan_and_reobserve"
    ACTIVE_OBSERVE = "active_observe"
    VISUAL_SERVO = "visual_servo"
    PREGRASP_ADJUST = "pregrasp_adjust"
    READY_FOR_GRASP = "ready_for_grasp"
    SAFE_STOP = "safe_stop"


class ActionNavigationNode(Node):
    def __init__(self):
        super().__init__("action_navigation_node")

        self.config = CONFIG
        self.phase = Phase.WAIT_SCENE

        self.latest_scene = None
        self.latest_scene_time = 0.0
        self.latest_detection_scene = None
        self.latest_detection_time = 0.0
        self.latest_detection_seq = 0
        self.active_task = None
        self.current_goal = None
        self.robot_pose = None
        self.phase_start_time = time.monotonic()
        self.last_ready_publish_time = 0.0
        self.last_ready_reason = None
        self.ready_task_id = None
        self.ready_target_snapshot = None
        self.ready_robot_pose = None
        self.spine_position = None
        self.spine_state_time = 0.0
        self.head_pitch_position = None
        self.pregrasp_stable_frames = 0
        self.pregrasp_lost_frames = 0
        self.pregrasp_last_safe_spine = None
        self.pregrasp_last_safe_head_pitch = None
        self.pregrasp_start_spine = None
        self.pregrasp_target_spine = None
        self.pregrasp_target_head_pitch = None
        self.head_state_time = 0.0
        self.current_dock_route = None
        self.current_dock_controller = None
        self.motion_astar_enabled = False
        self.motion_astar_waypoint_count = 0
        self.motion_astar_cost_m = 0.0
        self.last_dock_progress = None
        self.motion_target_metadata = {}
        self.current_motion_metadata = {}
        self.motion_task_metadata = {}
        self.grasp_handoff_active = False

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.config.cmd_vel_topic,
            10,
        )
        self.ready_pub = self.create_publisher(
            String,
            self.config.ready_topic,
            10,
        )

        self.scene_sub = self.create_subscription(
            String,
            self.config.scene_topic,
            self.on_scene,
            10,
        )
        self.detections_sub = self.create_subscription(
            String,
            self.config.detections_topic,
            self.on_detections,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.config.odom_topic,
            self.on_odom,
            10,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self.on_joint_state,
            10,
        )
        self.grasp_command_sub = self.create_subscription(
            String,
            self.config.grasp_command_topic,
            self.on_grasp_command,
            10,
        )
        self.grasp_status_sub = self.create_subscription(
            String,
            self.config.grasp_status_topic,
            self.on_grasp_status,
            10,
        )

        self.navigator = SimpleCmdVelNavigator(
            self.cmd_vel_pub,
            self.config,
        )
        self.visual_servo = VisualServo(
            self.cmd_vel_pub,
            self.config,
        )

        self.timer = self.create_timer(
            1.0 / self.config.control_rate_hz,
            self.on_timer,
        )
        self.arrived_scene_time = 0.0
        self.arrived_detection_time = 0.0
        self.last_observer_view_name = None
        self.navigation_mode = None
        self.search_areas = []
        self.search_area_index = 0
        self.search_yaw_index = 0
        self.search_center_yaw = None
        self.search_turn_goal = None
        self.observe_resume = False
        self.approach_retry_count = 0
        self.approach_target_snapshot = None
        self.servo_target_snapshot = None
        self.servo_target_snapshot_time = 0.0
        self.servo_target_observation_id = None
        self.search_target_lock = SearchTargetLock(self.config)
        self.confirmed_search_task = None
        self.observer = ActiveObserver(
            self,
            self.config,
        )

    def set_phase(self, phase):
        if phase != self.phase:
            self.get_logger().info(
                f"phase: {self.phase} -> {phase}"
            )
            self.phase = phase
            self.phase_start_time = time.monotonic()

    def on_scene(self, msg):
        try:
            self.latest_scene = parse_scene_message(msg.data)
            self.latest_scene_time = time.monotonic()
        except Exception as error:
            self.get_logger().warning(
                f"解析scene_understanding失败: {error}"
            )

    def on_detections(self, msg):
        try:
            payload = json.loads(msg.data)
            source_stamp = payload.get("source_stamp") or {}
            self.latest_detection_seq += 1
            self.latest_detection_scene = {
                "source_stamp_sec": source_stamp.get("sec"),
                "source_stamp_nanosec": source_stamp.get("nanosec"),
                "observation_seq": self.latest_detection_seq,
                "objects": payload.get("detections") or [],
            }
            self.latest_detection_time = time.monotonic()
            self.update_search_target_lock()
        except Exception as error:
            self.get_logger().warning(
                f"解析grounded_sam检测失败: {error}"
            )

    def on_odom(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        quat = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        yaw = Rotation.from_quat(quat).as_euler("xyz")[2]

        self.robot_pose = Pose2D(
            x=position.x,
            y=position.y,
            yaw=yaw,
        )

    def on_joint_state(self, msg):
        now = time.monotonic()

        try:
            index = msg.name.index(self.config.spine_joint_name)
            if index < len(msg.position):
                self.spine_position = float(msg.position[index])
                self.spine_state_time = now
        except ValueError:
            pass

        try:
            index = msg.name.index(self.config.head_pitch_joint_name)
            if index < len(msg.position):
                self.head_pitch_position = float(msg.position[index])
                self.head_state_time = now
        except ValueError:
            pass

    def on_grasp_command(self, msg):
        command = msg.data.strip().lower()
        if command == "start" and self.phase == Phase.READY_FOR_GRASP:
            self.enter_grasp_handoff("grasp_start_command")
        elif command == "abort":
            self.exit_grasp_handoff("grasp_abort_command")

    def on_grasp_status(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        event = str(payload.get("event") or "")
        state = str(payload.get("state") or "")
        if event == "safe_stop" or state == "SAFE_STOP":
            self.exit_grasp_handoff("grasp_executor_safe_stop")

    def enter_grasp_handoff(self, reason):
        if not self.grasp_handoff_active:
            self.get_logger().info(
                f"enter grasp handoff: {reason}"
            )
        self.grasp_handoff_active = True
        self.navigator.stop()
        self.visual_servo.stop(reset_stability=False)
        self.cmd_vel_pub.publish(Twist())

    def exit_grasp_handoff(self, reason):
        if self.grasp_handoff_active:
            self.get_logger().info(
                f"exit grasp handoff: {reason}"
            )
        self.grasp_handoff_active = False

    def target_is_table_target(self, target):
        return not (
            target.get("support_surface") == "shelf"
            or bool(target.get("on_shelf"))
        )

    def expected_table_spine(self, target):
        if not self.target_is_table_target(target):
            return None

        sequence = self.config.table_observe_sequence
        if not sequence:
            return None

        return float(sequence[0][3])

    #根据【pose_world.z】、【size_3d.height】动态决定腰部高度
    def estimate_pregrasp_spine_from_target(self, target):
        pose = target.get("pose_world") or {}
        size = target.get("size_3d") or {}

        if "z" not in pose:
            return None

        target_center_z = float(pose["z"])
        target_height = float(size.get("height") or 0.12)

        target_bottom_z = target_center_z - target_height * 0.5
        grasp_z = (
            target_bottom_z
            + target_height
            * self.config.pregrasp_grasp_height_ratio
        )

        desired_spine = (
            self.config.pregrasp_spine_reference_z
            - grasp_z
            + self.config.pregrasp_spine_lower_bias
        )

        current_spine = (
            float(self.spine_position)
            if self.spine_position is not None
            else desired_spine
        )

        desired_spine = max(
            current_spine + self.config.pregrasp_spine_min_delta,
            min(
                current_spine + self.config.pregrasp_spine_max_delta,
                desired_spine,
            ),
        )

        return max(
            self.config.table_pregrasp_spine_min,
            min(
                self.config.table_pregrasp_spine_max,
                desired_spine,
            ),
        )
    
    def table_target_distance(self, target):
        if self.robot_pose is None:
            return None

        pose = target.get("pose_world") or {}
        if "x" not in pose or "y" not in pose:
            return None

        return math.hypot(
            float(pose["x"]) - float(self.robot_pose.x),
            float(pose["y"]) - float(self.robot_pose.y),
        )
    def table_pregrasp_alignment_errors(self, target):
        if self.robot_pose is None:
            return None

        pose = target.get("pose_world") or {}
        if "x" not in pose or "y" not in pose:
            return None

        dx = float(pose["x"]) - float(self.robot_pose.x)
        dy = float(pose["y"]) - float(self.robot_pose.y)
        yaw = float(self.robot_pose.yaw)

        forward_error = (
            math.cos(yaw) * dx
            + math.sin(yaw) * dy
        )

        lateral_error = (
            -math.sin(yaw) * dx
            + math.cos(yaw) * dy
        )

        desired_yaw = math.atan2(dy, dx)
        heading_error = math.atan2(
            math.sin(desired_yaw - yaw),
            math.cos(desired_yaw - yaw),
        )

        return {
            "forward_m": forward_error,
            "lateral_m": lateral_error,
            "heading_rad": heading_error,
        }

    def table_front_yaw_error(self):
        if self.robot_pose is None:
            return None

        yaw_error = (
            float(self.robot_pose.yaw)
            - float(self.config.table_front_yaw)
        )

        return abs(
            math.atan2(
                math.sin(yaw_error),
                math.cos(yaw_error),
            )
        )

    def close_range_lock_ready(self, target):
        if not getattr(self.config, "close_range_lock_enable", True):
            return False

        if not target or not self.target_is_table_target(target):
            return False

        if not self.target_pose_valid_for_surface(target):
            return False

        errors = self.table_pregrasp_alignment_errors(target)
        if errors is None:
            return False

        table_yaw_error = self.table_front_yaw_error()
        if table_yaw_error is None:
            return False

        distance_extra = float(
            getattr(self.config, "close_range_lock_extra_distance_m", 0.08)
        )
        forward_min = max(
            0.0,
            float(self.config.table_grasp_distance_min) - distance_extra,
        )
        forward_max = (
            float(self.config.table_grasp_distance_max)
            + distance_extra
        )
        lateral_tolerance = float(
            getattr(
                self.config,
                "close_range_lock_lateral_tolerance_m",
                self.config.table_pregrasp_lateral_tolerance_m,
            )
        )
        heading_tolerance = float(
            getattr(
                self.config,
                "close_range_lock_heading_tolerance_rad",
                self.config.table_pregrasp_heading_tolerance_rad,
            )
        )
        table_yaw_tolerance = float(
            getattr(
                self.config,
                "close_range_lock_table_yaw_tolerance_rad",
                self.config.table_pregrasp_table_yaw_tolerance_rad,
            )
        )

        return (
            forward_min <= float(errors["forward_m"]) <= forward_max
            and abs(float(errors["lateral_m"])) <= lateral_tolerance
            and abs(float(errors["heading_rad"])) <= heading_tolerance
            and table_yaw_error <= table_yaw_tolerance
        )

    def close_range_lock_target(self, preferred_target=None):
        candidates = (
            preferred_target,
            self.recent_servo_target(),
            self.approach_target_snapshot,
            self.ready_target_snapshot,
        )

        for candidate in candidates:
            if not candidate:
                continue

            target = dict(candidate)
            target = self.restore_motion_metadata(target)
            target = self.apply_pregrasp_metadata(target)
            if self.close_range_lock_ready(target):
                target["close_range_locked"] = True
                return target

        return None
    def remember_motion_metadata(self, target):
        if not target:
            return

        keys = (
            "motion_source_area",
            "motion_grasp_profile",
            "motion_astar_enabled",
            "motion_astar_waypoint_count",
            "motion_astar_cost_m",
        )

        metadata = {
            key: target.get(key)
            for key in keys
            if target.get(key) is not None
        }

        if not metadata:
            return

        # 1. 当前任务级缓存：最稳，因为 active_task_id 不随检测帧变化。
        self.current_motion_metadata = dict(metadata)

        task_id = None
        if self.active_task is not None:
            task_id = self.active_task.get("task_id")

        if task_id is not None:
            self.motion_task_metadata[task_id] = dict(metadata)

        # 2. object_id 缓存保留，但只作为辅助。
        object_id = target.get("object_id")
        if object_id:
            self.motion_target_metadata[object_id] = dict(metadata)


    def restore_motion_metadata(self, target):
        if not target:
            return target

        metadata = None

        object_id = target.get("object_id")
        if object_id:
            metadata = self.motion_target_metadata.get(object_id)

        if metadata is None and self.active_task is not None:
            task_id = self.active_task.get("task_id")
            metadata = self.motion_task_metadata.get(task_id)

        if metadata is None:
            metadata = self.current_motion_metadata

        if not metadata:
            return target

        target = dict(target)

        # 注意：不能用 setdefault。
        # 如果 key 已经存在但值是 None，setdefault 不会覆盖。
        for key, value in metadata.items():
            if target.get(key) is None:
                target[key] = value

        return target

    def apply_pregrasp_metadata(self, target):
        if not target or self.current_goal is None:
            return target

        pose = target.get("pose_world") or {}
        if "x" not in pose:
            return target

        target = dict(target)
        if self.target_is_table_target(target):
            lateral = float(pose["x"]) - float(self.current_goal.x)
            target["selected_arm"] = "dual"
            target["servo_target_u"] = float(
                self.config.image_center_u
            )
            target["pregrasp_lateral_m"] = lateral
        else:
            target["selected_arm"] = None
            target["servo_target_u"] = float(
                self.config.image_center_u
            )
            target["pregrasp_lateral_m"] = 0.0

        target["pregrasp_goal_pose"] = self.current_goal.__dict__
        return target

    def target_pose_valid_for_surface(self, target):
        pose = target.get("pose_world") or {}
        if not all(axis in pose for axis in ("x", "y", "z")):
            return False

        bounds = (
            self.config.shelf_target_bounds_xyz
            if (
                target.get("support_surface") == "shelf"
                or bool(target.get("on_shelf"))
            )
            else self.config.table_target_bounds_xyz
        )

        for axis, limits in zip(("x", "y", "z"), bounds):
            value = float(pose[axis])
            if not (float(limits[0]) <= value <= float(limits[1])):
                return False

        return True

    def remember_servo_target(self, target):
        if not target or not self.target_pose_valid_for_surface(target):
            return

        self.servo_target_snapshot = dict(target)
        self.servo_target_snapshot_time = time.monotonic()
        if self.latest_detection_scene is not None:
            self.servo_target_observation_id = (
                self.latest_detection_scene.get("observation_seq"),
            )

    def recent_servo_target(self):
        if self.servo_target_snapshot is None:
            return None
        age = time.monotonic() - self.servo_target_snapshot_time
        if age > self.config.servo_target_lock_ttl_sec:
            return None
        return dict(self.servo_target_snapshot)

    def grasp_clearance_ready(self, target):
        if not self.target_is_table_target(target):
            return True

        errors = self.table_pregrasp_alignment_errors(target)
        if errors is None:
            return False

        forward_m = errors["forward_m"]
        lateral_m = abs(errors["lateral_m"])
        heading_rad = abs(errors["heading_rad"])

        table_yaw_error = self.table_front_yaw_error()
        if table_yaw_error is None:
            return False

        distance_min = float(self.config.table_grasp_distance_min)
        distance_max = float(self.config.table_grasp_distance_max)
        lateral_tolerance = float(
            self.config.table_pregrasp_lateral_tolerance_m
        )
        heading_tolerance = float(
            self.config.table_pregrasp_heading_tolerance_rad
        )
        table_yaw_tolerance = float(
            self.config.table_pregrasp_table_yaw_tolerance_rad
        )

        if target.get("close_range_locked"):
            distance_extra = float(
                getattr(
                    self.config,
                    "close_range_lock_extra_distance_m",
                    0.08,
                )
            )
            distance_min = max(0.0, distance_min - distance_extra)
            distance_max = distance_max + distance_extra
            lateral_tolerance = float(
                getattr(
                    self.config,
                    "close_range_lock_lateral_tolerance_m",
                    lateral_tolerance,
                )
            )
            heading_tolerance = float(
                getattr(
                    self.config,
                    "close_range_lock_heading_tolerance_rad",
                    heading_tolerance,
                )
            )
            table_yaw_tolerance = float(
                getattr(
                    self.config,
                    "close_range_lock_table_yaw_tolerance_rad",
                    table_yaw_tolerance,
                )
            )

        if (
            table_yaw_error
            > table_yaw_tolerance
        ):
            return False

        if not (
            distance_min
            <= forward_m
            <= distance_max
        ):
            return False

        if lateral_m > lateral_tolerance:
            return False

        if heading_rad > heading_tolerance:
            return False

        return True

    def grasp_window_ready(self, target):
        if not target or self.robot_pose is None:
            return False

        if not self.target_pose_valid_for_surface(target):
            return False

        if not self.grasp_clearance_ready(target):
            return False

        return self.visual_servo.ready_now(
            target,
            self.robot_pose,
        )

    def current_search_area(self):
        if (
            self.navigation_mode != "search"
            or not self.search_areas
            or self.search_area_index >= len(self.search_areas)
        ):
            return None
        return self.search_areas[self.search_area_index]

    def update_search_target_lock(self):
        area_name = self.current_search_area()
        if (
            area_name is None
            or self.latest_scene is None
            or self.phase not in (
                Phase.SEARCH_ROTATE,
                Phase.SCAN_AND_REOBSERVE,
                Phase.ACTIVE_OBSERVE,
            )
        ):
            return

        task = bind_task_target_from_objects(
            self.latest_detection_scene,
            get_active_task(self.latest_scene),
        )
        observation_id = (
            self.latest_detection_scene.get("observation_seq"),
        )
        confirmed_target = self.search_target_lock.update(
            task,
            area_name,
            observation_id,
        )
        if confirmed_target is None:
            return

        confirmed_task = dict(task or {})
        confirmed_task["target"] = confirmed_target
        self.confirmed_search_task = confirmed_task

    def get_confirmed_search_task(self):
        confirmed_target = self.search_target_lock.get_confirmed()
        if confirmed_target is None or self.confirmed_search_task is None:
            return None
        task = dict(self.confirmed_search_task)
        task["target"] = confirmed_target
        return task

    def publish_ready(self, ready, reason):
        task = self.active_task or {}
        target = task.get("target") or {}

        semantic_age = None
        if self.latest_scene_time > 0.0:
            semantic_age = time.monotonic() - self.latest_scene_time

        detection_age = None
        if self.latest_detection_time > 0.0:
            detection_age = time.monotonic() - self.latest_detection_time

        spine_state_age = None
        if self.spine_state_time > 0.0:
            spine_state_age = time.monotonic() - self.spine_state_time

        head_state_age = None
        if self.head_state_time > 0.0:
            head_state_age = time.monotonic() - self.head_state_time

        expected_spine = None
        table_distance = None
        if target:
            target = self.restore_motion_metadata(target)
            target = self.apply_pregrasp_metadata(target)
            expected_spine = self.expected_table_spine(target)
            table_distance = self.table_target_distance(target)

        payload = {
            "ready_for_grasp": bool(ready),
            "phase": self.phase,
            "task_id": task.get("task_id"),
            "target_object_id": target.get("object_id"),
            "target_label": target.get("label"),
            "target_pose_world": target.get("pose_world"),
            "target_box_xyxy": target.get("box_xyxy"),
            "target_centroid_uv": target.get("centroid_uv"),
            "selected_arm": target.get("selected_arm"),
            "servo_target_u": target.get("servo_target_u"),
            "pregrasp_lateral_m": target.get("pregrasp_lateral_m"),
            "pregrasp_stable_frames": self.pregrasp_stable_frames,
            "pregrasp_last_safe_spine": self.pregrasp_last_safe_spine,
            "pregrasp_last_safe_head_pitch": self.pregrasp_last_safe_head_pitch,
            "spine_position": self.spine_position,
            "expected_spine_position": expected_spine,
            "spine_state_age_sec": spine_state_age,
            "head_pitch_position": self.head_pitch_position,
            "head_state_age_sec": head_state_age,
            "table_target_distance": table_distance,
            "arm_clearance_ready": self.grasp_clearance_ready(target),
            "grasp_window_ready": self.grasp_window_ready(target),
            "close_range_locked": bool(target.get("close_range_locked")),
            "close_range_lock_ready": self.close_range_lock_ready(target),
            "base_goal_pose": (
                self.current_goal.__dict__
                if self.current_goal is not None
                else None
            ),
            "has_scene": self.latest_scene is not None,
            "has_odom": self.robot_pose is not None,
            "semantic_age_sec": semantic_age,
            "detection_age_sec": detection_age,
            "detection_seq": self.latest_detection_seq,
            "visual_servo_stable_frames": self.visual_servo.stable_frames,
            "visual_servo_last_observation_id": self.visual_servo.last_observation_id,
            "reason": reason,
            "navigation_mode": self.navigation_mode,
            "search_area": (
                self.search_areas[self.search_area_index]
                if (
                    self.navigation_mode == "search"
                    and self.search_areas
                    and self.search_area_index < len(self.search_areas)
                )
                else None
            ),
            "search_yaw_index": self.search_yaw_index,
            "approach_retry_count": self.approach_retry_count,
            "observer_view": (
                self.observer.current_view_name
                if hasattr(self, "observer")
                else None
            ),
            "search_candidate_frames": self.search_target_lock.frame_count,
            "search_candidate_rejection": (
                self.search_target_lock.last_rejection_reason
            ),
            "table_pregrasp_alignment": (
                self.table_pregrasp_alignment_errors(target)
                if target and self.target_is_table_target(target)
                else None
            ),
            "motion_source_area": target.get("motion_source_area"),
            "motion_grasp_profile": target.get("motion_grasp_profile"),
            "target_size_3d": target.get("size_3d"),
            "target_yaw_world_rad": target.get("yaw_world_rad"),
            "position_std_m": target.get("position_std_m"),
            "yaw_std_rad": target.get("yaw_std_rad"),
             #A*导入
            "motion_astar_enabled": self.motion_astar_enabled,
            "motion_astar_waypoint_count": self.motion_astar_waypoint_count,
            "motion_astar_cost_m": self.motion_astar_cost_m,
            "motion_dock_progress": (
                {
                    "phase": self.last_dock_progress.phase,
                    "position_error": self.last_dock_progress.position_error,
                    "yaw_error": self.last_dock_progress.yaw_error,
                    "stable_for": self.last_dock_progress.stable_for,
                    "completed": self.last_dock_progress.completed,
                }
                if self.last_dock_progress is not None
                else None
            ),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.ready_pub.publish(msg)
        self.last_ready_publish_time = time.monotonic()
        self.last_ready_reason = reason

    def publish_ready_heartbeat(self):
        now = time.monotonic()
        if now - self.last_ready_publish_time < 1.0:
            return

        if self.phase == Phase.WAIT_SCENE:
            reason = "waiting_scene_understanding"
        elif self.phase == Phase.SELECT_TASK:
            reason = "selecting_active_task"
        elif self.phase == Phase.PLAN:
            reason = "planning_navigation_goal"
        elif self.phase == Phase.NAVIGATE:
            reason = "navigating_to_goal"
        elif self.phase == Phase.SEARCH_ROTATE:
            reason = "rotating_base_to_search_target"
        elif self.phase == Phase.SCAN_AND_REOBSERVE:
            reason = "preparing_active_observe"
        elif self.phase == Phase.ACTIVE_OBSERVE:
            reason = "active_observing"
        elif self.phase == Phase.VISUAL_SERVO:
            reason = "visual_servo_aligning"
        elif self.phase == Phase.READY_FOR_GRASP:
            reason = "target_detected_and_base_aligned"
        else:
            reason = self.last_ready_reason or "safe_stop"

        self.publish_ready(
            self.phase == Phase.READY_FOR_GRASP,
            reason,
        )

    def on_timer(self):
        if self.phase == Phase.WAIT_SCENE:
            self.handle_wait_scene()
        elif self.phase == Phase.SELECT_TASK:
            self.handle_select_task()
        elif self.phase == Phase.PLAN:
            self.handle_plan()
        elif self.phase == Phase.NAVIGATE:
            self.handle_navigate()
        elif self.phase == Phase.SEARCH_ROTATE:
            self.handle_search_rotate()
        elif self.phase == Phase.SCAN_AND_REOBSERVE:
            self.handle_scan_and_reobserve()
        elif self.phase == Phase.ACTIVE_OBSERVE:
            self.handle_active_observe()
        elif self.phase == Phase.VISUAL_SERVO:
            self.handle_visual_servo()
        elif self.phase == Phase.PREGRASP_ADJUST:
            self.handle_pregrasp_adjust()
        elif self.phase == Phase.READY_FOR_GRASP:
            self.handle_ready_for_grasp()
        elif self.phase == Phase.SAFE_STOP:
            self.navigator.stop()

        self.publish_ready_heartbeat()

    def handle_wait_scene(self):
        self.navigator.stop()
        if self.latest_scene is not None:
            self.set_phase(Phase.SELECT_TASK)

    def bind_current_task(self):
        if self.latest_scene is None:
            return None

        perception_scene = (
            self.latest_detection_scene or self.latest_scene
        )
        return bind_task_target_from_objects(
            perception_scene,
            get_active_task(self.latest_scene),
        )

    def handle_select_task(self):
        self.active_task = self.bind_current_task()
        if self.active_task is None:
            self.publish_ready(False, "no_active_task")
            self.set_phase(Phase.WAIT_SCENE)
            return

        self.set_phase(Phase.PLAN)

    def plan_approach_to_current_target(self, reason):
        if not target_is_detected(self.active_task):
            return False

        target = self.active_task["target"]
        if not self.target_pose_valid_for_surface(target):
            return False

        distance_extra = (
            self.approach_retry_count
            * self.config.approach_retry_backoff_m
        )
        motion_goal = None

        if self.config.enable_motion_handoff_stance:
            try:
                motion_goal = build_motion_handoff(
                    self.latest_scene,
                    self.config,
                    robot_pose=self.robot_pose,
                )
            except Exception as error:
                self.get_logger().warning(
                    f"DG202612 motion handoff failed, fallback local goal: {error}"
                )

        if motion_goal is not None and motion_goal.status == "ok":
            self.current_goal = pose2d_from_motion_handoff(
                motion_goal.approach_pose
            )

            self.current_dock_route = motion_goal.dock_route
            self.current_dock_controller = None
            self.motion_astar_enabled = bool(motion_goal.astar_enabled)
            self.motion_astar_waypoint_count = int(
                motion_goal.astar_waypoint_count
            )
            self.motion_astar_cost_m = float(motion_goal.astar_cost_m)

            if self.current_dock_route is not None:
                self.current_dock_controller = make_dock_controller(
                    self.current_dock_route,
                    self.config,
                )

            target["motion_source_area"] = motion_goal.source_area
            target["motion_grasp_profile"] = motion_goal.grasp_profile
            target["motion_astar_enabled"] = self.motion_astar_enabled
            target["motion_astar_waypoint_count"] = self.motion_astar_waypoint_count
            target["motion_astar_cost_m"] = self.motion_astar_cost_m

            self.remember_motion_metadata(target)
        else:
            self.current_dock_route = None
            self.current_dock_controller = None
            self.motion_astar_enabled = False
            self.motion_astar_waypoint_count = 0
            self.motion_astar_cost_m = 0.0
            self.current_goal = build_approach_goal_from_target(
                target,
                self.config,
                self.robot_pose,
                distance_extra=distance_extra,
            )
        target = self.restore_motion_metadata(target)
        target = self.apply_pregrasp_metadata(target)
        self.active_task = dict(self.active_task)
        self.active_task["target"] = target
        self.approach_target_snapshot = dict(target)
        self.navigation_mode = "approach"
        # 底盘位姿变化后，旧观察高度已不对应当前视角。
        # 到达新接近点时必须从完整观察序列起点重新扫描。
        self.observe_resume = False
        self.publish_ready(False, reason)
        self.set_phase(Phase.NAVIGATE)
        return True

    def start_search_area(self, area_index):
        self.search_area_index = area_index
        self.search_yaw_index = 0
        self.search_turn_goal = None
        self.search_target_lock.reset()
        self.confirmed_search_task = None
        self.approach_target_snapshot = None
        area_name = self.search_areas[area_index]
        self.current_goal = build_search_goal(area_name)
        self.navigation_mode = "search"
        self.observe_resume = False
        self.servo_target_snapshot = None
        self.servo_target_snapshot_time = 0.0
        self.servo_target_observation_id = None
        self.publish_ready(
            False,
            f"navigate_to_search_area_{area_name}",
        )
        self.set_phase(Phase.NAVIGATE)

    def advance_search(self):
        self.search_yaw_index += 1
        self.search_turn_goal = None
        if self.search_yaw_index < len(
            self.config.search_yaw_offsets
        ):
            self.observe_resume = True
            self.set_phase(Phase.SEARCH_ROTATE)
            return

        next_area = self.search_area_index + 1
        if next_area < len(self.search_areas):
            self.start_search_area(next_area)
            return

        self.navigator.stop()
        self.publish_ready(False, "all_search_views_exhausted")
        self.set_phase(Phase.SAFE_STOP)

    def handle_plan(self):
        if self.robot_pose is None:
            self.publish_ready(False, "waiting_odom")
            return

        if self.latest_detection_scene is None:
            self.publish_ready(False, "waiting_grounded_sam_detections")
            return

        self.active_task = self.bind_current_task()
        self.approach_retry_count = 0

        if (
            target_is_detected(self.active_task)
            and self.target_pose_valid_for_surface(
                self.active_task.get("target") or {}
            )
        ):
            self.plan_approach_to_current_target(
                "target_detected_navigate_to_pregrasp"
            )
            return

        self.search_areas = choose_search_areas(
            self.active_task
        )
        self.start_search_area(0)

    def handle_navigate(self):
        if self.robot_pose is None or self.current_goal is None:
            self.set_phase(Phase.SAFE_STOP)
            return
        if (
            self.navigation_mode == "approach"
            and self.current_dock_controller is not None
        ):
            progress = self.current_dock_controller.update(
                self.robot_pose,
                time.monotonic(),
            )
            self.last_dock_progress = progress

            msg = Twist()
            msg.linear.x = float(progress.command.linear)
            msg.angular.z = float(progress.command.angular)
            self.cmd_vel_pub.publish(msg)

            if progress.completed:
                self.navigator.stop()
                self.arrived_scene_time = self.latest_scene_time
                self.arrived_detection_time = self.latest_detection_time
                self.set_phase(Phase.SCAN_AND_REOBSERVE)
                return

            return

        # 没有 A* route 时，保留原来的 SimpleCmdVelNavigator 逻辑
        if self.navigator.reached(
            self.robot_pose,
            self.current_goal,
        ):
            self.navigator.stop()

            # 记录到达前的旧 scene 时间。
            # 后面必须等到达后的新视觉结果，不能拿旧图像做视觉伺服。
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time

            if self.navigation_mode == "search":
                self.search_center_yaw = self.current_goal.yaw
                self.search_turn_goal = None
                self.set_phase(Phase.SEARCH_ROTATE)
            else:
                self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        self.navigator.step(
            self.robot_pose,
            self.current_goal,
        )

    def handle_search_rotate(self):
        if self.robot_pose is None or self.search_center_yaw is None:
            self.publish_ready(False, "search_rotate_missing_pose")
            self.set_phase(Phase.SAFE_STOP)
            return

        confirmed_task = self.get_confirmed_search_task()
        if confirmed_task is not None:
            self.active_task = confirmed_task
            self.approach_retry_count = 0
            self.plan_approach_to_current_target(
                "target_found_during_search_replan_approach"
            )
            return

        offset = self.config.search_yaw_offsets[
            self.search_yaw_index
        ]
        target_yaw = self.search_center_yaw + float(offset)
        if self.search_turn_goal is None:
            self.search_turn_goal = Pose2D(
                x=self.robot_pose.x,
                y=self.robot_pose.y,
                yaw=target_yaw,
            )
        turn_goal = self.search_turn_goal

        if self.navigator.reached(self.robot_pose, turn_goal):
            self.navigator.stop()
            self.arrived_detection_time = (
                self.latest_detection_time
            )
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        if (
            time.monotonic() - self.phase_start_time
            > self.config.search_turn_timeout_sec
        ):
            self.navigator.stop()
            self.publish_ready(False, "search_rotate_timeout")
            self.advance_search()
            return

        self.navigator.step(self.robot_pose, turn_goal)

    def handle_scan_and_reobserve(self):
        self.navigator.stop()

        if self.latest_scene is None:
            self.publish_ready(False, "waiting_scene_before_observe")
            return

        previous_task = self.active_task
        rebound_task = self.bind_current_task()
        if rebound_task is None:
            self.publish_ready(False, "no_active_task_before_observe")
            return

        if self.navigation_mode == "approach":
            target = rebound_task.get("target") or {}
            if (
                target_is_detected(rebound_task)
                and self.target_pose_valid_for_surface(target)
            ):
                target = self.apply_pregrasp_metadata(target)
                self.active_task = dict(rebound_task)
                self.active_task["target"] = target
                self.approach_target_snapshot = dict(target)
            elif (
                self.approach_target_snapshot is not None
                and previous_task is not None
            ):
                self.active_task = dict(previous_task)
                self.active_task["target"] = dict(
                    self.approach_target_snapshot
                )
            else:
                self.active_task = rebound_task
        else:
            self.active_task = rebound_task

        confirmed_task = self.get_confirmed_search_task()
        if self.navigation_mode == "search" and confirmed_task is not None:
            self.active_task = confirmed_task
            self.approach_retry_count = 0
            self.plan_approach_to_current_target(
                "target_found_before_scan_replan_approach"
            )
            return

        target = self.active_task.get("target") or {}
        support_surface = target.get("support_surface")

        if self.navigation_mode == "search":
            area_name = self.search_areas[
                self.search_area_index
            ]
            support_surface = (
                "shelf"
                if area_name == "shelf_front"
                else "table"
            )
        elif support_surface is None:
            support_surface = "table"

        self.observer.start(
            support_surface,
            resume=self.observe_resume,
            mode=self.navigation_mode,
            target=target,
        )
        self.observe_resume = False
        self.last_observer_view_name = None
        self.set_phase(Phase.ACTIVE_OBSERVE)

    def handle_active_observe(self):
        self.navigator.stop()

        observe_done = self.observer.step()

        if (
            self.observer.current_view_name is not None
            and self.observer.current_view_name
            != self.last_observer_view_name
        ):
            self.last_observer_view_name = (
                self.observer.current_view_name
            )
            self.arrived_detection_time = (
                self.latest_detection_time
            )

        # 必须等待当前观察位姿产生的新GroundingDINO检测。
        has_fresh_scene = (
            self.latest_detection_time
            > self.arrived_detection_time
        )
        fresh_target_detected = False

        if has_fresh_scene:
            rebound_task = self.bind_current_task()
            rebound_target = (
                rebound_task.get("target") if rebound_task else {}
            ) or {}
            rebound_is_valid = (
                target_is_detected(rebound_task)
                and self.target_pose_valid_for_surface(rebound_target)
            )
            if (
                self.navigation_mode == "approach"
                and not rebound_is_valid
                and self.approach_target_snapshot is not None
            ):
                self.active_task = dict(self.active_task or {})
                self.active_task["target"] = dict(
                    self.approach_target_snapshot
                )
            else:
                self.active_task = rebound_task
                if (
                    self.navigation_mode == "approach"
                    and rebound_is_valid
                ):
                    self.approach_target_snapshot = dict(
                        (self.active_task or {}).get("target") or {}
                    )
                    self.remember_servo_target(
                        (self.active_task or {}).get("target") or {}
                    )

            active_target = (
                (self.active_task or {}).get("target") or {}
            )
            fresh_target_detected = (
                target_is_detected(self.active_task)
                and self.target_pose_valid_for_surface(active_target)
            )

            # 搜索航点只负责发现目标。发现后必须依据最新Pose3D
            # 重新生成目标专属预抓取点，不能直接在搜索航点做伺服。
            confirmed_task = self.get_confirmed_search_task()
            if self.navigation_mode == "search" and confirmed_task is not None:
                self.active_task = confirmed_task
                self.observer.stop()
                self.observe_resume = True
                self.approach_retry_count = 0
                self.plan_approach_to_current_target(
                    "target_found_during_body_search_replan_approach"
                )
                return

            if (
                self.navigation_mode == "approach"
                and fresh_target_detected
            ):
                target = self.active_task.get("target") or {}
                self.observer.stop()
                self.visual_servo.reset()
                self.remember_servo_target(target)
                self.publish_ready(
                    False,
                    "target_visible_enter_visual_servo",
                )
                self.set_phase(Phase.VISUAL_SERVO)
                return

        if fresh_target_detected:
            self.publish_ready(
                False,
                "target_detected_but_not_servo_usable_keep_observing",
            )

        elapsed = time.monotonic() - self.phase_start_time

        if (
            observe_done
            or elapsed > self.config.active_observe_timeout_sec
        ):
            self.observer.stop()

            if self.navigation_mode == "search":
                self.publish_ready(
                    False,
                    "search_view_exhausted_rotate_to_next_view",
                )
                self.advance_search()
                return

            if (
                fresh_target_detected
                and self.approach_retry_count
                < self.config.max_approach_retries
            ):
                self.approach_retry_count += 1
                self.observe_resume = True
                self.plan_approach_to_current_target(
                    "target_cropped_backoff_and_replan_approach"
                )
                return

            reason = (
                "approach_views_exhausted_target_not_servo_usable"
                if fresh_target_detected
                else "approach_views_exhausted_target_not_found"
            )
            self.publish_ready(False, reason)
            self.set_phase(Phase.SAFE_STOP)
            return

        self.publish_ready(
            False,
            "active_observing",
        )
    def handle_visual_servo(self):
        if self.latest_detection_scene is None:
            locked_target = self.close_range_lock_target()
            if locked_target is None:
                self.visual_servo.stop()
                self.publish_ready(False, "visual_servo_waiting_scene")
                return

            self.active_task = dict(
                self.active_task
                or get_active_task(self.latest_scene or {})
                or {}
            )
            self.active_task["target"] = locked_target
            target = locked_target
            close_range_locked = True
            current_target_valid = False
        else:
            target = None
            close_range_locked = False
            current_target_valid = False

        if not close_range_locked:
            scene_age = time.monotonic() - self.latest_detection_time
            if scene_age > self.config.servo_scene_max_age_sec:
                locked_target = self.close_range_lock_target()
                if locked_target is None:
                    self.visual_servo.stop()
                    self.publish_ready(False, "visual_servo_scene_too_old")
                    self.arrived_scene_time = self.latest_scene_time
                    self.arrived_detection_time = self.latest_detection_time
                    self.observe_resume = True
                    self.set_phase(Phase.SCAN_AND_REOBSERVE)
                    return

                self.active_task = dict(
                    self.active_task
                    or get_active_task(self.latest_scene or {})
                    or {}
                )
                self.active_task["target"] = locked_target
                target = locked_target
                close_range_locked = True

        if not close_range_locked:
            base_task = self.active_task or get_active_task(
                self.latest_scene
            )
            self.active_task = bind_task_target_from_objects(
                self.latest_detection_scene,
                base_task,
            )
            target = (self.active_task or {}).get("target") or {}
            current_target_valid = (
                target_is_detected(self.active_task)
                and self.target_pose_valid_for_surface(target)
            )

            if not current_target_valid:
                recent_target = self.recent_servo_target()
                locked_target = self.close_range_lock_target(
                    recent_target
                )
                if locked_target is not None:
                    self.active_task = dict(base_task or {})
                    self.active_task["target"] = locked_target
                    target = locked_target
                    close_range_locked = True
                elif recent_target is None:
                    self.visual_servo.stop()
                    self.publish_ready(
                        False,
                        "visual_servo_waiting_fresh_target_detection",
                    )
                    self.arrived_scene_time = self.latest_scene_time
                    self.arrived_detection_time = self.latest_detection_time
                    self.observe_resume = True
                    self.set_phase(Phase.SCAN_AND_REOBSERVE)
                    return
                else:
                    self.active_task = dict(base_task or {})
                    self.active_task["target"] = recent_target
                    target = recent_target

            else:
                target = self.apply_pregrasp_metadata(target)
                self.active_task = dict(self.active_task)
                self.active_task["target"] = target
                self.remember_servo_target(target)

        target = self.restore_motion_metadata(target)
        target = self.apply_pregrasp_metadata(target)
        self.active_task = dict(self.active_task or {})
        self.active_task["target"] = target

        if close_range_locked:
            self.visual_servo.stop(reset_stability=False)
            if self.visual_servo.stable_frames < self.config.servo_stable_frames:
                self.visual_servo.stable_frames += 1
            servo_state = (
                ServoState.ALIGNED
                if (
                    self.visual_servo.stable_frames
                    >= self.config.servo_stable_frames
                )
                else ServoState.STABILIZING
            )
        else:
            observation_id = (
                self.latest_detection_scene.get("observation_seq"),
            )
            if not current_target_valid:
                observation_id = self.servo_target_observation_id
            servo_state = self.visual_servo.step_from_target(
                target,
                self.robot_pose,
                observation_id=observation_id,
            )

        if servo_state == ServoState.REOBSERVE:
            self.publish_ready(False, "target_not_fully_visible_reobserve")
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time
            self.observe_resume = True
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        if servo_state != ServoState.ALIGNED:
            reason = f"visual_servo_{servo_state}"
            if close_range_locked:
                reason = f"visual_servo_close_range_locked_{servo_state}"
            self.publish_ready(False, reason)

            elapsed = time.monotonic() - self.phase_start_time
            if elapsed > self.config.visual_servo_timeout_sec:
                self.visual_servo.stop()
                self.publish_ready(False, "visual_servo_timeout_reobserve")
                self.arrived_scene_time = self.latest_scene_time
                self.arrived_detection_time = self.latest_detection_time
                self.observe_resume = True
                self.set_phase(Phase.SCAN_AND_REOBSERVE)

            return

        # 只有走到这里，才说明底盘视觉伺服已经真正对准
        if not self.grasp_clearance_ready(target):
            self.publish_ready(False, "aligned_but_clearance_invalid_replan")

            if (
                self.target_is_table_target(target)
                and self.approach_retry_count
                < self.config.max_approach_retries
            ):
                self.approach_retry_count += 1
                self.plan_approach_to_current_target(
                    "clearance_invalid_replan_pregrasp"
                )
                return

            self.publish_ready(False, "aligned_but_clearance_invalid")
            self.set_phase(Phase.SAFE_STOP)
            return

        self.pregrasp_stable_frames = 0
        self.pregrasp_lost_frames = 0
        self.pregrasp_last_safe_spine = self.spine_position
        self.pregrasp_last_safe_head_pitch = self.head_pitch_position

        current_spine = (
            float(self.spine_position)
            if self.spine_position is not None
            else float(self.expected_table_spine(target) or 0.18)
        )

        self.pregrasp_start_spine = current_spine
        estimated_spine = self.estimate_pregrasp_spine_from_target(
            target
        )

        if estimated_spine is None:
            estimated_spine = (
                current_spine
                + self.config.pregrasp_spine_delta_after_servo
            )

        self.pregrasp_target_spine = max(
            self.config.table_pregrasp_spine_min,
            min(
                self.config.table_pregrasp_spine_max,
                estimated_spine,
            ),
        )

        self.pregrasp_target_head_pitch = (
            self.config.pregrasp_final_head_pitch
        )

        self.ready_robot_pose = Pose2D(
            x=self.robot_pose.x,
            y=self.robot_pose.y,
            yaw=self.robot_pose.yaw,
        )

        if (
            self.config.pregrasp_adjust_enable
            and self.target_is_table_target(target)
        ):
            self.publish_ready(False, "enter_pregrasp_adjust")
            self.set_phase(Phase.PREGRASP_ADJUST)
        else:
            self.ready_task_id = self.active_task.get("task_id")
            self.ready_target_snapshot = dict(target)
            self.set_phase(Phase.READY_FOR_GRASP)
            self.publish_ready(
                True,
                "target_aligned_for_grasp_distance_valid_and_stable",
            )

        return

    def handle_pregrasp_adjust(self):
        self.navigator.stop()
        self.cmd_vel_pub.publish(Twist())

        close_range_locked = False
        target = None
        allow_close_lock_pregrasp = bool(
            getattr(
                self.config,
                "close_range_lock_allow_pregrasp_without_detection",
                True,
            )
        )

        if self.latest_detection_scene is None:
            target = self.close_range_lock_target()
            if target is None or not allow_close_lock_pregrasp:
                self.publish_ready(False, "pregrasp_waiting_detection")
                return
            close_range_locked = True
            self.active_task = dict(
                self.active_task
                or get_active_task(self.latest_scene or {})
                or {}
            )
            self.active_task["target"] = target

        if not close_range_locked:
            scene_age = time.monotonic() - self.latest_detection_time
            if scene_age > self.config.servo_scene_max_age_sec:
                target = self.close_range_lock_target()
                if target is None or not allow_close_lock_pregrasp:
                    self.publish_ready(False, "pregrasp_detection_too_old")
                    self.observe_resume = True
                    self.set_phase(Phase.SCAN_AND_REOBSERVE)
                    return
                close_range_locked = True
                self.active_task = dict(
                    self.active_task
                    or get_active_task(self.latest_scene or {})
                    or {}
                )
                self.active_task["target"] = target

        if not close_range_locked:
            base_task = self.active_task or get_active_task(
                self.latest_scene
            )
            self.active_task = bind_task_target_from_objects(
                self.latest_detection_scene,
                base_task,
            )
            target = (self.active_task or {}).get("target") or {}

            if not (
                target_is_detected(self.active_task)
                and self.target_pose_valid_for_surface(target)
            ):
                locked_target = self.close_range_lock_target(target)
                if (
                    locked_target is not None
                    and allow_close_lock_pregrasp
                ):
                    close_range_locked = True
                    target = locked_target
                    self.active_task = dict(base_task or {})
                    self.active_task["target"] = target
                else:
                    self.pregrasp_lost_frames += 1
                    self.publish_ready(False, "pregrasp_target_lost")
                    if self.pregrasp_lost_frames >= self.config.pregrasp_lost_max_frames:
                        self.observe_resume = True
                        self.set_phase(Phase.SCAN_AND_REOBSERVE)

                    return
        self.pregrasp_lost_frames = 0
        target = self.restore_motion_metadata(target)
        target = self.apply_pregrasp_metadata(target)
        if close_range_locked:
            target["close_range_locked"] = True
        self.active_task = dict(self.active_task or {})
        self.active_task["target"] = target

        centroid_uv = target.get("centroid_uv") or []
        box_xyxy = target.get("box_xyxy") or []

        target_u = float(self.config.pregrasp_target_u)
        target_v = float(self.config.pregrasp_target_v)

        if len(centroid_uv) < 2 or len(box_xyxy) < 4:
            if close_range_locked and allow_close_lock_pregrasp:
                u = target_u
                v = target_v
                x1 = 0.0
                y1 = 0.0
                x2 = float(self.config.image_width)
                y2 = float(self.config.image_height)
            else:
                self.publish_ready(False, "pregrasp_target_no_uv")
                self.observe_resume = True
                self.set_phase(Phase.SCAN_AND_REOBSERVE)
                return
        else:
            u = float(centroid_uv[0])
            v = float(centroid_uv[1])
            x1, y1, x2, y2 = [float(value) for value in box_xyxy]

        pregrasp_margin_px = 30.0
        pregrasp_bottom_margin_px = 25.0

        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)
        bbox_area_ratio = (
            bbox_width * bbox_height
        ) / float(self.config.image_width * self.config.image_height)

        full_object_visible = (
            x1 >= pregrasp_margin_px
            and y1 >= pregrasp_margin_px
            and x2 <= self.config.image_width - pregrasp_margin_px
            and y2 <= self.config.image_height - pregrasp_bottom_margin_px
        )

        box_size_reasonable = (
            bbox_area_ratio >= 0.01
        )

        uv_safe = (
            abs(u - target_u) <= self.config.pregrasp_u_tolerance
            and abs(v - target_v) <= self.config.pregrasp_v_tolerance
        )

        pregrasp_view_ready = (
            full_object_visible
            and box_size_reasonable
            and uv_safe
        )
        if close_range_locked and allow_close_lock_pregrasp:
            pregrasp_view_ready = True

        expected_spine = self.pregrasp_target_spine
        if expected_spine is None:
            expected_spine = self.expected_table_spine(target)
        if expected_spine is None:
            expected_spine = self.spine_position

        target_head = self.pregrasp_target_head_pitch
        if target_head is None:
            target_head = self.config.pregrasp_final_head_pitch

        current_spine = (
                float(self.spine_position)
                if self.spine_position is not None
                else float(expected_spine)
            )

        current_head = (
                float(self.head_pitch_position)
                if self.head_pitch_position is not None
                else float(self.config.table_pregrasp_head_pitch)
            )

            # 腰部目标：朝 expected_spine 小步移动，避免一下子丢视野。
        spine_error = float(expected_spine) - current_spine
        spine_step = max(
                -self.config.pregrasp_spine_step,
                min(self.config.pregrasp_spine_step, spine_error),
            )
        next_spine = current_spine + spine_step

            # 头部目标：如果目标在画面偏下，头回正；如果目标偏上，稍微低头。
            # 注意：你当前系统里 pitch 越接近 0 越回正，-0.65 更俯视。
        v_error = v - target_v
        head_error = float(target_head) - current_head
        head_step = max(
            -self.config.pregrasp_head_pitch_step,
            min(self.config.pregrasp_head_pitch_step, head_error),
        )
        next_head = current_head + head_step

        # 目标在画面偏下时，头部更回正一点，防止腰下降后丢视野。
        if v_error > self.config.pregrasp_v_tolerance * 0.4:
            next_head += self.config.pregrasp_head_pitch_step
        elif v_error < -self.config.pregrasp_v_tolerance * 0.4:
            next_head -= self.config.pregrasp_head_pitch_step

        next_head = max(
                self.config.pregrasp_head_pitch_min,
                min(self.config.pregrasp_head_pitch_max, next_head),
            )

        self.observer.publish_observe_pose(
                head_yaw=0.0,
                head_pitch=next_head,
                spine=next_spine,
            )

        if pregrasp_view_ready:
            self.pregrasp_last_safe_spine = next_spine
            self.pregrasp_last_safe_head_pitch = next_head

        elapsed = time.monotonic() - self.phase_start_time

        spine_ready = (
            abs(float(expected_spine) - current_spine)
            <= self.config.pregrasp_spine_target_tolerance
        )
        head_ready = (
            abs(float(target_head) - current_head)
            <= self.config.pregrasp_head_pitch_tolerance
        )
        min_time_ready = (
            elapsed >= self.config.pregrasp_adjust_min_sec
        )

        view_and_grasp_ready = (
            min_time_ready
            and pregrasp_view_ready
            and self.grasp_clearance_ready(target)
        )

        if view_and_grasp_ready:
            self.pregrasp_stable_frames += 1
        else:
            self.pregrasp_stable_frames = 0

        if (
            self.pregrasp_stable_frames
            >= self.config.pregrasp_adjust_stable_frames
        ):
            self.ready_task_id = self.active_task.get("task_id")
            self.ready_target_snapshot = dict(target)
            self.set_phase(Phase.READY_FOR_GRASP)
            reason = "pregrasp_head_spine_adjusted_target_visible"
            if close_range_locked:
                reason = "pregrasp_close_range_locked_ready"
            self.publish_ready(
                True,
                reason,
            )
            return

        if elapsed > self.config.pregrasp_adjust_timeout_sec:
            if view_and_grasp_ready:
                self.publish_ready(
                    False,
                    "pregrasp_view_ready_waiting_stable_frames",
                )
            else:
                self.publish_ready(
                    False,
                    "pregrasp_adjust_timeout_view_not_ready",
                )
            return

        self.publish_ready(False, "pregrasp_adjusting_head_and_spine")

    def handle_ready_for_grasp(self):
        self.navigator.stop()
        self.cmd_vel_pub.publish(Twist())

        if self.grasp_handoff_active:
            self.visual_servo.stop(reset_stability=False)
            if self.active_task is None:
                self.active_task = {}
            else:
                self.active_task = dict(self.active_task)
            self.active_task["target"] = dict(
                self.ready_target_snapshot or {}
            )
            self.publish_ready(True, "grasp_handoff_active_ready_kept")
            return

        current_task = get_active_task(self.latest_scene or {})
        current_task_id = (
            current_task.get("task_id")
            if current_task is not None
            else None
        )
        if current_task_id != self.ready_task_id:
            self.ready_task_id = None
            self.ready_target_snapshot = None
            self.ready_robot_pose = None
            self.visual_servo.reset()
            self.set_phase(Phase.SELECT_TASK)
            return

        if self.robot_pose is None or self.ready_robot_pose is None:
            self.publish_ready(False, "ready_latch_missing_odom")
            self.set_phase(Phase.SAFE_STOP)
            return

        ready_target = self.ready_target_snapshot or {}

        if not self.target_pose_valid_for_surface(ready_target):
            self.ready_task_id = None
            self.ready_target_snapshot = None
            self.ready_robot_pose = None
            self.visual_servo.reset()
            self.publish_ready(False, "ready_latch_target_pose_invalid")
            self.set_phase(Phase.SAFE_STOP)
            return

        if not self.grasp_clearance_ready(ready_target):
            self.ready_task_id = None
            self.ready_target_snapshot = None
            self.ready_robot_pose = None
            self.visual_servo.reset()
            self.publish_ready(False, "ready_latch_clearance_invalid")
            self.set_phase(Phase.SAFE_STOP)
            return

        xy_drift = math.hypot(
            self.robot_pose.x - self.ready_robot_pose.x,
            self.robot_pose.y - self.ready_robot_pose.y,
        )
        yaw_delta = self.robot_pose.yaw - self.ready_robot_pose.yaw
        yaw_drift = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
        if (
            xy_drift > self.config.ready_latch_xy_tolerance
            or yaw_drift > self.config.ready_latch_yaw_tolerance
        ):
            self.ready_task_id = None
            self.ready_target_snapshot = None
            self.ready_robot_pose = None
            self.visual_servo.reset()
            self.publish_ready(False, "ready_latch_robot_pose_changed")
            self.observe_resume = True
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        if self.active_task is None:
            self.active_task = dict(current_task or {})
        else:
            self.active_task = dict(self.active_task)
        self.active_task["target"] = dict(
            self.ready_target_snapshot or {}
        )
        self.publish_ready(True, "ready_latched_waiting_grasp")


def main():
    rclpy.init()
    node = ActionNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.navigator.stop()
    finally:
        node.navigator.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
