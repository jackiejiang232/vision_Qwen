#!/usr/bin/env python3
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from scipy.spatial.transform import Rotation
from .active_observer import ActiveObserver

from .action_config import CONFIG
from .goal_builder import (
    Pose2D,
    build_approach_goal_from_target,
    build_search_goal,
)
from .scene_reader import (
    bind_task_target_from_objects,
    choose_search_area,
    get_active_task,
    parse_scene_message,
    target_is_detected,
    target_is_visible_for_servo,
)
from .simple_nav import SimpleCmdVelNavigator
from .visual_servo import ServoState, VisualServo


class Phase:
    WAIT_SCENE = "wait_scene"
    SELECT_TASK = "select_task"
    PLAN = "plan"
    NAVIGATE = "navigate"
    SCAN_AND_REOBSERVE = "scan_and_reobserve"
    ACTIVE_OBSERVE = "active_observe"
    VISUAL_SERVO = "visual_servo"
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
        self.active_task = None
        self.current_goal = None
        self.robot_pose = None
        self.phase_start_time = time.monotonic()
        self.last_ready_publish_time = 0.0

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
            self.latest_detection_scene = {
                "source_stamp_sec": source_stamp.get("sec"),
                "source_stamp_nanosec": source_stamp.get("nanosec"),
                "objects": payload.get("detections") or [],
            }
            self.latest_detection_time = time.monotonic()
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

    def publish_ready(self, ready, reason):
        task = self.active_task or {}
        target = task.get("target") or {}

        semantic_age = None
        if self.latest_scene_time > 0.0:
            semantic_age = time.monotonic() - self.latest_scene_time

        detection_age = None
        if self.latest_detection_time > 0.0:
            detection_age = time.monotonic() - self.latest_detection_time

        payload = {
            "ready_for_grasp": bool(ready),
            "phase": self.phase,
            "task_id": task.get("task_id"),
            "target_object_id": target.get("object_id"),
            "target_label": target.get("label"),
            "target_pose_world": target.get("pose_world"),
            "target_box_xyxy": target.get("box_xyxy"),
            "target_centroid_uv": target.get("centroid_uv"),
            "base_goal_pose": (
                self.current_goal.__dict__
                if self.current_goal is not None
                else None
            ),
            "has_scene": self.latest_scene is not None,
            "has_odom": self.robot_pose is not None,
            "semantic_age_sec": semantic_age,
            "detection_age_sec": detection_age,
            "reason": reason,
            "observer_view": (
                self.observer.current_view_name
                if hasattr(self, "observer")
                else None
            ),
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.ready_pub.publish(msg)
        self.last_ready_publish_time = time.monotonic()

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
        elif self.phase == Phase.SCAN_AND_REOBSERVE:
            reason = "preparing_active_observe"
        elif self.phase == Phase.ACTIVE_OBSERVE:
            reason = "active_observing"
        elif self.phase == Phase.VISUAL_SERVO:
            reason = "visual_servo_aligning"
        elif self.phase == Phase.READY_FOR_GRASP:
            reason = "target_detected_and_base_aligned"
        else:
            reason = "safe_stop"

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
        elif self.phase == Phase.SCAN_AND_REOBSERVE:
            self.handle_scan_and_reobserve()
        elif self.phase == Phase.ACTIVE_OBSERVE:
            self.handle_active_observe()
        elif self.phase == Phase.VISUAL_SERVO:
            self.handle_visual_servo()
        elif self.phase == Phase.READY_FOR_GRASP:
            self.handle_ready_for_grasp()
        elif self.phase == Phase.SAFE_STOP:
            self.navigator.stop()

        self.publish_ready_heartbeat()

    def handle_wait_scene(self):
        self.navigator.stop()
        if self.latest_scene is not None:
            self.set_phase(Phase.SELECT_TASK)

    def handle_select_task(self):
        self.active_task = bind_task_target_from_objects(
            self.latest_scene,
            get_active_task(self.latest_scene),
        )
        if self.active_task is None:
            self.publish_ready(False, "no_active_task")
            self.set_phase(Phase.WAIT_SCENE)
            return

        self.set_phase(Phase.PLAN)

    def handle_plan(self):
        if self.robot_pose is None:
            self.publish_ready(False, "waiting_odom")
            return

        if target_is_detected(self.active_task):
            target = self.active_task["target"]
            self.current_goal = build_approach_goal_from_target(
                target,
                self.config,
                self.robot_pose,
            )
            self.publish_ready(False, "target_detected_navigate_to_pregrasp")
        else:
            area = choose_search_area(self.active_task)
            self.current_goal = build_search_goal(area)
            self.publish_ready(False, f"target_not_detected_search_{area}")

        self.set_phase(Phase.NAVIGATE)

    def handle_navigate(self):
        if self.robot_pose is None or self.current_goal is None:
            self.set_phase(Phase.SAFE_STOP)
            return

        if self.navigator.reached(
            self.robot_pose,
            self.current_goal,
        ):
            self.navigator.stop()

            # 记录到达前的旧 scene 时间。
            # 后面必须等到达后的新视觉结果，不能拿旧图像做视觉伺服。
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time

            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        self.navigator.step(
            self.robot_pose,
            self.current_goal,
        )

    def handle_scan_and_reobserve(self):
        self.navigator.stop()

        if self.latest_scene is None:
            self.publish_ready(False, "waiting_scene_before_observe")
            return

        perception_scene = (
            self.latest_detection_scene or self.latest_scene
        )
        self.active_task = bind_task_target_from_objects(
            perception_scene,
            get_active_task(self.latest_scene),
        )

        if self.active_task is None:
            self.publish_ready(False, "no_active_task_before_observe")
            return

        target = self.active_task.get("target") or {}
        support_surface = target.get("support_surface")

        place_goal = self.active_task.get("place_goal") or {}
        if support_surface is None:
            if place_goal.get("type") in (
                "shelf_layer",
                "relative_position",
            ):
                support_surface = "shelf"
            else:
                support_surface = "table"

        self.observer.start(support_surface)
        self.last_observer_view_name = None
        self.set_phase(Phase.ACTIVE_OBSERVE)

    def handle_active_observe(self):
        self.navigator.stop()

        observe_done = self.observer.step()

        if (
            self.observer.current_view_name
            != self.last_observer_view_name
        ):
            self.last_observer_view_name = (
                self.observer.current_view_name
            )
            self.arrived_detection_time = (
                self.latest_detection_time
            )

        # 必须等待到达后新产生的GroundingDINO检测。
        has_fresh_scene = (
            self.latest_detection_time
            > self.arrived_detection_time
        )

        if has_fresh_scene:
            self.active_task = bind_task_target_from_objects(
                self.latest_detection_scene,
                get_active_task(self.latest_scene),
            )

            if target_is_visible_for_servo(
                self.latest_detection_scene,
                self.active_task,
                self.config,
            ):
                self.observer.stop()
                self.publish_ready(
                    False,
                    "target_visible_and_ready_for_visual_servo",
                )
                self.set_phase(Phase.VISUAL_SERVO)
                return
        if target_is_detected(self.active_task):
            self.publish_ready(
                False,
                "target_detected_but_not_servo_usable_keep_observing",
            )

        elapsed = time.monotonic() - self.phase_start_time

        if observe_done:
            self.publish_ready(
                False,
                "active_observe_done_target_not_found",
            )
            self.set_phase(Phase.SAFE_STOP)
            return

        if elapsed > self.config.active_observe_timeout_sec:
            self.publish_ready(
                False,
                "active_observe_timeout_target_not_found",
            )
            self.set_phase(Phase.SAFE_STOP)
            return

        self.publish_ready(
            False,
            "active_observing",
        )
    def handle_visual_servo(self):
        if self.latest_detection_scene is None:
            self.visual_servo.stop()
            self.publish_ready(False, "visual_servo_waiting_scene")
            return

        scene_age = time.monotonic() - self.latest_detection_time
        if scene_age > self.config.servo_scene_max_age_sec:
            self.visual_servo.stop()
            self.publish_ready(False, "visual_servo_scene_too_old")
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        self.active_task = bind_task_target_from_objects(
            self.latest_detection_scene,
            get_active_task(self.latest_scene),
        )
        target = (self.active_task or {}).get("target") or {}

        if not target_is_detected(self.active_task):
            self.visual_servo.stop()
            self.publish_ready(False, "visual_servo_target_lost")
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        observation_id = (
            self.latest_detection_scene.get("source_stamp_sec"),
            self.latest_detection_scene.get("source_stamp_nanosec"),
        )
        servo_state = self.visual_servo.step_from_target(
            target,
            self.robot_pose,
            observation_id=observation_id,
        )

        if servo_state == ServoState.ALIGNED:
            self.publish_ready(False, "visual_servo_stable_alignment_confirmed")
            self.set_phase(Phase.READY_FOR_GRASP)
            return

        if servo_state == ServoState.REOBSERVE:
            self.publish_ready(False, "target_not_fully_visible_reobserve")
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        self.publish_ready(False, f"visual_servo_{servo_state}")

        elapsed = time.monotonic() - self.phase_start_time
        if elapsed > self.config.visual_servo_timeout_sec:
            self.visual_servo.stop()
            self.publish_ready(False, "visual_servo_timeout_reobserve")
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time
            self.set_phase(Phase.SCAN_AND_REOBSERVE)

    def handle_ready_for_grasp(self):
        self.navigator.stop()

        if (
            self.latest_detection_scene is None
            or self.latest_scene is None
            or self.robot_pose is None
        ):
            self.publish_ready(False, "ready_validation_missing_scene_or_odom")
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        scene_age = time.monotonic() - self.latest_detection_time
        self.active_task = bind_task_target_from_objects(
            self.latest_detection_scene,
            get_active_task(self.latest_scene),
        )
        target = (self.active_task or {}).get("target") or {}

        if (
            scene_age > self.config.servo_scene_max_age_sec
            or not target_is_detected(self.active_task)
            or not self.visual_servo.ready_now(target, self.robot_pose)
        ):
            self.visual_servo.stop()
            self.publish_ready(False, "ready_validation_failed_reobserve")
            self.arrived_scene_time = self.latest_scene_time
            self.arrived_detection_time = self.latest_detection_time
            self.set_phase(Phase.SCAN_AND_REOBSERVE)
            return

        self.publish_ready(True, "target_centered_distance_valid_and_stable")


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