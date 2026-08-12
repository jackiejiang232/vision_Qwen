#!/usr/bin/env python3
from dataclasses import replace
import json
import math
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header, String

from third_party.dg202612.contracts import (
    CameraId,
    CameraObservation,
    ExecutionPhase,
    GraspEvidence,
    RobotTargets,
)
from third_party.dg202612.executor import ExecutorEvent, MinimumPickExecutor
from third_party.dg202612.navigation import PathPlan
from third_party.dg202612.ros_adapter import legacy_control_vector

from .action_config import CONFIG
from .grasp_state_builder import (
    make_solver,
    measured_targets,
    perception_limits,
    robot_state_from_ros,
    scene_goal_from_ready,
    table_side_hug_profile,
)


class AlreadyDockedPlanner:
    def plan(self, start, goal):
        return PathPlan(
            waypoints=(start, goal),
            grid_cells=(),
            cost=math.hypot(goal.x - start.x, goal.y - start.y),
        )


class GraspExecutorNode(Node):
    def __init__(self):
        super().__init__("grasp_executor_node")
        self.config = CONFIG
        self.latest_ready = None
        self.latest_ready_rx = 0.0
        self.latest_odom = None
        self.latest_joints = None
        self.pick_executor = None
        self.scene = None
        self.goal = None
        self.pending_frames = []
        self.current_control_target = None
        self.reset_target = None
        self.reset_started_at = 0.0
        self.reset_stable_since = 0.0
        self.state = "WAIT_READY"
        self.manual_sim_pick = False
        self.last_ready_log_at = 0.0
        self.last_ready_target = None
        self.last_idle_request_log_at = 0.0

        self.create_subscription(String, self.config.ready_for_grasp_topic, self.on_ready, 10)
        self.create_subscription(String, self.config.grasp_command_topic, self.on_command, 10)
        self.create_subscription(Odometry, self.config.odom_topic, self.on_odom, 10)
        self.create_subscription(JointState, self.config.joint_states_topic, self.on_joint, 10)

        self.control_pub = self.create_publisher(Float64MultiArray, self.config.dg_control_request_topic, 10)
        self.heartbeat_pub = self.create_publisher(Header, self.config.dg_control_heartbeat_topic, 10)
        self.status_pub = self.create_publisher(String, self.config.grasp_status_topic, 10)
        self.create_timer(1.0 / float(self.config.grasp_control_rate_hz), self.tick)
        self.get_logger().info("grasp executor started")

    def on_ready(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.publish_status("bad_ready_json", error=str(exc))
            return
        if not payload.get("ready_for_grasp"):
            return
        self.latest_ready = payload
        self.latest_ready_rx = time.time()
        target = payload.get("target_object_id")
        now = time.time()
        if target != self.last_ready_target or now - self.last_ready_log_at > 1.0:
            self.last_ready_target = target
            self.last_ready_log_at = now
            self.publish_status(
                "ready_received",
                task_id=payload.get("task_id"),
                target=target,
            )
        if self.config.grasp_auto_start:
            self.start_grasp()

    def on_odom(self, message):
        self.latest_odom = message

    def on_joint(self, message):
        self.latest_joints = message

    def on_command(self, message):
        command = message.data.strip().lower()
        self.publish_status("command_received", command=command)
        if command == "start":
            self.start_grasp()
        elif command == "pregrasp_done":
            if self.advance_manual_sim_pick("pregrasp_done"):
                return
            self.advance(ExecutorEvent.PREGRASP_REACHED)
        elif command == "contact_done":
            if self.advance_manual_sim_pick("contact_done"):
                return
            self.manual_contact_done()
        elif command == "hold_done":
            if self.advance_manual_sim_pick("hold_done"):
                return
            self.manual_hold_done()
        elif command == "lift_done":
            if self.advance_manual_sim_pick("lift_done"):
                return
            self.manual_lift_done()
        elif command == "retreat_done":
            self.advance(ExecutorEvent.RETREAT_COMPLETE)
        elif command in ("reset", "home_reset", "reset_pose"):
            self.start_reset()
        elif command == "abort":
            self.safe_stop("operator_abort")
        else:
            self.publish_status("unknown_command", command=command)

    def start_reset(self):
        try:
            robot = robot_state_from_ros(self.latest_odom, self.latest_joints, self.config)
        except ValueError as exc:
            self.state = "WAIT_READY"
            self.pending_frames = []
            self.current_control_target = None
            self.publish_status("reset_waiting_robot_state", reason=str(exc))
            return

        start = measured_targets(robot)
        target = RobotTargets(
            base_linear=0.0,
            base_angular=0.0,
            slide=float(self.config.reset_slide),
            head_yaw=float(self.config.reset_head_yaw),
            head_pitch=float(self.config.reset_head_pitch),
            left_arm=tuple(float(value) for value in self.config.reset_left_arm),
            left_gripper=float(self.config.reset_left_gripper),
            right_arm=tuple(float(value) for value in self.config.reset_right_arm),
            right_gripper=float(self.config.reset_right_gripper),
        )
        self.pick_executor = None
        self.pending_frames = self.interpolate_targets(
            start,
            target,
            max_joint_step=float(self.config.reset_max_joint_step),
            max_slide_step=float(self.config.reset_max_slide_step),
        )
        self.current_control_target = start
        self.reset_target = target
        self.reset_started_at = time.time()
        self.reset_stable_since = 0.0
        self.state = "RESETTING"
        self.publish_status("reset_started", frames=len(self.pending_frames))
        if self.config.grasp_dry_run:
            self.pending_frames = []
            self.current_control_target = None
            self.reset_target = None
            self.state = "WAIT_READY"
            self.publish_status("reset_done", dry_run=True)

    def interpolate_targets(self, start, target, *, max_joint_step, max_slide_step):
        joint_values = (
            list(start.left_arm)
            + [start.left_gripper]
            + list(start.right_arm)
            + [start.right_gripper]
            + [start.head_yaw, start.head_pitch]
        )
        target_values = (
            list(target.left_arm)
            + [target.left_gripper]
            + list(target.right_arm)
            + [target.right_gripper]
            + [target.head_yaw, target.head_pitch]
        )
        max_joint_delta = max(
            [abs(after - before) for before, after in zip(joint_values, target_values)]
            or [0.0]
        )
        max_slide_delta = abs(target.slide - start.slide)
        joint_steps = int(math.ceil(max_joint_delta / max(max_joint_step, 1e-6)))
        slide_steps = int(math.ceil(max_slide_delta / max(max_slide_step, 1e-6)))
        count = max(1, joint_steps, slide_steps)
        frames = []
        for index in range(1, count + 1):
            ratio = index / count
            frames.append(
                RobotTargets(
                    base_linear=0.0,
                    base_angular=0.0,
                    slide=start.slide + (target.slide - start.slide) * ratio,
                    head_yaw=start.head_yaw + (target.head_yaw - start.head_yaw) * ratio,
                    head_pitch=start.head_pitch + (target.head_pitch - start.head_pitch) * ratio,
                    left_arm=tuple(
                        before + (after - before) * ratio
                        for before, after in zip(start.left_arm, target.left_arm)
                    ),
                    left_gripper=start.left_gripper
                    + (target.left_gripper - start.left_gripper) * ratio,
                    right_arm=tuple(
                        before + (after - before) * ratio
                        for before, after in zip(start.right_arm, target.right_arm)
                    ),
                    right_gripper=start.right_gripper
                    + (target.right_gripper - start.right_gripper) * ratio,
                )
            )
        return frames

    def start_grasp(self):
        if not self.config.enable_grasp_executor:
            self.publish_status("disabled", hint="set enable_grasp_executor=True")
            return
        self.state = "STARTING"
        self.manual_sim_pick = False
        self.pick_executor = None
        self.pending_frames = []
        self.current_control_target = None
        if self.latest_ready is None:
            self.state = "WAIT_READY"
            self.publish_status("waiting_ready")
            return
        if time.time() - self.latest_ready_rx > float(self.config.grasp_ready_max_age_sec):
            self.state = "WAIT_READY"
            self.publish_status("ready_too_old")
            return
        try:
            ready_task_id = str(self.latest_ready.get("task_id") or "")
            ready_profile = self.latest_ready.get("motion_grasp_profile")
            ready_source = self.latest_ready.get("motion_source_area")
            if ready_profile is None or ready_source is None:
                self.start_manual_sim_pick(
                    task_id=ready_task_id or "unknown",
                    profile=ready_profile or "manual_sim_missing_motion_metadata",
                )
                return

            robot = robot_state_from_ros(self.latest_odom, self.latest_joints, self.config)
            self.scene, self.goal = scene_goal_from_ready(self.latest_ready, robot)

            target = self.scene.object_by_id(self.goal.target_id)
            if target is None:
                raise ValueError("target is absent from grasp scene")
            self.publish_status(
                "grasp_target_normalized",
                pose={
                    "x": target.pose.x,
                    "y": target.pose.y,
                    "z": target.pose.z,
                    "yaw": target.pose.yaw,
                },
                size={
                    "length": target.size.length,
                    "width": target.size.width,
                    "height": target.size.height,
                },
            )
            if (
                self.goal.task_id.value != "task_1"
                or self.goal.grasp_profile.value != "table_side_hug"
            ):
                self.start_manual_sim_pick(
                    task_id=self.goal.task_id.value,
                    profile=self.goal.grasp_profile.value,
                )
                return
            approach_direction, standoff = self.docked_approach(robot, target)
            planner = AlreadyDockedPlanner()
            solver = make_solver(robot, self.config)
            self.pick_executor = MinimumPickExecutor(
                planner=planner,
                solver=solver,
                hug_profile=table_side_hug_profile(self.config),
                approach_direction=approach_direction,
                standoff=standoff,
                max_scene_age=float(self.config.grasp_ready_max_age_sec),
                perception_limits=perception_limits(self.config),
            )
            feedback = self.pick_executor.prepare(self.scene, self.goal, now=time.time(), safety_chain_ready=True)
            self.publish_feedback("prepare", feedback)
            if feedback.failed:
                self.publish_ik_debug("prepare_failed")
                self.safe_stop(self.reason_with_ik_debug(feedback.reason))
                return
            if feedback.phase is not ExecutionPhase.NAVIGATE_PICK:
                self.safe_stop(
                    "prepare did not reach navigate_pick: "
                    f"phase={feedback.phase.value}, reason={feedback.reason}"
                )
                return

            nav_feedback = self.pick_executor.advance(ExecutorEvent.NAVIGATION_REACHED, time.time())
            self.publish_feedback("navigation_reached", nav_feedback)
            if nav_feedback.failed or nav_feedback.phase is not ExecutionPhase.DOCK_PICK:
                self.safe_stop(
                    "navigation_reached did not reach dock_pick: "
                    f"phase={nav_feedback.phase.value}, reason={nav_feedback.reason}"
                )
                return
            dock_feedback = self.pick_executor.advance(ExecutorEvent.DOCKED, time.time())
            self.publish_feedback("docked", dock_feedback)
            if dock_feedback.failed or dock_feedback.phase is not ExecutionPhase.REFINE_PICK:
                self.safe_stop(
                    "docked did not reach refine_pick: "
                    f"phase={dock_feedback.phase.value}, reason={dock_feedback.reason}"
                )
                return
            refined = self.pick_executor.refine(self.scene, now=time.time(), safety_chain_ready=True)
            self.publish_feedback("refine", refined)
            if refined.failed:
                self.publish_ik_debug("refine_failed")
                self.safe_stop(self.reason_with_ik_debug(refined.reason))
                return
            if refined.phase is not ExecutionPhase.PREGRASP:
                self.safe_stop(
                    "refine did not reach pregrasp: "
                    f"phase={refined.phase.value}, reason={refined.reason}"
                )
                return

            self.queue_arm_targets("pregrasp")
            self.state = "EXECUTE_PREGRASP"
        except Exception as exc:
            self.safe_stop(str(exc))

    def start_manual_sim_pick(self, *, task_id, profile):
        self.manual_sim_pick = True
        self.pick_executor = None
        self.pending_frames = []
        self.current_control_target = None
        self.state = "EXECUTE_PREGRASP"
        self.publish_status(
            "manual_sim_pick_started",
            task_id=task_id,
            profile=profile,
            hint=(
                "real grasp is not implemented for this task/profile; "
                "use pregrasp_done/contact_done/hold_done/lift_done"
            ),
        )
        self.publish_status("queued", label="manual_sim_pregrasp", frames=0)

    def advance_manual_sim_pick(self, command):
        if not self.manual_sim_pick:
            return False
        if command == "pregrasp_done":
            if self.state != "EXECUTE_PREGRASP":
                self.publish_status(
                    "manual_sim_command_ignored",
                    command=command,
                    expected_state="EXECUTE_PREGRASP",
                )
                return True
            self.state = "EXECUTE_APPROACH"
            self.publish_status("queued", label="manual_sim_approach", frames=0)
            return True
        if command == "contact_done":
            if self.state != "EXECUTE_APPROACH":
                self.publish_status(
                    "manual_sim_command_ignored",
                    command=command,
                    expected_state="EXECUTE_APPROACH",
                )
                return True
            self.state = "EXECUTE_HOLD"
            self.publish_status("queued", label="manual_sim_hold", frames=0)
            return True
        if command == "hold_done":
            if self.state != "EXECUTE_HOLD":
                self.publish_status(
                    "manual_sim_command_ignored",
                    command=command,
                    expected_state="EXECUTE_HOLD",
                )
                return True
            self.state = "EXECUTE_LIFT"
            self.publish_status("queued", label="manual_sim_lift", frames=0)
            return True
        if command == "lift_done":
            if self.state != "EXECUTE_LIFT":
                self.publish_status(
                    "manual_sim_command_ignored",
                    command=command,
                    expected_state="EXECUTE_LIFT",
                )
                return True
            self.state = "LIFT_DONE"
            self.publish_status("lift_done_waiting_retreat", manual_sim=True)
            return True
        return False

    def docked_approach(self, robot, target):
        dx = target.pose.x - robot.base.x
        dy = target.pose.y - robot.base.y
        distance = math.hypot(dx, dy)
        if distance < 0.05:
            raise ValueError("robot base is too close to target for a valid grasp standoff")
        self.publish_status(
            "docked_approach",
            direction=[dx / distance, dy / distance],
            standoff=distance,
            base_yaw=robot.base.yaw,
            target_heading=math.atan2(dy, dx),
        )
        return (dx / distance, dy / distance), distance

    def publish_ik_debug(self, label):
        attempt = None if self.pick_executor is None else self.pick_executor.last_attempt
        if attempt is None:
            return
        for name, check in (
            ("pregrasp", attempt.pregrasp_ik),
            ("hold", attempt.hold_ik),
        ):
            solution = check.solution
            self.publish_status(
                "ik_debug",
                label=label,
                target=name,
                feasible=check.feasible,
                reason=check.reason,
                candidate_count=check.candidate_count,
                collision_checked=check.collision_checked,
                solution_slide=None if solution is None else solution.slide,
                residual=None if solution is None else solution.residual,
                orientation_error=None if solution is None else solution.orientation_error,
            )

    def reason_with_ik_debug(self, reason):
        attempt = None if self.pick_executor is None else self.pick_executor.last_attempt
        if attempt is None:
            return reason
        parts = []
        for name, check in (
            ("pregrasp", attempt.pregrasp_ik),
            ("hold", attempt.hold_ik),
        ):
            parts.append(
                f"{name}: feasible={check.feasible}, "
                f"candidates={check.candidate_count}, reason={check.reason}"
            )
        return f"{reason}; " + "; ".join(parts)

    def scene_with_manual_evidence(self, *, contact=False, lifted=False):
        if self.scene is None or self.goal is None:
            raise ValueError("grasp scene is not ready")
        # Keep manual evidence just behind the validator's ``now`` value.
        # Otherwise a few milliseconds of call ordering can look like a
        # "future" robot/camera/evidence timestamp.
        now = time.time() - 0.05
        robot = robot_state_from_ros(self.latest_odom, self.latest_joints, self.config, now=now)
        evidence = GraspEvidence(
            target_id=self.goal.target_id,
            observed_at=now,
            source_cameras=(CameraId.LEFT_WRIST_RGB, CameraId.RIGHT_WRIST_RGB),
            safe_to_continue=True,
            left_contact_confirmed=bool(contact),
            right_contact_confirmed=bool(contact),
            centered_error_m=0.0 if contact else None,
            object_lifted=bool(lifted),
        )
        return replace(
            self.scene,
            timestamp=now,
            robot=robot,
            camera_observations=(
                CameraObservation(CameraId.HEAD_RGBD, now),
                CameraObservation(CameraId.LEFT_WRIST_RGB, now),
                CameraObservation(CameraId.RIGHT_WRIST_RGB, now),
            ),
            grasp_evidence=evidence,
        )

    def manual_contact_done(self):
        if self.pick_executor is None:
            self.publish_status(
                "not_started",
                hint="send start first, wait for queued label=pregrasp, then send pregrasp_done",
            )
            return
        if self.pick_executor.phase is not ExecutionPhase.APPROACH:
            self.publish_status(
                "contact_done_ignored",
                phase=self.pick_executor.phase.value,
                hint="contact_done is only valid after pregrasp_done enters approach",
            )
            return
        now = time.time()
        guarded_scene = self.scene_with_manual_evidence(contact=False, lifted=False)
        guard = self.pick_executor.check_approach(guarded_scene, now=now)
        self.publish_feedback("manual_approach_guard", guard)
        if guard.failed or guard.need_reobserve:
            self.safe_stop(f"manual approach guard rejected: {guard.reason}")
            return
        self.advance(ExecutorEvent.CONTACT_REACHED)

    def manual_hold_done(self):
        if self.pick_executor is None:
            self.publish_status(
                "not_started",
                hint="send start first, wait for queued label=pregrasp, then send pregrasp_done",
            )
            return
        if self.pick_executor.phase is not ExecutionPhase.HOLD:
            self.publish_status(
                "hold_done_ignored",
                phase=self.pick_executor.phase.value,
                hint="hold_done is only valid after contact_done queues and executes hold",
            )
            return
        feedback = self.pick_executor.advance(ExecutorEvent.HOLD_CONFIRMED, time.time())
        self.publish_feedback("hold_confirmed", feedback)
        if feedback.failed:
            self.safe_stop(feedback.reason)
            return
        verified = self.pick_executor.verify_hold(
            self.scene_with_manual_evidence(contact=True, lifted=False),
            now=time.time(),
        )
        self.publish_feedback("manual_verify_hold", verified)
        if verified.failed or verified.need_reobserve:
            self.safe_stop(f"manual hold verification rejected: {verified.reason}")
            return
        if self.pick_executor.phase is ExecutionPhase.LIFT:
            self.queue_lift_targets()
            self.state = "EXECUTE_LIFT"

    def manual_lift_done(self):
        if self.pick_executor is None:
            self.publish_status(
                "not_started",
                hint="send start first, wait for queued label=pregrasp, then send pregrasp_done",
            )
            return
        if self.pick_executor.phase is not ExecutionPhase.LIFT:
            self.publish_status(
                "lift_done_ignored",
                phase=self.pick_executor.phase.value,
                hint="lift_done is only valid after hold_done queues and executes lift",
            )
            return
        feedback = self.pick_executor.advance(ExecutorEvent.LIFT_REACHED, time.time())
        self.publish_feedback("lift_reached", feedback)
        if feedback.failed:
            self.safe_stop(feedback.reason)
            return
        verified = self.pick_executor.verify_lift(
            self.scene_with_manual_evidence(contact=True, lifted=True),
            now=time.time(),
        )
        self.publish_feedback("manual_verify_lift", verified)
        if verified.failed or verified.need_reobserve:
            self.safe_stop(f"manual lift verification rejected: {verified.reason}")
            return
        self.state = "LIFT_DONE"
        self.publish_status("lift_done_waiting_retreat")

    def advance(self, event):
        if self.pick_executor is None:
            self.publish_status(
                "not_started",
                hint="send start first, wait for queued label=pregrasp, then send pregrasp_done",
            )
            return
        feedback = self.pick_executor.advance(event, time.time())
        self.publish_feedback("advance", feedback)
        if feedback.failed:
            self.safe_stop(feedback.reason)
            return
        if self.pick_executor.phase is ExecutionPhase.APPROACH:
            self.queue_arm_targets("approach")
            self.state = "EXECUTE_APPROACH"
        elif self.pick_executor.phase is ExecutionPhase.HOLD:
            self.queue_arm_targets("hold")
            self.state = "EXECUTE_HOLD"
        elif self.pick_executor.phase is ExecutionPhase.LIFT:
            self.queue_lift_targets()
            self.state = "EXECUTE_LIFT"
        elif self.pick_executor.phase is ExecutionPhase.MINIMAL_DONE:
            self.state = "DONE"
            self.publish_status("done")

    def queue_arm_targets(self, label):
        robot = robot_state_from_ros(self.latest_odom, self.latest_joints, self.config)
        measured = measured_targets(robot)
        self.current_control_target = measured
        frames = self.pick_executor.arm_targets(
            measured,
            max_joint_step=float(self.config.grasp_max_joint_step),
            max_slide_step=float(self.config.grasp_max_slide_step),
        )
        self.pending_frames = list(frames)
        self.publish_status("queued", label=label, frames=len(self.pending_frames))
        if self.config.grasp_dry_run:
            self.pending_frames = []
            self.current_control_target = None
            self.publish_status("dry_run_skip_publish", label=label)

    def queue_lift_targets(self):
        robot = robot_state_from_ros(self.latest_odom, self.latest_joints, self.config)
        start = measured_targets(robot)
        self.current_control_target = start
        delta = float(self.config.grasp_lift_delta_m)
        step = abs(float(self.config.grasp_lift_step_m))
        count = max(1, int(abs(delta) / step))
        self.pending_frames = []
        for index in range(1, count + 1):
            slide = start.slide + delta * index / count
            self.pending_frames.append(
                RobotTargets(
                    base_linear=0.0,
                    base_angular=0.0,
                    slide=slide,
                    head_yaw=start.head_yaw,
                    head_pitch=start.head_pitch,
                    left_arm=start.left_arm,
                    left_gripper=start.left_gripper,
                    right_arm=start.right_arm,
                    right_gripper=start.right_gripper,
                )
            )
        self.publish_status("queued", label="lift", frames=len(self.pending_frames))
        if self.config.grasp_dry_run:
            self.pending_frames = []
            self.current_control_target = None
            self.publish_status("dry_run_skip_publish", label="lift")

    def tick(self):
        if self.state == "WAIT_READY":
            return
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        self.heartbeat_pub.publish(header)
        if self.state == "SAFE_STOP":
            return
        if self.pending_frames:
            self.current_control_target = self.pending_frames.pop(0)
        elif self.state == "RESETTING":
            self.current_control_target = self.reset_target
            if self.reset_reached():
                self.state = "WAIT_READY"
                self.current_control_target = None
                self.reset_target = None
                self.publish_status("reset_done")
                return
            if time.time() - self.reset_started_at > float(self.config.reset_timeout_sec):
                self.publish_status("reset_timeout", hint="check safety gateway is enabled")
                self.reset_started_at = time.time()
        elif self.current_control_target is None:
            self.current_control_target = self.idle_hold_target()
        if self.current_control_target is None:
            return
        self.publish_control_target(self.current_control_target)

    def reset_reached(self):
        if self.reset_target is None:
            return False
        try:
            current = measured_targets(
                robot_state_from_ros(self.latest_odom, self.latest_joints, self.config)
            )
        except ValueError:
            self.reset_stable_since = 0.0
            return False

        arm_error = max(
            [abs(before - after) for before, after in zip(current.left_arm, self.reset_target.left_arm)]
            + [abs(before - after) for before, after in zip(current.right_arm, self.reset_target.right_arm)]
            + [
                abs(current.left_gripper - self.reset_target.left_gripper),
                abs(current.right_gripper - self.reset_target.right_gripper),
                abs(current.head_yaw - self.reset_target.head_yaw),
                abs(current.head_pitch - self.reset_target.head_pitch),
            ]
        )
        slide_error = abs(current.slide - self.reset_target.slide)
        reached = (
            arm_error <= float(self.config.reset_position_tolerance)
            and slide_error <= float(self.config.reset_slide_tolerance)
        )
        if not reached:
            self.reset_stable_since = 0.0
            return False
        now = time.time()
        if self.reset_stable_since <= 0.0:
            self.reset_stable_since = now
            return False
        return now - self.reset_stable_since >= float(self.config.reset_hold_sec)

    def idle_hold_target(self):
        try:
            robot = robot_state_from_ros(self.latest_odom, self.latest_joints, self.config)
        except ValueError:
            return None
        now = time.time()
        if now - self.last_idle_request_log_at > 2.0:
            self.last_idle_request_log_at = now
            self.publish_status("idle_hold_request")
        return measured_targets(robot)

    def publish_control_target(self, target):
        msg = Float64MultiArray()
        msg.data = list(legacy_control_vector(target))
        self.control_pub.publish(msg)

    def safe_stop(self, reason):
        self.pending_frames = []
        self.current_control_target = None
        self.state = "SAFE_STOP"
        self.publish_status("safe_stop", reason=reason)

    def publish_feedback(self, source, feedback):
        self.publish_status(
            source,
            phase=feedback.phase.value,
            failed=feedback.failed,
            completed=feedback.completed,
            need_reobserve=feedback.need_reobserve,
            reason=feedback.reason,
        )

    def publish_status(self, event, **extra):
        payload = {"event": event, "state": self.state, "time": time.time(), **extra}
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.get_logger().info(json.dumps(payload, ensure_ascii=False))


def main():
    rclpy.init()
    node = GraspExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
