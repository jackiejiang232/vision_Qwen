"""带主动感知检查点的最小抓取状态机。

视觉与运动分模块，但不是两条互不相干的流水线：执行器在导航前、停靠后、接近、
抱持和抬升时提出结构化观测请求；视觉返回新 ``SceneState`` 后，执行器才放行
下一步。这里不启动 ROS、不循环重试，也不自动越过人工确认。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from .candidate import CandidateReport, validate_candidate
from .contracts import (
    ActionCandidate,
    ActionSkill,
    CameraId,
    ExecutionFeedback,
    ExecutionPhase,
    ObservationPurpose,
    ObservationRequest,
    PickPlaceGoal,
    Pose2D,
    Pose3D,
    RecoveryCode,
    RobotTargets,
    SceneState,
)
from .kinematics import (
    DualArmSolver,
    JointPair,
    KinematicCheck,
    check_dual_arm_hug,
    synchronized_joint_path,
)
from .manipulation import DualArmHugPlan, HugProfileGeometry, dual_arm_hug
from .navigation import PathPlan, StaticAStarPlanner, operating_stance
from .scene import validate_grasp_evidence, validate_observation


class ExecutorEvent(str, Enum):
    NAVIGATION_REACHED = "navigation_reached"
    DOCKED = "docked"
    PREGRASP_REACHED = "pregrasp_reached"
    CONTACT_REACHED = "contact_reached"
    HOLD_CONFIRMED = "hold_confirmed"
    LIFT_REACHED = "lift_reached"
    RETREAT_COMPLETE = "retreat_complete"
    FAILED = "failed"


@dataclass(frozen=True)
class PickAttempt:
    goal: PickPlaceGoal
    candidate: ActionCandidate
    path: PathPlan
    hug: DualArmHugPlan
    pregrasp_ik: KinematicCheck
    hold_ik: KinematicCheck
    validation: CandidateReport


@dataclass(frozen=True)
class PerceptionMotionLimits:
    """视觉结果进入运动状态机前必须满足的显式门槛。

    这些值最终来自已确认配置；当前测试必须主动传入，避免代码静默使用一套看似
    权威、实际尚未标定的比赛参数。
    """

    coarse_max_age: float
    coarse_min_confidence: float
    coarse_position_std_m: float
    fine_max_age: float
    fine_min_confidence: float
    fine_position_std_m: float
    fine_yaw_std_rad: float
    approach_max_age: float
    max_centered_error_m: float
    redock_position_tolerance_m: float
    redock_yaw_tolerance_rad: float
    observation_extra_standoff_m: float

    def __post_init__(self) -> None:
        positive = (
            "coarse_max_age",
            "coarse_position_std_m",
            "fine_max_age",
            "fine_position_std_m",
            "fine_yaw_std_rad",
            "approach_max_age",
            "max_centered_error_m",
            "redock_position_tolerance_m",
            "redock_yaw_tolerance_rad",
            "observation_extra_standoff_m",
        )
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("coarse_min_confidence", "fine_min_confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


class MinimumPickExecutor:
    """最小抓取链路的计划器、视觉门槛与人工确认状态机。"""

    def __init__(
        self,
        planner: StaticAStarPlanner,
        solver: DualArmSolver,
        hug_profile: HugProfileGeometry,
        approach_direction: tuple[float, float],
        standoff: float,
        max_scene_age: float,
        perception_limits: PerceptionMotionLimits,
    ) -> None:
        if standoff <= 0.0 or max_scene_age <= 0.0:
            raise ValueError("standoff and max_scene_age must be positive")
        self.planner = planner
        self.solver = solver
        self.hug_profile = hug_profile
        self.approach_direction = approach_direction
        self.standoff = standoff
        self.max_scene_age = max_scene_age
        self.perception_limits = perception_limits
        self.phase = ExecutionPhase.BUILD_GOAL
        self.last_attempt: PickAttempt | None = None
        self._last_approach_check_at: float | None = None

    def prepare(
        self,
        scene: SceneState,
        goal: PickPlaceGoal,
        *,
        now: float,
        safety_chain_ready: bool,
    ) -> ExecutionFeedback:
        """使用头部 RGB-D 粗定位计算可审阅方案；缺数据时明确请求重新观测。"""

        if self.phase is not ExecutionPhase.BUILD_GOAL:
            return self._safe_stop("prepare may only run from build_goal", now)
        if goal.grasp_profile is not self.hug_profile.name:
            return self._safe_stop("goal and calibrated grasp profile do not match", now)
        target = scene.object_by_id(goal.target_id)
        target_pose = goal.target_pose if target is None else target.pose
        request = self._observation_request(
            ObservationPurpose.ACQUIRE_TARGET,
            goal.target_id,
            target_pose,
        )
        observation = validate_observation(scene, request, now=now)
        if not observation.accepted:
            return self._wait_for_observation(
                "; ".join(observation.reasons),
                request,
                now,
            )
        self.phase = ExecutionPhase.PLAN_PICK
        try:
            attempt = self._build_attempt(
                scene,
                goal,
                now=now,
                safety_chain_ready=safety_chain_ready,
            )
        except ValueError as exc:
            return self._safe_stop(f"static navigation rejected: {exc}", now)
        self.last_attempt = attempt
        if not attempt.validation.accepted:
            return self._safe_stop("; ".join(attempt.validation.reasons), now)
        self.phase = ExecutionPhase.NAVIGATE_PICK
        return ExecutionFeedback(self.phase, False, False, "pick plan is ready for manual review", now)

    def refine(
        self,
        scene: SceneState,
        *,
        now: float,
        safety_chain_ready: bool,
    ) -> ExecutionFeedback:
        """停靠并停车后重观测目标，重新计算双臂目标和是否需要再次停靠。"""

        if self.phase is not ExecutionPhase.REFINE_PICK or self.last_attempt is None:
            return self._safe_stop("refine is only valid after docking", now)
        goal = self.last_attempt.goal
        target = scene.object_by_id(goal.target_id)
        target_pose = goal.target_pose if target is None else target.pose
        request = self._observation_request(
            ObservationPurpose.REFINE_TARGET,
            goal.target_id,
            target_pose,
        )
        observation = validate_observation(scene, request, now=now)
        if not observation.accepted:
            return self._wait_for_observation(
                "; ".join(observation.reasons),
                request,
                now,
            )
        try:
            attempt = self._build_attempt(
                scene,
                goal,
                now=now,
                safety_chain_ready=safety_chain_ready,
            )
        except ValueError as exc:
            return self._safe_stop(f"refined plan rejected: {exc}", now)
        self.last_attempt = attempt
        if not attempt.validation.accepted:
            return self._safe_stop("; ".join(attempt.validation.reasons), now)

        position_error, yaw_error = _pose_error(
            scene.robot.base,
            attempt.candidate.approach_pose,
        )
        if (
            position_error
            > self.perception_limits.redock_position_tolerance_m
            or yaw_error > self.perception_limits.redock_yaw_tolerance_rad
        ):
            self.phase = ExecutionPhase.DOCK_PICK
            return ExecutionFeedback(
                self.phase,
                False,
                False,
                "refined target changed the operating stance; redock is required",
                now,
                need_reobserve=False,
                position_error=position_error,
                angle_error=yaw_error,
                recovery=RecoveryCode.REDOCK,
            )
        self.phase = ExecutionPhase.PREGRASP
        return ExecutionFeedback(
            self.phase,
            False,
            False,
            "fresh head and wrist observations accepted; refined IK is ready",
            now,
            position_error=position_error,
            angle_error=yaw_error,
        )

    def check_approach(
        self,
        scene: SceneState,
        *,
        now: float,
    ) -> ExecutionFeedback:
        """接近过程中检查腕部画面；这里只放行或暂停，不改动夹爪目标。"""

        if self.phase is not ExecutionPhase.APPROACH or self.last_attempt is None:
            return self._safe_stop("approach guard is only valid during approach", now)
        request = self._observation_request(
            ObservationPurpose.GUARD_APPROACH,
            self.last_attempt.goal.target_id,
            self.last_attempt.goal.target_pose,
        )
        observation = validate_observation(scene, request, now=now)
        evidence = validate_grasp_evidence(
            scene,
            request,
            now=now,
            max_centered_error_m=self.perception_limits.max_centered_error_m,
        )
        reasons = observation.reasons + evidence.reasons
        if evidence.unsafe:
            return self._safe_stop(
                "; ".join(reasons),
                now,
                recovery=RecoveryCode.RETRY_GRASP,
            )
        if reasons:
            return self._wait_for_observation("; ".join(reasons), request, now)
        self._last_approach_check_at = now
        return ExecutionFeedback(
            self.phase,
            False,
            False,
            "fresh bilateral wrist guard accepted",
            now,
        )

    def verify_hold(self, scene: SceneState, *, now: float) -> ExecutionFeedback:
        """抱持结束后，必须由三相机事件证据确认双侧接触和居中。"""

        return self._verify_grasp_checkpoint(
            scene,
            now=now,
            expected_phase=ExecutionPhase.VERIFY_HOLD,
            purpose=ObservationPurpose.VERIFY_HOLD,
            next_phase=ExecutionPhase.LIFT,
            success_reason="bilateral hold is visually confirmed; lift may be planned",
        )

    def verify_lift(self, scene: SceneState, *, now: float) -> ExecutionFeedback:
        """抬升后确认箱体确实离开支撑面，未确认就不能撤离。"""

        return self._verify_grasp_checkpoint(
            scene,
            now=now,
            expected_phase=ExecutionPhase.VERIFY_LIFT,
            purpose=ObservationPurpose.VERIFY_LIFT,
            next_phase=ExecutionPhase.RETREAT_PICK,
            success_reason="box lift is visually confirmed; retreat may be planned",
        )

    def arm_targets(
        self,
        measured: RobotTargets,
        max_joint_step: float,
        max_slide_step: float,
    ) -> tuple[RobotTargets, ...]:
        """为当前手臂阶段生成同步目标，且始终将底盘速度置零。

        本函数只生成数据；实际发布仍必须由现有 SafetyGateway 完成。
        """

        if self.last_attempt is None:
            raise RuntimeError("prepare a pick attempt before requesting arm targets")
        check = self.last_attempt.pregrasp_ik if self.phase is ExecutionPhase.PREGRASP else self.last_attempt.hold_ik
        if self.phase not in {ExecutionPhase.PREGRASP, ExecutionPhase.APPROACH, ExecutionPhase.HOLD}:
            raise RuntimeError(f"arm targets are not valid in phase {self.phase.value}")
        if check.solution is None:
            raise RuntimeError(check.reason)
        start = JointPair(
            measured.left_arm,
            measured.right_arm,
            0.0,
            True,
            measured.slide,
        )
        frames = synchronized_joint_path(
            start,
            check.solution,
            max_joint_step,
            max_slide_step,
        )
        return tuple(
            RobotTargets(
                base_linear=0.0,
                base_angular=0.0,
                slide=measured.slide if frame.slide is None else frame.slide,
                head_yaw=measured.head_yaw,
                head_pitch=measured.head_pitch,
                left_arm=frame.left,
                left_gripper=measured.left_gripper,
                right_arm=frame.right,
                right_gripper=measured.right_gripper,
            )
            for frame in frames
        )

    def advance(self, event: ExecutorEvent, now: float) -> ExecutionFeedback:
        """只接受下一步的单个确认事件，任何异常事件立即转入 ``safe_stop``。"""

        if event is ExecutorEvent.FAILED:
            return self._safe_stop("operator or sensor reported a failed action", now)
        if self.phase is ExecutionPhase.DOCK_PICK and event is ExecutorEvent.DOCKED:
            self.phase = ExecutionPhase.REFINE_PICK
            assert self.last_attempt is not None
            request = self._observation_request(
                ObservationPurpose.REFINE_TARGET,
                self.last_attempt.goal.target_id,
                self.last_attempt.goal.target_pose,
            )
            return ExecutionFeedback(
                self.phase,
                False,
                False,
                "docking accepted; stop and refresh head plus wrist observations",
                now,
                need_reobserve=True,
                recovery=RecoveryCode.REOBSERVE,
                observation_request=request,
            )
        if self.phase is ExecutionPhase.APPROACH and event is ExecutorEvent.CONTACT_REACHED:
            if (
                self._last_approach_check_at is None
                or now - self._last_approach_check_at
                > self.perception_limits.approach_max_age
            ):
                assert self.last_attempt is not None
                request = self._observation_request(
                    ObservationPurpose.GUARD_APPROACH,
                    self.last_attempt.goal.target_id,
                    self.last_attempt.goal.target_pose,
                )
                return self._wait_for_observation(
                    "contact cannot be accepted without a fresh wrist guard",
                    request,
                    now,
                )
            self.phase = ExecutionPhase.HOLD
            return ExecutionFeedback(
                self.phase,
                False,
                False,
                "fresh wrist guard and contact event accepted",
                now,
            )
        if self.phase is ExecutionPhase.HOLD and event is ExecutorEvent.HOLD_CONFIRMED:
            self.phase = ExecutionPhase.VERIFY_HOLD
            assert self.last_attempt is not None
            request = self._observation_request(
                ObservationPurpose.VERIFY_HOLD,
                self.last_attempt.goal.target_id,
                self.last_attempt.goal.target_pose,
            )
            return ExecutionFeedback(
                self.phase,
                False,
                False,
                "hold motion ended; visual hold evidence is required",
                now,
                need_reobserve=True,
                recovery=RecoveryCode.REOBSERVE,
                observation_request=request,
            )
        if self.phase is ExecutionPhase.LIFT and event is ExecutorEvent.LIFT_REACHED:
            self.phase = ExecutionPhase.VERIFY_LIFT
            assert self.last_attempt is not None
            request = self._observation_request(
                ObservationPurpose.VERIFY_LIFT,
                self.last_attempt.goal.target_id,
                self.last_attempt.goal.target_pose,
            )
            return ExecutionFeedback(
                self.phase,
                False,
                False,
                "lift motion ended; visual lift evidence is required",
                now,
                need_reobserve=True,
                recovery=RecoveryCode.REOBSERVE,
                observation_request=request,
            )
        transitions = {
            (ExecutionPhase.NAVIGATE_PICK, ExecutorEvent.NAVIGATION_REACHED): ExecutionPhase.DOCK_PICK,
            (ExecutionPhase.PREGRASP, ExecutorEvent.PREGRASP_REACHED): ExecutionPhase.APPROACH,
            (ExecutionPhase.RETREAT_PICK, ExecutorEvent.RETREAT_COMPLETE): ExecutionPhase.MINIMAL_DONE,
        }
        next_phase = transitions.get((self.phase, event))
        if next_phase is None:
            return self._safe_stop(f"event {event.value} is invalid in phase {self.phase.value}", now)
        self.phase = next_phase
        completed = self.phase is ExecutionPhase.MINIMAL_DONE
        return ExecutionFeedback(self.phase, completed, False, "manual step confirmation accepted", now)

    def _build_attempt(
        self,
        scene: SceneState,
        goal: PickPlaceGoal,
        *,
        now: float,
        safety_chain_ready: bool,
    ) -> PickAttempt:
        target = scene.object_by_id(goal.target_id)
        if target is None:
            raise ValueError("goal target is absent from SceneState")
        updated_goal = replace(
            goal,
            target_pose=target.pose,
            target_size=target.size,
        )
        stance = operating_stance(target, self.approach_direction, self.standoff)
        path = self.planner.plan(scene.robot.base, stance)
        hug = dual_arm_hug(target, self.hug_profile)
        current = JointPair(
            scene.robot.left_arm,
            scene.robot.right_arm,
            0.0,
            True,
            scene.robot.slide,
        )
        # 接近与接触分别求解；只验证最终抱持解无法证明接近过程可达。
        pregrasp_hug = DualArmHugPlan(
            hug.profile,
            _pregrasp_as_contact(hug.left),
            _pregrasp_as_contact(hug.right),
        )
        pregrasp_ik = check_dual_arm_hug(self.solver, pregrasp_hug, current)
        hold_ik = check_dual_arm_hug(self.solver, hug, current)
        candidate = ActionCandidate(
            skill=ActionSkill.PICK,
            target_id=updated_goal.target_id,
            grasp_profile=updated_goal.grasp_profile,
            approach_pose=stance,
            place_pose=None,
            recovery=RecoveryCode.SAFE_STOP.value,
            confidence=target.confidence,
        )
        validation = validate_candidate(
            scene,
            updated_goal,
            candidate,
            now=now,
            max_scene_age=self.max_scene_age,
            path_found=True,
            ik_feasible=pregrasp_ik.feasible and hold_ik.feasible,
            safety_chain_ready=safety_chain_ready,
        )
        return PickAttempt(
            updated_goal,
            candidate,
            path,
            hug,
            pregrasp_ik,
            hold_ik,
            validation,
        )

    def _observation_request(
        self,
        purpose: ObservationPurpose,
        target_id: str,
        target_pose: Pose3D,
    ) -> ObservationRequest:
        limits = self.perception_limits
        look_at = target_pose
        if purpose is ObservationPurpose.ACQUIRE_TARGET:
            requested_base = _stance_from_pose(
                target_pose,
                self.approach_direction,
                self.standoff + limits.observation_extra_standoff_m,
            )
            return ObservationRequest(
                purpose,
                target_id,
                (CameraId.HEAD_RGBD,),
                limits.coarse_max_age,
                limits.coarse_min_confidence,
                max_position_std_m=limits.coarse_position_std_m,
                require_stationary=True,
                requested_base_pose=requested_base,
                look_at=look_at,
            )
        if purpose is ObservationPurpose.REFINE_TARGET:
            return ObservationRequest(
                purpose,
                target_id,
                (
                    CameraId.HEAD_RGBD,
                    CameraId.LEFT_WRIST_RGB,
                    CameraId.RIGHT_WRIST_RGB,
                ),
                limits.fine_max_age,
                limits.fine_min_confidence,
                max_position_std_m=limits.fine_position_std_m,
                max_yaw_std_rad=limits.fine_yaw_std_rad,
                require_stationary=True,
                requested_base_pose=_stance_from_pose(
                    target_pose,
                    self.approach_direction,
                    self.standoff,
                ),
                look_at=look_at,
            )
        if purpose is ObservationPurpose.GUARD_APPROACH:
            cameras = (CameraId.LEFT_WRIST_RGB, CameraId.RIGHT_WRIST_RGB)
            max_age = limits.approach_max_age
        else:
            cameras = (
                CameraId.HEAD_RGBD,
                CameraId.LEFT_WRIST_RGB,
                CameraId.RIGHT_WRIST_RGB,
            )
            max_age = limits.fine_max_age
        return ObservationRequest(
            purpose,
            target_id,
            cameras,
            max_age,
            0.0,
            require_target_pose=False,
            require_stationary=True,
            look_at=look_at,
        )

    def _verify_grasp_checkpoint(
        self,
        scene: SceneState,
        *,
        now: float,
        expected_phase: ExecutionPhase,
        purpose: ObservationPurpose,
        next_phase: ExecutionPhase,
        success_reason: str,
    ) -> ExecutionFeedback:
        if self.phase is not expected_phase or self.last_attempt is None:
            return self._safe_stop(
                f"{purpose.value} is invalid in phase {self.phase.value}",
                now,
            )
        request = self._observation_request(
            purpose,
            self.last_attempt.goal.target_id,
            self.last_attempt.goal.target_pose,
        )
        observation = validate_observation(scene, request, now=now)
        evidence = validate_grasp_evidence(
            scene,
            request,
            now=now,
            max_centered_error_m=self.perception_limits.max_centered_error_m,
        )
        reasons = observation.reasons + evidence.reasons
        if evidence.unsafe:
            return self._safe_stop(
                "; ".join(reasons),
                now,
                recovery=RecoveryCode.RETRY_GRASP,
            )
        if reasons:
            return self._wait_for_observation("; ".join(reasons), request, now)
        self.phase = next_phase
        return ExecutionFeedback(
            self.phase,
            False,
            False,
            success_reason,
            now,
            held_object_id=self.last_attempt.goal.target_id,
        )

    def _wait_for_observation(
        self,
        reason: str,
        request: ObservationRequest,
        now: float,
    ) -> ExecutionFeedback:
        return ExecutionFeedback(
            self.phase,
            False,
            False,
            reason,
            now,
            need_reobserve=True,
            recovery=RecoveryCode.REOBSERVE,
            observation_request=request,
        )

    def _safe_stop(
        self,
        reason: str,
        now: float,
        *,
        recovery: RecoveryCode = RecoveryCode.SAFE_STOP,
    ) -> ExecutionFeedback:
        self.phase = ExecutionPhase.SAFE_STOP
        return ExecutionFeedback(
            self.phase,
            False,
            True,
            reason,
            now,
            recovery=recovery,
        )


def _pregrasp_as_contact(contact):
    """给 IK 接口构造预抓取目标，保留同一法向和切线。"""

    from .manipulation import ArmContact

    return ArmContact(
        arm=contact.arm,
        pregrasp=contact.pregrasp,
        contact=contact.pregrasp,
        pregrasp_surface=contact.pregrasp_surface,
        contact_surface=contact.pregrasp_surface,
        outward_normal=contact.outward_normal,
        surface_tangent=contact.surface_tangent,
    )


def _stance_from_pose(
    target: Pose3D,
    approach_direction: tuple[float, float],
    standoff: float,
) -> Pose2D:
    dx, dy = approach_direction
    length = math.hypot(dx, dy)
    if length == 0.0 or standoff <= 0.0:
        raise ValueError("approach direction and standoff must be non-zero")
    dx /= length
    dy /= length
    return Pose2D(
        target.x - dx * standoff,
        target.y - dy * standoff,
        math.atan2(dy, dx),
    )


def _pose_error(measured: Pose2D, target: Pose2D) -> tuple[float, float]:
    distance = math.hypot(target.x - measured.x, target.y - measured.y)
    yaw = abs(math.atan2(math.sin(target.yaw - measured.yaw), math.cos(target.yaw - measured.yaw)))
    return distance, yaw
