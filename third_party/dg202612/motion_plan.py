"""单动作计划号与执行前快照复核。

本模块只比较数据，不读取 ROS，也不发布控制。计划工具与未来执行器共享同一套规则，
避免“终端显示已审核，但执行时换了目标或机器人已经移动”。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from .contracts import MotionAction, RobotState


@dataclass(frozen=True)
class PlanValidationLimits:
    """计划复核的固定门槛，数值来自本项目已批准的阶段计划。"""

    plan_max_age: float = 300.0
    data_max_age: float = 0.5
    base_position: float = 0.02
    base_yaw: float = 0.03
    slide: float = 0.01
    arm_joint: float = 0.03


@dataclass(frozen=True)
class MotionPlan:
    """一份可审核、但不等于已经获准执行的动作计划。"""

    plan_id: str
    action: MotionAction
    created_at: float
    expires_at: float
    git_sha: str
    config_digest: str
    robot: RobotState
    target: Mapping[str, Any]
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "action": self.action.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "git_sha": self.git_sha,
            "config_digest": self.config_digest,
            "robot_state": robot_state_dict(self.robot),
            "target": dict(self.target),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "published_control_messages": 0,
        }


@dataclass(frozen=True)
class PlanValidation:
    accepted: bool
    reasons: tuple[str, ...]
    data_age: float
    base_position_error: float
    base_yaw_error: float
    slide_error: float
    max_arm_joint_error: float


def robot_state_dict(state: RobotState) -> dict[str, Any]:
    """固定字段和左右臂顺序，保证计划号可重复。"""

    return {
        "base": {"x": state.base.x, "y": state.base.y, "yaw": state.base.yaw},
        "base_linear": state.base_linear,
        "base_angular": state.base_angular,
        "slide": state.slide,
        "head_yaw": state.head_yaw,
        "head_pitch": state.head_pitch,
        "left_arm": list(state.left_arm),
        "left_gripper": state.left_gripper,
        "right_arm": list(state.right_arm),
        "right_gripper": state.right_gripper,
        "observed_at": state.observed_at,
    }


def robot_state_from_dict(value: Mapping[str, Any]) -> RobotState:
    """读取计划中的机器人快照；所有字段必须显式存在。"""

    def number(item: Any, field: str) -> float:
        try:
            result = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(result):
            raise ValueError(f"{field} must be finite")
        return result

    def joints(field: str) -> tuple[float, ...]:
        raw = value.get(field)
        if not isinstance(raw, list) or len(raw) != 6:
            raise ValueError(f"{field} must contain six joints")
        return tuple(number(item, f"{field}[{index}]") for index, item in enumerate(raw))

    base = value.get("base")
    if not isinstance(base, Mapping):
        raise ValueError("robot_state.base must be an object")
    from .contracts import Pose2D

    return RobotState(
        base=Pose2D(
            number(base.get("x"), "base.x"),
            number(base.get("y"), "base.y"),
            number(base.get("yaw"), "base.yaw"),
        ),
        base_linear=number(value.get("base_linear"), "base_linear"),
        base_angular=number(value.get("base_angular"), "base_angular"),
        slide=number(value.get("slide"), "slide"),
        head_yaw=number(value.get("head_yaw"), "head_yaw"),
        head_pitch=number(value.get("head_pitch"), "head_pitch"),
        left_arm=joints("left_arm"),
        left_gripper=number(value.get("left_gripper"), "left_gripper"),
        right_arm=joints("right_arm"),
        right_gripper=number(value.get("right_gripper"), "right_gripper"),
        observed_at=number(value.get("observed_at"), "observed_at"),
    )


def motion_plan_from_dict(value: Mapping[str, Any]) -> MotionPlan:
    """从磁盘恢复计划并重新计算 plan_id，拒绝手工篡改。"""

    if value.get("schema_version") != 1:
        raise ValueError("unsupported motion plan schema")
    target = value.get("target")
    robot_raw = value.get("robot_state")
    reasons_raw = value.get("rejection_reasons")
    if not isinstance(target, Mapping):
        raise ValueError("motion plan target must be an object")
    if not isinstance(robot_raw, Mapping):
        raise ValueError("motion plan robot_state must be an object")
    if not isinstance(reasons_raw, list):
        raise ValueError("motion plan rejection_reasons must be a list")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("motion plan accepted must be boolean")
    action = MotionAction(str(value.get("action")))
    robot = robot_state_from_dict(robot_raw)
    created_at = float(value.get("created_at"))
    expires_at = float(value.get("expires_at"))
    if not all(math.isfinite(item) for item in (created_at, expires_at)):
        raise ValueError("motion plan times must be finite")
    if expires_at <= created_at:
        raise ValueError("motion plan expiry must follow creation")
    git_sha = str(value.get("git_sha"))
    config_digest = str(value.get("config_digest"))
    if not git_sha or not config_digest:
        raise ValueError("motion plan git and config digests are required")
    reasons = tuple(str(item) for item in reasons_raw)
    if accepted and reasons:
        raise ValueError("accepted motion plan cannot contain rejection reasons")
    expected_id = calculate_plan_id(
        action,
        robot,
        target,
        config_digest,
        git_sha,
    )
    plan_id = str(value.get("plan_id"))
    if plan_id != expected_id:
        raise ValueError("motion plan id does not match its contents")
    return MotionPlan(
        plan_id,
        action,
        created_at,
        expires_at,
        git_sha,
        config_digest,
        robot,
        dict(target),
        accepted,
        reasons,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def calculate_plan_id(
    action: MotionAction,
    robot: RobotState,
    target: Mapping[str, Any],
    config_digest: str,
    git_sha: str,
) -> str:
    """计划号只取决于动作、快照、目标、配置和 Git，不受输出格式影响。"""

    payload = {
        "action": action.value,
        "robot_state": robot_state_dict(robot),
        "target": dict(target),
        "config_digest": config_digest,
        "git_sha": git_sha,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def make_motion_plan(
    action: MotionAction,
    robot: RobotState,
    target: Mapping[str, Any],
    config_digest: str,
    git_sha: str,
    created_at: float,
    accepted: bool,
    rejection_reasons: tuple[str, ...] = (),
    limits: PlanValidationLimits = PlanValidationLimits(),
) -> MotionPlan:
    if not math.isfinite(created_at):
        raise ValueError("created_at must be finite")
    if accepted and rejection_reasons:
        raise ValueError("an accepted plan cannot contain rejection reasons")
    return MotionPlan(
        plan_id=calculate_plan_id(action, robot, target, config_digest, git_sha),
        action=action,
        created_at=created_at,
        expires_at=created_at + limits.plan_max_age,
        git_sha=git_sha,
        config_digest=config_digest,
        robot=robot,
        target=dict(target),
        accepted=accepted,
        rejection_reasons=tuple(rejection_reasons),
    )


def _wrapped_error(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def validate_plan_for_execution(
    plan: MotionPlan,
    current: RobotState,
    *,
    now: float,
    git_sha: str,
    config_digest: str,
    target: Mapping[str, Any],
    limits: PlanValidationLimits = PlanValidationLimits(),
) -> PlanValidation:
    """按明确门槛复核计划；只返回原因，不尝试修复或重新规划。"""

    data_age = now - current.observed_at
    base_position_error = math.hypot(
        current.base.x - plan.robot.base.x,
        current.base.y - plan.robot.base.y,
    )
    base_yaw_error = _wrapped_error(current.base.yaw, plan.robot.base.yaw)
    slide_error = abs(current.slide - plan.robot.slide)
    max_arm_joint_error = max(
        abs(current_value - planned_value)
        for current_value, planned_value in zip(
            current.left_arm + current.right_arm,
            plan.robot.left_arm + plan.robot.right_arm,
        )
    )

    reasons: list[str] = []
    if not plan.accepted:
        reasons.append("计划本身未通过规划安全检查")
    if now < plan.created_at or now > plan.expires_at:
        reasons.append("计划已过期或系统时间异常")
    if data_age < 0.0 or data_age > limits.data_max_age:
        reasons.append(f"机器人数据过期：{data_age:.3f}s")
    if base_position_error > limits.base_position:
        reasons.append(f"底盘位置已变化：{base_position_error:.3f}m")
    if base_yaw_error > limits.base_yaw:
        reasons.append(f"底盘朝向已变化：{base_yaw_error:.3f}rad")
    if slide_error > limits.slide:
        reasons.append(f"升降位置已变化：{slide_error:.3f}m")
    if max_arm_joint_error > limits.arm_joint:
        reasons.append(f"机械臂关节已变化：{max_arm_joint_error:.3f}rad")
    if git_sha != plan.git_sha:
        reasons.append("Git 提交已变化")
    if config_digest != plan.config_digest:
        reasons.append("配置内容已变化")
    if _canonical_json(dict(target)) != _canonical_json(dict(plan.target)):
        reasons.append("动作目标已变化")

    return PlanValidation(
        accepted=not reasons,
        reasons=tuple(reasons),
        data_age=data_age,
        base_position_error=base_position_error,
        base_yaw_error=base_yaw_error,
        slide_error=slide_error,
        max_arm_joint_error=max_arm_joint_error,
    )
