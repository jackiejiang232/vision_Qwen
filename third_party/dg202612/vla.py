"""任务级 VLA 的受限动作 JSON 契约。

此处没有模型依赖，也没有控制发布器。未来 Qwen 的输出先在离线阶段被解析为
``ActionCandidate``，再经过 ``candidate.py`` 校验；自由文本永远不能直接控制机器人。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from .contracts import (
    ActionCandidate,
    ActionSkill,
    ExecutionFeedback,
    GraspProfile,
    Pose2D,
    Pose3D,
    SceneState,
)


class ActionSchemaError(ValueError):
    """模型输出不是允许的动作 JSON。"""


ACTION_VOCABULARY = tuple(skill.value for skill in ActionSkill)


def _mapping(raw: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionSchemaError("action candidate is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ActionSchemaError("action candidate must be a JSON object")
    return raw


def _number(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ActionSchemaError(f"{field} must be numeric") from exc


def _pose2(value: Any, field: str) -> Pose2D:
    if not isinstance(value, Mapping):
        raise ActionSchemaError(f"{field} must be an object")
    return Pose2D(_number(value.get("x"), f"{field}.x"), _number(value.get("y"), f"{field}.y"), _number(value.get("yaw"), f"{field}.yaw"))


def _pose3_or_none(value: Any, field: str) -> Pose3D | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ActionSchemaError(f"{field} must be an object or null")
    return Pose3D(
        _number(value.get("x"), f"{field}.x"),
        _number(value.get("y"), f"{field}.y"),
        _number(value.get("z"), f"{field}.z"),
        _number(value.get("roll", 0.0), f"{field}.roll"),
        _number(value.get("pitch", 0.0), f"{field}.pitch"),
        _number(value.get("yaw", 0.0), f"{field}.yaw"),
    )


def decode_action_candidate(raw: str | Mapping[str, Any]) -> ActionCandidate:
    """严格解码离线模型输出；字段缺失或多余都会被拒绝，便于人工审核。"""

    payload = _mapping(raw)
    allowed = {
        "skill",
        "target_id",
        "grasp_profile",
        "approach_pose",
        "place_pose",
        "recovery",
        "confidence",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ActionSchemaError(f"unexpected action fields: {sorted(unknown)}")
    required = allowed - {"place_pose"}
    missing = [field for field in required if field not in payload]
    if missing:
        raise ActionSchemaError(f"missing action fields: {sorted(missing)}")
    try:
        skill = ActionSkill(str(payload["skill"]))
        profile = GraspProfile(str(payload["grasp_profile"]))
    except ValueError as exc:
        raise ActionSchemaError("skill or grasp_profile is outside the action vocabulary") from exc
    return ActionCandidate(
        skill=skill,
        target_id=str(payload["target_id"]).strip(),
        grasp_profile=profile,
        approach_pose=_pose2(payload["approach_pose"], "approach_pose"),
        place_pose=_pose3_or_none(payload.get("place_pose"), "place_pose"),
        recovery=str(payload["recovery"]).strip(),
        confidence=_number(payload["confidence"], "confidence"),
    )


@dataclass(frozen=True)
class VLASample:
    """一条人工审核后的训练样本；图像只保留引用，权重和图像文件不进入代码包。"""

    sample_id: str
    image_refs: tuple[str, ...]
    raw_instruction: Mapping[str, Any]
    scene: SceneState
    candidate: ActionCandidate
    feedback: ExecutionFeedback
    random_seed: int | None
    reviewer: str

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.reviewer.strip():
            raise ValueError("sample_id and reviewer are required for a VLA sample")
        if not self.image_refs:
            raise ValueError("VLA sample requires at least one reviewed image reference")

    def action_json(self) -> dict[str, Any]:
        """导出监督目标，不包含底层控制量。"""

        candidate = self.candidate
        return {
            "skill": candidate.skill.value,
            "target_id": candidate.target_id,
            "grasp_profile": candidate.grasp_profile.value,
            "approach_pose": {
                "x": candidate.approach_pose.x,
                "y": candidate.approach_pose.y,
                "yaw": candidate.approach_pose.yaw,
            },
            "place_pose": None,
            "recovery": candidate.recovery,
            "confidence": candidate.confidence,
        }
