"""官方指令 JSON 与新框架之间的窄接口。

这里不做自然语言推理，也不猜测物体坐标。它只验证任务编号、目标颜色等
已明确字段；视觉层选定具体对象后，才由 ``goal_from_intent`` 形成可执行目标。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol

from .contracts import GraspProfile, ObjectState, PickPlaceGoal, TaskId


class InstructionError(ValueError):
    """官方输入缺字段或字段不能安全解释。"""


class InstructionSource(Protocol):
    """未来 ROS 订阅器应实现的只读边界。"""

    def latest(self) -> str | Mapping[str, Any] | None:
        ...


@dataclass(frozen=True)
class InstructionIntent:
    task_id: TaskId
    target_color: str
    target_kind: str | None
    raw: Mapping[str, Any]
    place_type: str | None = None


def _mapping(raw: str | Mapping[str, Any] | list[Any]) -> Mapping[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InstructionError("instruction is not valid JSON") from exc
    if isinstance(raw, list):
        if len(raw) != 1 or not isinstance(raw[0], Mapping):
            raise InstructionError("new framework accepts one instruction object at a time")
        raw = raw[0]
    if not isinstance(raw, Mapping):
        raise InstructionError("instruction must be an object")
    return raw


def _task_id(value: Any) -> TaskId:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise InstructionError("task or task_number is required") from exc
    mapping = {1: TaskId.TASK_1, 2: TaskId.TASK_2, 3: TaskId.TASK_3}
    try:
        return mapping[number]
    except KeyError as exc:
        raise InstructionError(f"unsupported task number: {number}") from exc


def parse_official_instruction(raw: str | Mapping[str, Any] | list[Any]) -> InstructionIntent:
    """读取旧 baseline 已使用的 ``task``/``target.target_color`` 字段形状。"""

    payload = _mapping(raw)
    target = payload.get("target")
    target = target if isinstance(target, Mapping) else payload
    task_value = payload.get("task", payload.get("task_number", payload.get("id")))
    color = str(target.get("target_color", payload.get("target_color", ""))).strip()
    if not color:
        raise InstructionError("target_color is required")
    kind = str(target.get("target_kind", "")).strip() or None
    place = payload.get("place")
    place = place if isinstance(place, Mapping) else payload
    place_type = str(place.get("place_type", payload.get("place_type", ""))).strip() or None
    return InstructionIntent(_task_id(task_value), color, kind, dict(payload), place_type)


def goal_from_intent(
    intent: InstructionIntent,
    target: ObjectState,
    source_area: str,
    grasp_profile: GraspProfile,
) -> PickPlaceGoal:
    """只在颜色与视觉目标一致时创建目标，防止语言直接控制一个未确认物体。"""

    if target.color != intent.target_color:
        raise InstructionError(
            f"instruction requests {intent.target_color}, but target is {target.color}"
        )
    return PickPlaceGoal(
        task_id=intent.task_id,
        target_id=target.object_id,
        target_color=target.color,
        target_pose=target.pose,
        target_size=target.size,
        source_area=source_area,
        grasp_profile=grasp_profile,
        place_type=intent.place_type,
    )
