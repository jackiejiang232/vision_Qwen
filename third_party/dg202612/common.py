"""共享数据与运行记录。

本模块只定义任务、检测、控制命令和状态机等基础对象，不依赖 ROS 或仿真环境。
因此单元测试可以直接在普通 Python 环境中运行。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence


class TaskPhase(str, Enum):
    """一次抓取放置任务的阶段。"""

    WAIT_READY = "WAIT_READY"
    PARSE_TASK = "PARSE_TASK"
    OBSERVE_TARGET = "OBSERVE_TARGET"
    LOCK_TARGET = "LOCK_TARGET"
    NAVIGATE_PICK = "NAVIGATE_PICK"
    PREGRASP = "PREGRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    VERIFY_CARRY = "VERIFY_CARRY"
    RETREAT = "RETREAT"
    NAVIGATE_PLACE = "NAVIGATE_PLACE"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    RETREAT_SAFE = "RETREAT_SAFE"
    RETURN_END = "RETURN_END"
    DONE = "DONE"
    SAFE_STOP = "SAFE_STOP"


_PHASE_SEQUENCE = (
    TaskPhase.WAIT_READY,
    TaskPhase.PARSE_TASK,
    TaskPhase.OBSERVE_TARGET,
    TaskPhase.LOCK_TARGET,
    TaskPhase.NAVIGATE_PICK,
    TaskPhase.PREGRASP,
    TaskPhase.GRASP,
    TaskPhase.LIFT,
    TaskPhase.RETREAT,
    TaskPhase.VERIFY_CARRY,
    TaskPhase.NAVIGATE_PLACE,
    TaskPhase.PLACE,
    TaskPhase.RELEASE,
    TaskPhase.RETREAT_SAFE,
    TaskPhase.RETURN_END,
    TaskPhase.DONE,
)

ALLOWED_TRANSITIONS = {
    phase: {next_phase, TaskPhase.SAFE_STOP}
    for phase, next_phase in zip(_PHASE_SEQUENCE, _PHASE_SEQUENCE[1:])
}
ALLOWED_TRANSITIONS[TaskPhase.VERIFY_CARRY].add(TaskPhase.NAVIGATE_PICK)
ALLOWED_TRANSITIONS[TaskPhase.DONE] = set()
ALLOWED_TRANSITIONS[TaskPhase.SAFE_STOP] = set()


def assert_transition(previous: TaskPhase, current: TaskPhase) -> None:
    """拒绝跳过步骤或从终态重新启动。"""

    if current not in ALLOWED_TRANSITIONS[previous]:
        raise ValueError(
            f"illegal task phase transition: {previous.value} -> {current.value}"
        )


@dataclass(frozen=True)
class TaskSpec:
    """从赛事指令中提取出的任务。"""

    task_number: int
    instruction: str
    target_kind: str
    target_body: str
    target_color: str
    place_type: str
    place_world: tuple[float, float, float] | None
    place_radius: float
    reference_prop: str | None = None
    target_side: str | None = None
    reference_body: str | None = None
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionObservation:
    """一次世界坐标系中的目标检测。"""

    class_id: str
    score: float
    world: tuple[float, float, float]
    stamp: float
    frame_id: str

    def age(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        return max(0.0, current - self.stamp)


@dataclass(frozen=True)
class ControlCommand:
    """赛事机器人使用的 19 维控制向量。"""

    values: tuple[float, ...]
    created_at: float

    SIZE = 19

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
        created_at: float | None = None,
    ) -> "ControlCommand":
        data = tuple(float(value) for value in values)
        if len(data) != cls.SIZE:
            raise ValueError(f"control command must contain {cls.SIZE} values")
        if not all(math.isfinite(value) for value in data):
            raise ValueError("control command contains a non-finite value")
        stamp = time.monotonic() if created_at is None else float(created_at)
        return cls(data, stamp)

    def to_dict(self) -> Mapping[str, Any]:
        return {"values": list(self.values), "created_at": self.created_at}


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"cannot serialize {type(value)!r}")


class RunLogger:
    """把一次运行的元数据、事件和得分写入 JSON 文件。"""

    def __init__(self, run_dir: str | os.PathLike[str] | None = None) -> None:
        path = run_dir or os.environ.get("DG_RUN_DIR", "logs/runs/manual")
        self.run_dir = Path(path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write_metadata(self, extra: dict[str, Any] | None = None) -> None:
        metadata = {
            "git_sha": os.environ.get("DG_GIT_SHA", "unknown"),
            "git_dirty": os.environ.get("DG_GIT_DIRTY", "unknown"),
            "image_digest": os.environ.get("DG_IMAGE_DIGEST", "unknown"),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "99"),
            "rmw_implementation": os.environ.get(
                "RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"
            ),
            "run_mode": os.environ.get("DG_RUN_MODE", "observe"),
            "randomize": os.environ.get("DG_RANDOMIZE", "0"),
            "seed": os.environ.get("DG_SEED", "0"),
            "detection_backend": os.environ.get("DG_DETECTION_BACKEND", "color"),
            "started_unix": time.time(),
        }
        metadata.update(extra or {})
        target = self.run_dir / "metadata.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def write_result(self, result: dict[str, Any]) -> None:
        """原子写入本局最终结果，避免中途停止留下半个 JSON 文件。"""

        target = self.run_dir / "result.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def event(self, kind: str, **payload: Any) -> None:
        self._append("events.jsonl", {"time": time.time(), "kind": kind, **payload})

    def score(self, score: int, **payload: Any) -> None:
        self._append(
            "score.jsonl",
            {"time": time.time(), "score": int(score), **payload},
        )

    def _append(self, filename: str, item: dict[str, Any]) -> None:
        line = json.dumps(item, ensure_ascii=False, default=_json_default) + "\n"
        with self._lock:
            with (self.run_dir / filename).open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
