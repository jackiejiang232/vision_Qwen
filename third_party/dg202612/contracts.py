"""新框架的共享数据契约。

这个模块只定义各层交换的数据，不包含 ROS、路径规划或机械臂求解。
坐标约定固定为：所有 ``Pose`` 都在 world 坐标系中；箱体 ``pose`` 是几何中心，
``BoxSize.length/width/height`` 分别沿箱体局部 x/y/z 轴。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, TypeAlias


JointVector: TypeAlias = tuple[float, float, float, float, float, float]


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def joint_vector(values: tuple[float, ...] | list[float], name: str) -> JointVector:
    """把六轴关节值固定成不可变元组，避免左右臂顺序在模块间漂移。"""

    if len(values) != 6:
        raise ValueError(f"{name} must contain exactly six joints")
    return tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))  # type: ignore[return-value]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "yaw"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class Pose3D:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "roll", "pitch", "yaw"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


@dataclass(frozen=True)
class BoxSize:
    length: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("length", "width", "height"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


class TaskId(str, Enum):
    TASK_1 = "task_1"
    TASK_2 = "task_2"
    TASK_3 = "task_3"


class GraspProfile(str, Enum):
    TABLE_SIDE_HUG = "table_side_hug"
    SHELF_EXTRACT_HUG = "shelf_extract_hug"
    TABLE_TOP_HUG = "table_top_hug"


class ActionSkill(str, Enum):
    PICK = "pick"
    PLACE = "place"
    INSPECT = "inspect"
    RECOVER = "recover"


class CameraId(str, Enum):
    """场景层使用的三台逻辑相机。

    头部彩色图与对齐深度必须先同步成一份 RGB-D 观测，因此在核心数据流中算作
    一个逻辑来源；左右腕相机没有深度，只能提供局部接触证据。
    """

    HEAD_RGBD = "head_rgbd"
    LEFT_WRIST_RGB = "left_wrist_rgb"
    RIGHT_WRIST_RGB = "right_wrist_rgb"


class ObservationPurpose(str, Enum):
    """运动状态机请求视觉工作的原因，而不是自由文本命令。"""

    ACQUIRE_TARGET = "acquire_target"
    REFINE_TARGET = "refine_target"
    GUARD_APPROACH = "guard_approach"
    VERIFY_HOLD = "verify_hold"
    VERIFY_LIFT = "verify_lift"


class ExecutionPhase(str, Enum):
    BUILD_GOAL = "build_goal"
    PLAN_PICK = "plan_pick"
    NAVIGATE_PICK = "navigate_pick"
    DOCK_PICK = "dock_pick"
    REFINE_PICK = "refine_pick"
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    HOLD = "hold"
    VERIFY_HOLD = "verify_hold"
    LIFT = "lift"
    VERIFY_LIFT = "verify_lift"
    RETREAT_PICK = "retreat_pick"
    MINIMAL_DONE = "minimal_done"
    SAFE_STOP = "safe_stop"


class MotionAction(str, Enum):
    """人工逐项审核的底层动作；枚举值同时用于命令行和运行记录。"""

    # LOOK 是**前置观测**动作：只转头部两个关节，把目标拉进相机视野，然后才轮到
    # 视觉层识别。它必须是一个显式动作而不是藏在别的动作里的副作用，理由有三条：
    #   1) 上游交回来的 SceneState 隐含了「视觉看得见」这个前提，而这个前提在
    #      场景随机化之后不再自动成立（开局位姿看不见货架，货架方位角约 −77°，
    #      而头部偏航行程只有 ±28.6°）——前提要有人负责满足；
    #   2) 头部行程不够时必须有人报出「还差多少、需要底盘补多少偏航」，这个结论
    #      只有一个独立动作才有地方写；
    #   3) 与其余动作一样要留计划、要人工审核、要可复现，不能是隐式行为。
    # LOOK 只发布 head_yaw/head_pitch，不动底盘、不动升降、不动手臂。
    LOOK = "LOOK"
    DOCK = "DOCK"
    PREGRASP = "PREGRASP"
    APPROACH = "APPROACH"
    HOLD = "HOLD"
    LIFT = "LIFT"
    RETREAT = "RETREAT"
    SAFE_STOP = "SAFE_STOP"


class RecoveryCode(str, Enum):
    """方案 §13.2 允许的恢复动作，是运动层回给决策/视觉层的枚举信号。

    自由文本 ``reason`` 供人阅读，``recovery`` 供上游分支：不同失败对应不同恢复
    （§18.6：目标丢失→重观测、IK 无解→换站位、放置偏差→局部修正）。
    """

    REOBSERVE = "reobserve"
    REDOCK = "redock"
    REPLAN_PATH = "replan_path"
    RETRY_GRASP = "retry_grasp"
    RETRY_PLACE = "retry_place"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class ObjectState:
    """视觉层确认后的一个可抓取物体；``pose`` 是箱体中心而不是底面。"""

    object_id: str
    color: str
    pose: Pose3D
    size: BoxSize
    observed_at: float
    confidence: float = 1.0
    source_cameras: tuple[CameraId, ...] = ()
    position_std_m: float | None = None
    yaw_std_rad: float | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("object_id is required")
        if not self.color.strip():
            raise ValueError("color is required")
        object.__setattr__(self, "observed_at", _finite(self.observed_at, "observed_at"))
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        cameras = tuple(
            item if isinstance(item, CameraId) else CameraId(item)
            for item in self.source_cameras
        )
        if len(cameras) != len(set(cameras)):
            raise ValueError("source_cameras must not contain duplicates")
        object.__setattr__(self, "source_cameras", cameras)
        for name in ("position_std_m", "yaw_std_rad"):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if number < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)


@dataclass(frozen=True)
class CameraObservation:
    """融合进 ``SceneState`` 的最近一帧相机元数据，不保存图像本身。"""

    camera: CameraId
    observed_at: float
    frame_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.camera, CameraId):
            object.__setattr__(self, "camera", CameraId(self.camera))
        object.__setattr__(self, "observed_at", _finite(self.observed_at, "observed_at"))
        object.__setattr__(self, "frame_id", str(self.frame_id))


@dataclass(frozen=True)
class ObservationRequest:
    """执行器向视觉/运动协调层提出的一次明确观测请求。

    ``requested_base_pose`` 与 ``look_at`` 是主动感知建议：视角不足时可先移动到底盘
    观察位、转头看目标，再由视觉层返回新的 ``SceneState``。视觉层本身不发布运动
    命令。
    """

    purpose: ObservationPurpose
    target_id: str
    cameras: tuple[CameraId, ...]
    max_age: float
    min_confidence: float
    require_target_pose: bool = True
    max_position_std_m: float | None = None
    max_yaw_std_rad: float | None = None
    require_stationary: bool = True
    max_base_linear: float = 0.02
    max_base_angular: float = 0.02
    requested_base_pose: Pose2D | None = None
    look_at: Pose3D | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, ObservationPurpose):
            object.__setattr__(self, "purpose", ObservationPurpose(self.purpose))
        if not self.target_id.strip():
            raise ValueError("observation target_id is required")
        cameras = tuple(
            item if isinstance(item, CameraId) else CameraId(item)
            for item in self.cameras
        )
        if not cameras or len(cameras) != len(set(cameras)):
            raise ValueError("observation cameras must be non-empty and unique")
        object.__setattr__(self, "cameras", cameras)
        max_age = _finite(self.max_age, "max_age")
        if max_age <= 0.0:
            raise ValueError("max_age must be positive")
        object.__setattr__(self, "max_age", max_age)
        confidence = _finite(self.min_confidence, "min_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        object.__setattr__(self, "min_confidence", confidence)
        for name in (
            "max_position_std_m",
            "max_yaw_std_rad",
            "max_base_linear",
            "max_base_angular",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            number = _finite(value, name)
            if number < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)


@dataclass(frozen=True)
class GraspEvidence:
    """腕部视觉在接近、抱持和抬升检查点给出的结构化证据。"""

    target_id: str
    observed_at: float
    source_cameras: tuple[CameraId, ...]
    safe_to_continue: bool
    left_contact_confirmed: bool = False
    right_contact_confirmed: bool = False
    centered_error_m: float | None = None
    object_lifted: bool = False

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("grasp evidence target_id is required")
        object.__setattr__(self, "observed_at", _finite(self.observed_at, "observed_at"))
        cameras = tuple(
            item if isinstance(item, CameraId) else CameraId(item)
            for item in self.source_cameras
        )
        if not cameras or len(cameras) != len(set(cameras)):
            raise ValueError("grasp evidence cameras must be non-empty and unique")
        object.__setattr__(self, "source_cameras", cameras)
        if self.centered_error_m is not None:
            error = _finite(self.centered_error_m, "centered_error_m")
            if error < 0.0:
                raise ValueError("centered_error_m cannot be negative")
            object.__setattr__(self, "centered_error_m", error)


@dataclass(frozen=True)
class RobotState:
    """同一时刻取得的底盘、升降和双臂反馈快照。"""

    base: Pose2D
    base_linear: float
    base_angular: float
    slide: float
    head_yaw: float
    head_pitch: float
    left_arm: JointVector
    left_gripper: float
    right_arm: JointVector
    right_gripper: float
    observed_at: float

    def __post_init__(self) -> None:
        for name in (
            "base_linear",
            "base_angular",
            "slide",
            "head_yaw",
            "head_pitch",
            "left_gripper",
            "right_gripper",
            "observed_at",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "left_arm", joint_vector(self.left_arm, "left_arm"))
        object.__setattr__(self, "right_arm", joint_vector(self.right_arm, "right_arm"))


@dataclass(frozen=True)
class ShelfState:
    """货架的层级记忆，由视觉层填充；运动层只读取，不自行推断层高。

    ``empty_levels`` 是可放置的空层编号；``obstacle_level`` 是白色长方体障碍物
    所在层（任务三的参照）；``level_poses`` 可选，给出每层中心的 world 位姿，
    便于放置时对齐层高。字段留空表示视觉层尚未确认，而不是默认某个值。
    """

    empty_levels: tuple[int, ...] = ()
    obstacle_level: int | None = None
    level_poses: Mapping[int, Pose3D] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "empty_levels", tuple(int(level) for level in self.empty_levels))
        if self.obstacle_level is not None:
            object.__setattr__(self, "obstacle_level", int(self.obstacle_level))


@dataclass(frozen=True)
class TableState:
    """桌面的场景记忆，由视觉层填充。

    ``side_original_pose`` 记录任务一开始时桌边彩色箱的原始位姿，任务二"放回原位"
    需要它；``side`` 标注彩色箱位于白色正方位的左侧还是右侧。
    """

    side_original_pose: Pose3D | None = None
    side: str | None = None

    def __post_init__(self) -> None:
        if self.side is not None:
            side = str(self.side).strip().lower()
            if side not in {"left", "right"}:
                raise ValueError("table side must be left or right")
            object.__setattr__(self, "side", side)


@dataclass(frozen=True)
class SceneState:
    """供规划使用的场景快照，不负责从相机或 ROS 话题生成自己。

    ``shelf``/``table`` 是视觉层维护的场景记忆，运动层只读；为空表示尚未确认。
    """

    timestamp: float
    robot: RobotState
    objects: tuple[ObjectState, ...]
    instruction: Mapping[str, Any] | None = None
    carry_object_id: str | None = None
    last_failure: str | None = None
    shelf: ShelfState | None = None
    table: TableState | None = None
    camera_observations: tuple[CameraObservation, ...] = ()
    grasp_evidence: GraspEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        ids = [item.object_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("objects must have unique object_id values")
        observations = tuple(self.camera_observations)
        cameras = [item.camera for item in observations]
        if len(cameras) != len(set(cameras)):
            raise ValueError("camera_observations must contain one latest frame per camera")
        object.__setattr__(self, "camera_observations", observations)

    def object_by_id(self, object_id: str) -> ObjectState | None:
        return next((item for item in self.objects if item.object_id == object_id), None)

    def camera_by_id(self, camera: CameraId) -> CameraObservation | None:
        return next(
            (item for item in self.camera_observations if item.camera is camera),
            None,
        )


@dataclass(frozen=True)
class PickPlaceGoal:
    """视觉和语言已经消歧后的具体物体目标；不保存任务专用路径。

    ``place_type`` 来自官方指令（如 shelf_empty_level / table_side_original），
    与 ``place_pose`` 一起描述"放到哪"；运动层据此选择放置流程，但不解析语义。
    """

    task_id: TaskId
    target_id: str
    target_color: str
    target_pose: Pose3D
    target_size: BoxSize
    source_area: str
    grasp_profile: GraspProfile
    place_pose: Pose3D | None = None
    place_type: str | None = None
    retry_limit: int = 0

    def __post_init__(self) -> None:
        if not self.target_id.strip() or not self.target_color.strip():
            raise ValueError("target_id and target_color are required")
        if not self.source_area.strip():
            raise ValueError("source_area is required")
        if self.retry_limit < 0:
            raise ValueError("retry_limit cannot be negative")
        if self.place_type is not None:
            object.__setattr__(self, "place_type", str(self.place_type).strip() or None)


@dataclass(frozen=True)
class ActionCandidate:
    """VLA 或规则层提出的高层候选，必须经校验后才可进入执行器。"""

    skill: ActionSkill
    target_id: str
    grasp_profile: GraspProfile
    approach_pose: Pose2D
    place_pose: Pose3D | None
    recovery: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("target_id is required")
        if not self.recovery.strip():
            raise ValueError("recovery is required")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class RobotTargets:
    """具名控制目标。ROS 适配器负责把它转换成旧的 19 维顺序。"""

    base_linear: float
    base_angular: float
    slide: float
    head_yaw: float
    head_pitch: float
    left_arm: JointVector
    left_gripper: float
    right_arm: JointVector
    right_gripper: float

    def __post_init__(self) -> None:
        for name in (
            "base_linear",
            "base_angular",
            "slide",
            "head_yaw",
            "head_pitch",
            "left_gripper",
            "right_gripper",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "left_arm", joint_vector(self.left_arm, "left_arm"))
        object.__setattr__(self, "right_arm", joint_vector(self.right_arm, "right_arm"))

    def with_stopped_base(self) -> "RobotTargets":
        """机械臂阶段调用此方法，明确让底盘速度归零。"""

        return RobotTargets(
            base_linear=0.0,
            base_angular=0.0,
            slide=self.slide,
            head_yaw=self.head_yaw,
            head_pitch=self.head_pitch,
            left_arm=self.left_arm,
            left_gripper=self.left_gripper,
            right_arm=self.right_arm,
            right_gripper=self.right_gripper,
        )


@dataclass(frozen=True)
class ExecutionFeedback:
    """一个执行阶段回给决策/视觉层的结果。

    ``need_reobserve`` 提示上游在重规划前先重新观测；``position_error``/
    ``angle_error``/``joint_error`` 是可选的量化偏差，驱动 §18.6 的分类恢复，
    为空表示该阶段不产生该项误差。
    """

    phase: ExecutionPhase
    completed: bool
    failed: bool
    reason: str
    timestamp: float
    held_object_id: str | None = None
    need_reobserve: bool = False
    position_error: float | None = None
    angle_error: float | None = None
    joint_error: float | None = None
    recovery: RecoveryCode | None = None
    observation_request: ObservationRequest | None = None

    def __post_init__(self) -> None:
        if self.completed and self.failed:
            raise ValueError("feedback cannot be both completed and failed")
        if not self.reason.strip():
            raise ValueError("feedback reason is required")
        object.__setattr__(self, "timestamp", _finite(self.timestamp, "timestamp"))
        for name in ("position_error", "angle_error", "joint_error"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
