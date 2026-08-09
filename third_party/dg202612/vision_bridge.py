"""视觉层 ``/vlm/scene_understanding`` 报文 → 运动层数据结构的适配层。

这个模块只做一件事：把视觉同学发布的语义 JSON **原样读进来**，做类型与量纲上的
清洗，然后交给运动层的 :mod:`dg202612.contracts`。它刻意不做的事同样重要：

* 不猜任何坐标、尺寸、层高。视觉没给的字段就是没给，缺就报错或返回 ``None``，
  绝不填一个"看起来合理"的默认值——一个被默认值掩盖的缺字段，在真机上表现为
  机械臂伸向一个从未被观测过的位置。
* 不做语义决策。选哪个任务、抓哪个箱子是上游（VLA/决策层）的结论，本模块只把
  这些结论翻译成运动层能审核的结构。
* 不发运动命令。需要重新观测时，本模块给出的是一个 :class:`ObservationRequest`
  形式的**请求**，由运动层的 ``LOOK`` 动作去满足（见 ``MotionAction.LOOK``）。

报文来源与本模块要处理的几处已知偏差
------------------------------------
依据 ``Qwen_优化后_scene_understanding_JSON解析文档``（视觉同学 2026-07-29 交付）
的实测输出，有五处必须在边界上处理，否则会直接把运动层打崩或者打歪：

1. ``grounding.confidence = -999.0``：这是"未选中"的哨兵值，不是置信度。直接
   丢给 ``ObjectState.confidence`` 会命中 ``0 <= c <= 1`` 校验并抛 ``ValueError``。
   本模块把它翻译成 ``None``（= 未知），并且**不**顺手改成 1.0。
2. ``sam_score = 1.0087``：分割质量分会略微超过 1.0。契约层的置信度区间是
   [0, 1]，因此需要显式截断，并且记录"截断过"这件事。
3. ``objects`` 里含 ``shelf``（``semantic_role = context_object``）：货架是环境
   结构，不是可抓物。它必须被挡在抓取候选之外——运动层如果对着货架本体做 IK，
   得到的是一个 194272 像素大的"箱子"。
4. ``shelf_layer`` 用的是 6 个板面的编号，而比赛语义讲的是三层货架。两套编号
   不能直接互认，见 :class:`ShelfLayerResolver` 的说明。
5. 时间戳分成 ``source_stamp_sec`` / ``source_stamp_nanosec`` 两个字段，而运动层
   的新鲜度检查（``scene.freshness``）要的是一个浮点秒。

另外一条读取顺序上的结论，直接来自该文档第七节：**按 ``active_task_id`` 去
``task_queue`` 里找当前任务**，而不是只看 ``grounding.selected_object_id``。原因
是 grounding 只描述"当前这一帧绑定成功没有"，而 task_queue 描述"三条任务各自
还缺什么"。只看 grounding，任务 2 的目标（已经检测到了）会被当成不存在。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    BoxSize,
    CameraId,
    ObjectState,
    ObservationPurpose,
    ObservationRequest,
    Pose3D,
)

__all__ = [
    "INVALID_CONFIDENCE_SENTINELS",
    "SemanticRole",
    "ShelfLayerResolver",
    "ShelfLayerReading",
    "VisionMessageError",
    "VisionObject",
    "VisionTask",
    "SceneUnderstanding",
    "parse_scene_understanding",
    "parse_confidence",
    "stamp_seconds",
    "to_object_states",
    "reobserve_request",
]


class VisionMessageError(ValueError):
    """视觉报文缺字段、类型不对或量纲不对；调用方应当当成一次观测失败处理。"""


# 视觉层用来表达"这个数没有意义"的哨兵值。之所以列成集合而不是写 ``< 0`` 判断，
# 是因为哨兵和"真的算出来一个负数"是两回事：前者要翻译成"未知"，后者是视觉层的
# bug，应该报出来而不是悄悄归零。文档第八节已建议上游把 -999.0 改成 0.0/null，
# 改完之后这里保留兼容即可，不需要跟着改。
INVALID_CONFIDENCE_SENTINELS: frozenset[float] = frozenset({-999.0, -1.0})


class SemanticRole(str, Enum):
    """视觉层给每个检测对象打的语义角色。

    只有 ``ACTIVE_TASK_TARGET`` 和 ``FUTURE_TASK_TARGET`` 可能成为抓取候选；
    ``CONTEXT_OBJECT`` 是参考物或环境结构（白色圆柱、货架本体），运动层必须过滤。
    """

    ACTIVE_TASK_TARGET = "active_task_target"
    FUTURE_TASK_TARGET = "future_task_target"
    CONTEXT_OBJECT = "context_object"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: Any) -> "SemanticRole":
        text = str(value or "").strip().lower()
        try:
            return cls(text)
        except ValueError:
            # 未知角色不报错、也不当成可抓：新增角色时默认最保守的一侧。
            return cls.UNKNOWN


# 即便语义角色标错了，这些标签也永远不是可抓物。这是第二道闸：角色是模型输出，
# 会错；"货架不是箱子"是场景事实，不会错。
NEVER_GRASPABLE_LABELS: frozenset[str] = frozenset(
    {"shelf", "table", "wall", "floor", "货架", "桌子"}
)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisionMessageError(f"{field} must be an object")
    return value


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VisionMessageError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise VisionMessageError(f"{field} must be finite")
    return number


def _text(value: Any) -> str:
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise VisionMessageError(f"{field} must be an integer or null") from exc


def _string_tuple(value: Any) -> tuple[str, ...]:
    """``uncertainties`` 既可能是列表，也可能是 ``"a; b"`` 这样的一行文本。"""

    if value is None:
        return ()
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(",", ";").split(";")]
        return tuple(part for part in parts if part)
    if isinstance(value, Sequence):
        return tuple(_text(item) for item in value if _text(item))
    raise VisionMessageError("uncertainties must be a list or a string")


def stamp_seconds(payload: Mapping[str, Any]) -> float:
    """把 ROS 的 ``sec`` / ``nanosec`` 合成一个浮点秒。

    运动层的新鲜度检查（``dg202612.scene.freshness``）比较的是单个浮点时间戳，
    所以合成必须在边界上做一次，而不是让每个调用点各自 ``sec + nsec/1e9``——那
    是同一个换算写 N 遍、错一遍就查半天的典型来源。
    """

    seconds = _number(payload.get("source_stamp_sec", 0.0), "source_stamp_sec")
    nanoseconds = _number(
        payload.get("source_stamp_nanosec", 0.0), "source_stamp_nanosec"
    )
    if nanoseconds < 0.0:
        raise VisionMessageError("source_stamp_nanosec cannot be negative")
    return seconds + nanoseconds * 1e-9


def parse_confidence(value: Any, field: str = "confidence") -> tuple[float | None, bool]:
    """清洗一个置信度，返回 ``(值或 None, 是否被截断)``。

    三种输入分开处理，因为它们代表三件不同的事：

    * ``None`` 或哨兵值（见 :data:`INVALID_CONFIDENCE_SENTINELS`）→ ``(None, False)``：
      未知。调用方必须自己决定"未知"能不能继续，本模块不替它决定。
    * 略微越界（例如 SAM 的 1.0087）→ 截到 [0, 1] 并把第二个返回值置 ``True``，
      让调用方有机会把"我截断过"记进日志。
    * 明显越界（例如 5.0 或 -3.0）→ 同样截断并标记。真要区分"轻微数值溢出"和
      "上游算错了"，靠的是看日志里截断了多少，而不是在这里定一个魔法阈值。
    """

    if value is None:
        return None, False
    number = _number(value, field)
    if number in INVALID_CONFIDENCE_SENTINELS:
        return None, False
    clamped = min(1.0, max(0.0, number))
    return clamped, clamped != number


@dataclass(frozen=True)
class ShelfLayerReading:
    """一次货架层号解析的结果，故意把"视觉说的"和"量出来的"分开放。

    ``vision_layer`` 是视觉报文里的 ``shelf_layer``，``board_index`` 是本模块用
    ``shelf_surface_z`` 去官方 MJCF 的板面高度表里比对出来的下标。两者一致
    (``consistent``) 才说明两边编号规则确实是已知的那条偏移。
    """

    vision_layer: int | None
    surface_z: float | None
    board_index: int | None
    consistent: bool
    note: str = ""


@dataclass(frozen=True)
class ShelfLayerResolver:
    """把视觉层的货架层号对回官方板面。

    为什么需要它：视觉报文里的 ``shelf_layer`` 是**视觉自己的编号**（实测
    ``surface_z=0.403 → shelf_layer=2``、``surface_z=0.732 → shelf_layer=3``），
    而官方 MJCF 里货架有 6 块板，中心高度是
    ``[0.403, 0.732, 1.061, 1.366, 1.695, 2.024]``（见
    ``config/scene_geometry.json`` 的 ``fixed_geometry.shelf.board_center_z``）。
    两个实测样本对上的是 ``board_index = shelf_layer - 2``，也就是视觉把地面算作
    了第 1 层。

    **两个样本推不出一条规则**，所以本类不把偏移写死成常量，而是：以
    ``surface_z`` 去板面高度表里做最近匹配（这一步是可测量、可复现的），再回头
    检查 ``shelf_layer`` 是否等于 ``board_index + offset``。不一致时不抛异常，只
    把 ``consistent`` 置 ``False`` 并写明原因——层号对不上是要人去看的事，不是运动
    层可以就地决定的事。

    还有一件**没有解决**的事，必须留在这里而不是假装解决了：6 块板面到"比赛
    语义里的三层货架"的映射目前没有证据。哪三块板是任务允许放置的层，需要一次
    官方场景确认（步骤见 ``docs/当前进度与验收.md``）。在那之前
    :meth:`competition_level` 一律返回 ``None``。
    """

    board_center_z: tuple[float, ...]
    tolerance_m: float = 0.03
    vision_layer_offset: int = 2

    def __post_init__(self) -> None:
        boards = tuple(float(item) for item in self.board_center_z)
        if not boards:
            raise VisionMessageError("board_center_z must not be empty")
        if any(not math.isfinite(item) for item in boards):
            raise VisionMessageError("board_center_z must be finite")
        object.__setattr__(self, "board_center_z", boards)
        if self.tolerance_m <= 0.0:
            raise VisionMessageError("tolerance_m must be positive")

    @staticmethod
    def from_scene_geometry(config: Mapping[str, Any]) -> "ShelfLayerResolver":
        """从 ``config/scene_geometry.json`` 已加载的字典里取板面高度。"""

        fixed = config.get("fixed_geometry")
        shelf = _mapping(fixed, "fixed_geometry").get("shelf") if isinstance(fixed, Mapping) else None
        boards = _mapping(shelf, "fixed_geometry.shelf").get("board_center_z")
        if not isinstance(boards, Sequence) or isinstance(boards, (str, bytes)):
            raise VisionMessageError("fixed_geometry.shelf.board_center_z must be a list")
        return ShelfLayerResolver(
            board_center_z=tuple(
                _number(item, "board_center_z") for item in boards
            )
        )

    def resolve(self, vision_layer: Any, surface_z: Any) -> ShelfLayerReading:
        layer = _optional_int(vision_layer, "shelf_layer")
        height = None if surface_z is None else _number(surface_z, "shelf_surface_z")
        if height is None:
            return ShelfLayerReading(layer, None, None, False, "缺 shelf_surface_z，无法核对层号")
        distances = [abs(height - board) for board in self.board_center_z]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        if distances[nearest] > self.tolerance_m:
            return ShelfLayerReading(
                layer,
                height,
                None,
                False,
                f"surface_z={height:.4f} 与任何官方板面的距离都超过 {self.tolerance_m:.3f} m",
            )
        if layer is None:
            return ShelfLayerReading(layer, height, nearest, False, "视觉未给出 shelf_layer")
        expected = nearest + self.vision_layer_offset
        if layer != expected:
            return ShelfLayerReading(
                layer,
                height,
                nearest,
                False,
                f"shelf_layer={layer} 与按高度匹配到的板面 {nearest}（期望 {expected}）不一致",
            )
        return ShelfLayerReading(layer, height, nearest, True)

    def competition_level(self, reading: ShelfLayerReading) -> int | None:
        """板面下标 → 比赛语义的三层货架层号。**目前一律返回 ``None``。**

        6 块板到 3 层的对应关系还没有任何证据支撑（哪几块板在机械臂可达高度内、
        任务允许放在哪几层，都要在官方场景里确认过才能写）。在拿到证据之前返回
        ``None``，比返回一个猜出来的 1/2/3 安全：``None`` 会让放置流程停下来问人，
        猜错则会让机械臂把箱子放到错误的层上而且没人知道。
        """

        _ = reading
        return None


@dataclass(frozen=True)
class VisionObject:
    """视觉报文 ``objects[]`` 里的一项，清洗后的形态。"""

    object_id: str
    label: str
    semantic_role: SemanticRole
    pose_world: tuple[float, float, float]
    detection_confidence: float | None
    segmentation_confidence: float | None
    on_shelf: bool
    shelf_layer: ShelfLayerReading
    confidence_clamped: bool = False

    @property
    def color(self) -> str:
        """从 ``label``（形如 ``"brown box"``）里取颜色词。

        取第一个词而不是做颜色词表匹配：颜色词表一旦写死，视觉换个说法就漏。
        取不到就返回空串，由调用方决定这算不算致命。
        """

        parts = self.label.split()
        return parts[0].lower() if parts else ""

    @property
    def graspable(self) -> bool:
        """能否作为抓取候选。两道闸都要过。"""

        if self.label.strip().lower() in NEVER_GRASPABLE_LABELS:
            return False
        return self.semantic_role in {
            SemanticRole.ACTIVE_TASK_TARGET,
            SemanticRole.FUTURE_TASK_TARGET,
        }


@dataclass(frozen=True)
class VisionTask:
    """``task_queue[]`` 里的一条任务。"""

    task_id: int
    status: str
    target_label: str
    target_object_id: str | None
    place_type: str | None
    reference_label: str | None
    reference_object_id: str | None
    spatial_relation: str | None
    requires_reobserve: bool
    requires_scene_memory: bool
    uncertainties: tuple[str, ...] = ()

    @property
    def target_detected(self) -> bool:
        return bool(self.target_object_id)

    @property
    def executable(self) -> bool:
        """目标已绑定、且视觉自己没有要求重看，才允许进入抓取流程。

        注意 ``requires_scene_memory`` 不在这里判断：它挡的是**放置**（任务 2 要
        知道粉色箱原来在桌上的哪儿），挡不住抓取。把两件事混在一个布尔里，会让
        "能抓但还不知道放哪"被误报成"什么都不能做"。
        """

        return self.target_detected and not self.requires_reobserve


@dataclass(frozen=True)
class SceneUnderstanding:
    """一帧 ``/vlm/scene_understanding`` 报文，清洗后的完整形态。"""

    schema_version: str
    observed_at: float
    active_task_id: int | None
    scene_summary: str
    objects: tuple[VisionObject, ...]
    tasks: tuple[VisionTask, ...]
    selected_object_id: str | None
    selected_label: str | None
    grounding_confidence: float | None
    requires_reobserve: bool
    grounding_reason: str = ""
    warnings: tuple[str, ...] = ()

    def object_by_id(self, object_id: str | None) -> VisionObject | None:
        if not object_id:
            return None
        return next(
            (item for item in self.objects if item.object_id == object_id), None
        )

    def task_by_id(self, task_id: int | None) -> VisionTask | None:
        if task_id is None:
            return None
        return next((item for item in self.tasks if item.task_id == task_id), None)

    @property
    def active_task(self) -> VisionTask | None:
        """当前应处理的任务。

        这是文档第七节给动作模块的建议读法：按 ``active_task_id`` 去
        ``task_queue`` 里找，而不是只看 ``grounding.selected_object_id``。
        """

        return self.task_by_id(self.active_task_id)

    def graspable_objects(self) -> tuple[VisionObject, ...]:
        """可作为抓取候选的对象；货架、参考物已被剔除。"""

        return tuple(item for item in self.objects if item.graspable)

    def blocking_reasons(self) -> tuple[str, ...]:
        """当前帧为什么还不能进入抓取；为空表示可以往下走。"""

        reasons: list[str] = []
        if self.active_task_id is None:
            reasons.append("视觉未给出 active_task_id")
        task = self.active_task
        if task is None:
            if self.active_task_id is not None:
                reasons.append(f"task_queue 里没有 task_id={self.active_task_id}")
        else:
            if not task.target_detected:
                reasons.append(f"任务 {task.task_id} 的目标 {task.target_label} 未检测到")
            if task.requires_reobserve:
                reasons.append(f"任务 {task.task_id} 要求重新观测")
            target = self.object_by_id(task.target_object_id)
            if task.target_detected and target is None:
                reasons.append(
                    f"任务 {task.task_id} 绑定的 object_id 不在 objects 列表里"
                )
            elif target is not None and not target.graspable:
                reasons.append(
                    f"任务 {task.task_id} 绑定的对象 {target.label} 不是可抓物"
                )
        if self.requires_reobserve and not reasons:
            reasons.append("grounding 要求重新观测")
        return tuple(reasons)


def _parse_object(
    payload: Mapping[str, Any], index: int, resolver: ShelfLayerResolver | None
) -> tuple[VisionObject, tuple[str, ...]]:
    field = f"objects[{index}]"
    object_id = _text(payload.get("object_id"))
    if not object_id:
        raise VisionMessageError(f"{field}.object_id is required")
    pose = payload.get("pose_world")
    if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) != 3:
        raise VisionMessageError(f"{field}.pose_world must contain exactly three numbers")
    dino, dino_clamped = parse_confidence(
        payload.get("dino_score"), f"{field}.dino_score"
    )
    sam, sam_clamped = parse_confidence(payload.get("sam_score"), f"{field}.sam_score")
    warnings: list[str] = []
    if dino_clamped:
        warnings.append(f"{field}.dino_score 越界，已截断到 [0,1]")
    if sam_clamped:
        warnings.append(f"{field}.sam_score 越界，已截断到 [0,1]")
    reading = ShelfLayerReading(None, None, None, False, "未配置货架板面高度表")
    if resolver is not None:
        reading = resolver.resolve(
            payload.get("shelf_layer"), payload.get("shelf_surface_z")
        )
        if reading.note:
            warnings.append(f"{field}: {reading.note}")
    return (
        VisionObject(
            object_id=object_id,
            label=_text(payload.get("label")),
            semantic_role=SemanticRole.parse(payload.get("semantic_role")),
            pose_world=tuple(
                _number(item, f"{field}.pose_world[{axis}]")
                for axis, item in enumerate(pose)
            ),
            detection_confidence=dino,
            segmentation_confidence=sam,
            on_shelf=bool(payload.get("on_shelf", False)),
            shelf_layer=reading,
            confidence_clamped=dino_clamped or sam_clamped,
        ),
        tuple(warnings),
    )


def _parse_task(payload: Mapping[str, Any], index: int) -> VisionTask:
    field = f"task_queue[{index}]"
    task_id = _optional_int(payload.get("task_id"), f"{field}.task_id")
    if task_id is None:
        raise VisionMessageError(f"{field}.task_id is required")
    target = payload.get("target")
    target = target if isinstance(target, Mapping) else {}
    place = payload.get("place_goal")
    place = place if isinstance(place, Mapping) else {}
    return VisionTask(
        task_id=task_id,
        status=_text(payload.get("status")) or "unknown",
        target_label=_text(target.get("label")),
        target_object_id=_optional_text(target.get("object_id")),
        place_type=_optional_text(place.get("place_type")),
        reference_label=_optional_text(place.get("reference_label")),
        reference_object_id=_optional_text(place.get("reference_object_id")),
        spatial_relation=_optional_text(place.get("spatial_relation")),
        # 缺字段时默认"要重看"而不是"不用重看"：默认值应当倒向更保守的一侧。
        requires_reobserve=bool(target.get("requires_reobserve", True)),
        requires_scene_memory=bool(place.get("requires_scene_memory", False)),
        uncertainties=_string_tuple(payload.get("uncertainties")),
    )


def parse_scene_understanding(
    raw: str | bytes | Mapping[str, Any],
    resolver: ShelfLayerResolver | None = None,
) -> SceneUnderstanding:
    """解析一帧 ``/vlm/scene_understanding`` 报文。

    ``resolver`` 为 ``None`` 时不做货架层号核对（``shelf_layer`` 原样保留但
    ``consistent=False``）。传入 :meth:`ShelfLayerResolver.from_scene_geometry`
    构造的解析器，才会把层号对回官方板面。
    """

    if isinstance(raw, (str, bytes)):
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VisionMessageError("scene_understanding is not valid JSON") from exc
    else:
        payload = raw
    payload = _mapping(payload, "scene_understanding")

    warnings: list[str] = []
    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
        raise VisionMessageError("objects must be a list")
    objects: list[VisionObject] = []
    for index, item in enumerate(raw_objects):
        parsed, item_warnings = _parse_object(_mapping(item, f"objects[{index}]"), index, resolver)
        objects.append(parsed)
        warnings.extend(item_warnings)
    ids = [item.object_id for item in objects]
    if len(ids) != len(set(ids)):
        raise VisionMessageError("objects must have unique object_id values")

    raw_tasks = payload.get("task_queue", [])
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise VisionMessageError("task_queue must be a list")
    tasks = tuple(
        _parse_task(_mapping(item, f"task_queue[{index}]"), index)
        for index, item in enumerate(raw_tasks)
    )
    task_ids = [item.task_id for item in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise VisionMessageError("task_queue must have unique task_id values")

    grounding = payload.get("grounding")
    grounding = grounding if isinstance(grounding, Mapping) else {}
    confidence, clamped = parse_confidence(
        grounding.get("confidence"), "grounding.confidence"
    )
    if clamped:
        warnings.append("grounding.confidence 越界，已截断到 [0,1]")
    if grounding.get("confidence") is not None and confidence is None:
        # 这就是 -999.0。留一条 warning 而不是静默处理：上游文档第八节已经承诺要
        # 改成 0.0/null，warning 是我们这边"改完了没有"的观察点。
        warnings.append(
            f"grounding.confidence={grounding.get('confidence')} 是哨兵值，已按未知处理"
        )

    return SceneUnderstanding(
        schema_version=_text(payload.get("schema_version")),
        observed_at=stamp_seconds(payload),
        active_task_id=_optional_int(payload.get("active_task_id"), "active_task_id"),
        scene_summary=_text(payload.get("scene_summary")),
        objects=tuple(objects),
        tasks=tasks,
        selected_object_id=_optional_text(grounding.get("selected_object_id")),
        selected_label=_optional_text(grounding.get("selected_label")),
        grounding_confidence=confidence,
        requires_reobserve=bool(grounding.get("requires_reobserve", False)),
        grounding_reason=_text(grounding.get("reason")),
        warnings=tuple(warnings),
    )


def to_object_states(
    scene: SceneUnderstanding,
    box_size: BoxSize,
    cameras: Iterable[CameraId] = (CameraId.HEAD_RGBD,),
    include_context: bool = False,
) -> tuple[ObjectState, ...]:
    """把视觉对象转成运动层契约里的 :class:`ObjectState`。

    ``box_size`` 必须由调用方给出，本函数不设默认值。视觉报文里没有尺寸字段，
    而"标准箱 0.24×0.16×0.19"是场景配置里的量（``config/scene_geometry.json``
    的 ``fixed_geometry.standard_box_size``），不是这个模块能知道的常量——写死在
    这里，等于把一个可能随赛题变化的量藏进代码。

    ``include_context=False`` 时只转可抓物。默认排除货架/参考物，正是文档第八节
    列的"仍需优化"第三条：``shelf`` 不应进入动作候选列表。

    置信度取 ``dino_score``（检测置信度），未知时取 0.0。取 0.0 而不是 1.0：契约
    要求一个 [0,1] 的数，而"不知道"最保守的表示就是最低分——下游的
    ``min_confidence`` 闸门会因此挡住它，这正是我们想要的行为。
    """

    selected = scene.objects if include_context else scene.graspable_objects()
    camera_ids = tuple(cameras)
    states: list[ObjectState] = []
    for item in selected:
        x, y, z = item.pose_world
        states.append(
            ObjectState(
                object_id=item.object_id,
                # 颜色取不到时退回 label，保证 ObjectState 的非空校验能过；真正
                # 该报的问题是"视觉给的 label 没有颜色词"，由调用方看 warnings。
                color=item.color or item.label or "unknown",
                pose=Pose3D(x, y, z, 0.0, 0.0, 0.0),
                size=box_size,
                observed_at=scene.observed_at,
                confidence=item.detection_confidence or 0.0,
                source_cameras=camera_ids,
            )
        )
    return tuple(states)


def reobserve_request(
    scene: SceneUnderstanding,
    max_age: float,
    min_confidence: float,
    cameras: Iterable[CameraId] = (CameraId.HEAD_RGBD,),
) -> ObservationRequest | None:
    """当前帧不足以抓取时，生成一次明确的重观测请求；否则返回 ``None``。

    返回的是"请求"而不是运动命令：满足它的是运动层的 ``MotionAction.LOOK``
    （只转头部两个关节），头部行程不够时由 LOOK 自己报出还差多少偏航需要底盘补。
    视觉层不发运动命令，运动层不猜视觉意图，这个函数就是两者之间的那张单子。
    """

    if not scene.blocking_reasons():
        return None
    task = scene.active_task
    # 目标还没绑定时没有 object_id 可填，用任务里的标签占位——ObservationRequest
    # 要求 target_id 非空，而"我要找一个 pink box"本身就是一个合法的观测目标。
    target_id = (
        (task.target_object_id or task.target_label) if task is not None else ""
    ) or "unknown_target"
    return ObservationRequest(
        purpose=ObservationPurpose.ACQUIRE_TARGET,
        target_id=target_id,
        cameras=tuple(cameras),
        max_age=max_age,
        min_confidence=min_confidence,
        require_target_pose=True,
    )