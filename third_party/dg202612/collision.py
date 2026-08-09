"""基于官方 MJCF 的轨迹级碰撞（间隙）检查。

为什么需要这一层
----------------
官方 ``MMK2Kdl.inverse_kinematics`` 只回答「这个末端位姿有没有逆解」。它验证的是
**一个端点**：关节限位加上前向运动学残差。它完全不回答两个致命问题——

1. 这条解本身会不会让手臂穿进桌面、货架或箱体；
2. 从当前姿态**走到**这条解的过程中会不会扫到东西。

所以 :class:`~dg202612.kinematics.JointPair` 的 ``collision_free`` 在本模块接入前
恒为 ``None``，执行器据此拒绝一切机械臂动作。本模块的唯一职责就是把这个 ``None``
变成有依据的 ``True`` / ``False``。

怎么做
------
直接复用**官方竞赛 MJCF 本身**作为碰撞几何来源，而不是另写一套简化包围盒。这样
桌子、货架、围墙、箱体、机器人自身连杆和夹爪指垫的形状与官方仿真逐字一致，不存在
「我的模型和裁判的模型不一样」这种最难查的偏差。

三个关键实现结论（都是在官方镜像里实测出来的，不是照搬文档，改动前请重跑
``tests/test_collision.py``）：

**一、几何体对必须自己枚举，并且要能被人审。**
``mj_collision`` 只报「已经进入 margin」的接触，其 margin 语义偏保守且随几何类型
波动（实测 margin=0.02 时仍会报出真实距离 0.030 的一对）。安全闸不能建立在模糊语义
上，因此本模块显式枚举候选对，过滤规则逐条对齐 MuJoCo 自身的 ``filterBodyPair``。
:func:`candidate_geom_pairs` 是纯函数，可脱离 MuJoCo 单测；同时有一条实机用例验证
它是 MuJoCo 自报接触对的**超集**（只多不漏）。

**二、**``mj_geomDistance`` **的 distmax 要当阈值用，不能当「取个大数拿距离」用。**
实测：底盘立柱与头部两个 box 真实相距 0.0837 m，但

    distmax=0.08 -> 0.080000   (返回阈值，说明真实距离在阈值之外)
    distmax=0.10 -> 0.000000   (返回 0，**不是** 0.0837)

box-box 走的是只算侵入深度、不算分离距离的旧碰撞路径，一旦 distmax 越过真实距离就
退化成 0。若按「传个大 distmax 读距离」来写，8 cm 的安全间隙会被误报成贴死。

正确用法是把 distmax 当作**所需安全间隙**：

    返回值 >= 阈值  <=>  真实距离 >= 阈值   -> 安全
    返回值 <  阈值  <=>  真实距离 <  阈值   -> 违规

该单调性已在 box-box、plane-box、mesh-mesh、cyl-box、mesh-box 五种组合上逐一验证。
判定方向永远偏保守：只会把安全的报成危险，不会把危险的报成安全。需要精确数值时用
:meth:`TrajectoryClearanceChecker.measure_clearance` 二分（实测二分值与分离轴法手算
结果吻合到 1e-4）。

**三、离散采样必须自证足够密。**
只在路径点上检查，两点之间照样可能扫过桌角。本模块不靠「感觉够密了」，而是实测
相邻采样之间**任一几何体的最大位移**，若超过所需间隙就自动加密，直到位移小于间隙
或达到加密上限（上限用尽仍不够密则判定为不安全）。这个位移量会写进报告供人复核。

合法接触
--------
有些接触是正常的，必须白名单化，否则机器人一站在地上就「碰撞」了：驱动轮和万向轮
压地面、夹爪两指闭合时互相接触。抓取阶段还要额外允许指垫接触目标箱体——这类按动作
临时追加，不写进默认白名单，避免「为了让某一步过闸而永久放宽安全条件」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .contracts import JointVector, Pose2D, joint_vector
from .official_assets import (
    SCENE_MARKER,
    OfficialAssetNotFound,
    resolve_scene_root,
)


# ---------------------------------------------------------------------------
# 纯数据：与 MuJoCo 解耦，便于在没有 mujoco 的机器上做单测
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTopology:
    """碰撞对过滤所需的全部模型拓扑，不含任何 MuJoCo 对象。

    字段命名刻意与 MuJoCo 的 ``mjModel`` 保持一致，方便对照官方源码复核。
    """

    geom_names: tuple[str, ...]
    geom_bodies: tuple[int, ...]
    geom_contype: tuple[int, ...]
    geom_conaffinity: tuple[int, ...]
    body_names: tuple[str, ...]
    body_parentid: tuple[int, ...]
    body_weldid: tuple[int, ...]
    # MJCF ``<contact><exclude>`` 声明的 body 对，顺序无关。
    exclusions: frozenset[tuple[int, int]]
    # 机器人根 body（MMK2 为 ``mmk2``）；其所有后代都算机器人。
    robot_root_body: int

    def __post_init__(self) -> None:
        count = len(self.geom_names)
        for name in ("geom_bodies", "geom_contype", "geom_conaffinity"):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} must have one entry per geom")
        if len(self.body_parentid) != len(self.body_names):
            raise ValueError("body_parentid must have one entry per body")
        if len(self.body_weldid) != len(self.body_names):
            raise ValueError("body_weldid must have one entry per body")
        if not 0 <= self.robot_root_body < len(self.body_names):
            raise ValueError("robot_root_body is out of range")

    def is_robot_body(self, body: int) -> bool:
        """沿 body 树上溯，判断该 body 是否属于机器人。"""

        current = body
        # body 0 是 world；上溯到 world 仍没遇到机器人根说明是场景物体。
        while current != 0:
            if current == self.robot_root_body:
                return True
            current = self.body_parentid[current]
        return False

    def weld_parent(self, body: int) -> int:
        """该 body 所在焊接组的父焊接组，对应 MuJoCo 的 ``weldparent``。"""

        weld = self.body_weldid[body]
        return self.body_weldid[self.body_parentid[weld]]

    def describe(self, geom: int) -> str:
        """人类可读的几何体标识：``body/geom``，无名几何体退化为编号。"""

        name = self.geom_names[geom] or f"geom#{geom}"
        return f"{self.body_names[self.geom_bodies[geom]]}/{name}"


@dataclass(frozen=True)
class GeomPair:
    """一组需要逐构型检查间隙的几何体。"""

    first: int
    second: int
    first_label: str
    second_label: str

    def __post_init__(self) -> None:
        if self.first >= self.second:
            raise ValueError("GeomPair must be stored with first < second")


@dataclass(frozen=True)
class AllowedContact:
    """一条「这个接触是正常的」白名单规则。

    ``reason`` 是强制字段：白名单等于主动放弃一部分安全检查，必须写清依据，
    不允许出现无人能解释的豁免项。匹配与左右顺序无关；``*_geom`` 留空表示该
    body 上的任意几何体。
    """

    first_body: str
    second_body: str
    reason: str
    first_geom: str | None = None
    second_geom: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("AllowedContact.reason must explain why it is safe")

    def _side_matches(
        self,
        body_name: str,
        geom_name: str,
        want_body: str,
        want_geom: str | None,
    ) -> bool:
        if body_name != want_body:
            return False
        return want_geom is None or geom_name == want_geom

    def matches(
        self,
        first_body: str,
        first_geom: str,
        second_body: str,
        second_geom: str,
    ) -> bool:
        forward = self._side_matches(
            first_body, first_geom, self.first_body, self.first_geom
        ) and self._side_matches(
            second_body, second_geom, self.second_body, self.second_geom
        )
        backward = self._side_matches(
            second_body, second_geom, self.first_body, self.first_geom
        ) and self._side_matches(
            first_body, first_geom, self.second_body, self.second_geom
        )
        return forward or backward


# MMK2 站在官方场地上时恒定存在、且属于正常受力的接触。
# 数值依据：DOCK 实测停靠位姿下的基线间隙——驱动轮 -0.0018 m、万向轮 -0.0015 m
# （软接触下压量），夹爪两指闭合 -0.000021 m。除这些之外 home 姿态下再无 <0.02 m 的对。
DEFAULT_ALLOWED_CONTACTS: tuple[AllowedContact, ...] = (
    AllowedContact(
        "world", "lft_wheel_link", "左驱动轮压在地面上是正常支撑", first_geom="floor"
    ),
    AllowedContact(
        "world", "rgt_wheel_link", "右驱动轮压在地面上是正常支撑", first_geom="floor"
    ),
    AllowedContact(
        "world",
        "agv_link",
        "右前万向轮压在地面上是正常支撑",
        first_geom="floor",
        second_geom="rgt_front_wheel",
    ),
    AllowedContact(
        "world",
        "agv_link",
        "左前万向轮压在地面上是正常支撑",
        first_geom="floor",
        second_geom="lft_front_wheel",
    ),
    AllowedContact(
        "world",
        "agv_link",
        "右后万向轮压在地面上是正常支撑",
        first_geom="floor",
        second_geom="rgt_behind_wheel",
    ),
    AllowedContact(
        "world",
        "agv_link",
        "左后万向轮压在地面上是正常支撑",
        first_geom="floor",
        second_geom="lft_behind_wheel",
    ),
    AllowedContact(
        "lft_finger_right_link",
        "lft_finger_left_link",
        "左夹爪两指闭合时本就贴合",
    ),
    AllowedContact(
        "rgt_finger_right_link",
        "rgt_finger_left_link",
        "右夹爪两指闭合时本就贴合",
    ),
)


def candidate_geom_pairs(topology: ModelTopology) -> tuple[GeomPair, ...]:
    """枚举所有需要检查的几何体对。

    过滤规则逐条对应 MuJoCo ``engine_collision_driver.c`` 中的
    ``filterBodyPair`` 与 broadphase 前置条件，外加一条本项目自己的收窄：

    1. **至少一侧属于机器人。** 箱子压在桌面上、货架托着箱子都是场景自身的接触，
       与运动安全无关，检查它们只会制造噪声。
    2. **contype / conaffinity 必须相交。** 官方模型里纯视觉网格两者皆为 0，
       本来就不参与碰撞。
    3. **同一焊接组内不检查。** 刚性连在一起，永远「接触」。
    4. **父子焊接组之间不检查**——但 ``world`` 除外。这是最容易写错的一条：
       MuJoCo 的原式是 ``weld1 != 0 && weld2 != 0 && (...)``，少了这个保护会把
       「底盘 vs 地面」整组误排除掉，而那恰恰是必须检查的。
    5. **尊重 MJCF 显式 ``<exclude>``。**
    """

    pairs: list[GeomPair] = []
    for first in range(len(topology.geom_names)):
        for second in range(first + 1, len(topology.geom_names)):
            if not _pair_is_relevant(topology, first, second):
                continue
            pairs.append(
                GeomPair(
                    first,
                    second,
                    topology.describe(first),
                    topology.describe(second),
                )
            )
    return tuple(pairs)


def _pair_is_relevant(topology: ModelTopology, first: int, second: int) -> bool:
    first_body = topology.geom_bodies[first]
    second_body = topology.geom_bodies[second]

    # 规则 1：场景与场景之间的接触不属于运动安全范畴。
    if not (
        topology.is_robot_body(first_body) or topology.is_robot_body(second_body)
    ):
        return False

    # 规则 2：任一方向的碰撞掩码相交即可参与碰撞。
    crosses = (
        topology.geom_contype[first] & topology.geom_conaffinity[second]
    ) or (topology.geom_contype[second] & topology.geom_conaffinity[first])
    if not crosses:
        return False

    first_weld = topology.body_weldid[first_body]
    second_weld = topology.body_weldid[second_body]

    # 规则 3：同一刚体。
    if first_weld == second_weld:
        return False

    # 规则 4：父子焊接组，world（0）不参与该豁免。
    if first_weld != 0 and second_weld != 0:
        if (
            topology.weld_parent(first_body) == second_weld
            or topology.weld_parent(second_body) == first_weld
        ):
            return False

    # 规则 5：MJCF 显式排除。
    if (first_body, second_body) in topology.exclusions:
        return False
    if (second_body, first_body) in topology.exclusions:
        return False
    return True


# ---------------------------------------------------------------------------
# 构型与检查结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotConfiguration:
    """一个待检查的整机构型。

    这里刻意要求写全**所有**会影响碰撞的自由度，包括夹爪开度和头部角度。
    默认值只在调用方明确说明「这一维本次不动」时才应当省略。
    """

    base: Pose2D
    slide: float
    left_arm: JointVector
    right_arm: JointVector
    left_gripper: float = 0.0
    right_gripper: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_arm", joint_vector(self.left_arm, "left_arm"))
        object.__setattr__(
            self, "right_arm", joint_vector(self.right_arm, "right_arm")
        )
        for name in (
            "slide",
            "left_gripper",
            "right_gripper",
            "head_yaw",
            "head_pitch",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


def interpolate_configuration(
    start: RobotConfiguration,
    goal: RobotConfiguration,
    ratio: float,
) -> RobotConfiguration:
    """按比例线性插值两个构型；底盘位姿在机械臂动作期间通常保持不变。"""

    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be within [0, 1]")

    def blend(before: float, after: float) -> float:
        return before + (after - before) * ratio

    return RobotConfiguration(
        base=Pose2D(
            blend(start.base.x, goal.base.x),
            blend(start.base.y, goal.base.y),
            # 偏航按最短弧插值，避免跨 ±pi 时绕远路产生虚假的中间姿态。
            start.base.yaw
            + _wrap_to_pi(goal.base.yaw - start.base.yaw) * ratio,
        ),
        slide=blend(start.slide, goal.slide),
        left_arm=tuple(
            blend(before, after)
            for before, after in zip(start.left_arm, goal.left_arm)
        ),
        right_arm=tuple(
            blend(before, after)
            for before, after in zip(start.right_arm, goal.right_arm)
        ),
        left_gripper=blend(start.left_gripper, goal.left_gripper),
        right_gripper=blend(start.right_gripper, goal.right_gripper),
        head_yaw=blend(start.head_yaw, goal.head_yaw),
        head_pitch=blend(start.head_pitch, goal.head_pitch),
    )


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ClearanceViolation:
    """某个采样点上某一对几何体的间隙不足。"""

    sample_index: int
    first: str
    second: str
    # 注意是**下界**：低于阈值时 mj_geomDistance 的返回值可能比真实距离更小
    # （box-box 会直接给 0）。判定用它是安全的，但不要当作精确测量值上报。
    clearance_lower_bound_m: float
    required_m: float


@dataclass(frozen=True)
class ClearanceReport:
    """一次检查的完整结论，字段都以「能被人复核」为准绳。"""

    collision_free: bool
    required_clearance_m: float
    sample_count: int
    checked_pair_count: int
    # 相邻采样点之间任一几何体的最大位移。它必须小于 required_clearance_m，
    # 否则两点之间可能扫过障碍而检查不到。
    max_sample_step_m: float
    # 加密到上限仍不够密时为 False，此时 collision_free 强制为 False。
    discretization_resolved: bool
    violations: tuple[ClearanceViolation, ...] = ()
    allowed_contacts_used: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        """给计划/日志用的一句话结论。"""

        if not self.discretization_resolved:
            return (
                "轨迹采样加密到上限仍不足以覆盖相邻采样间的扫掠"
                f"（最大位移 {self.max_sample_step_m:.4f} m ≥ "
                f"间隙要求 {self.required_clearance_m:.4f} m）"
            )
        if self.collision_free:
            return (
                f"{self.sample_count} 个采样点、{self.checked_pair_count} 组几何体对"
                f"全部满足 ≥{self.required_clearance_m:.3f} m 间隙"
            )
        worst = min(self.violations, key=lambda item: item.clearance_lower_bound_m)
        return (
            f"间隙不足：{worst.first} 与 {worst.second} 在第 {worst.sample_index} "
            f"个采样点相距不足 {worst.required_m:.3f} m"
            f"（另有 {len(self.violations) - 1} 处）"
            if len(self.violations) > 1
            else (
                f"间隙不足：{worst.first} 与 {worst.second} 在第 "
                f"{worst.sample_index} 个采样点相距不足 {worst.required_m:.3f} m"
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "collision_free": self.collision_free,
            "required_clearance_m": self.required_clearance_m,
            "sample_count": self.sample_count,
            "checked_pair_count": self.checked_pair_count,
            "max_sample_step_m": self.max_sample_step_m,
            "discretization_resolved": self.discretization_resolved,
            "reason": self.reason,
            "violations": [
                {
                    "sample_index": item.sample_index,
                    "first": item.first,
                    "second": item.second,
                    "clearance_lower_bound_m": item.clearance_lower_bound_m,
                    "required_m": item.required_m,
                }
                for item in self.violations
            ],
            "allowed_contacts_used": list(self.allowed_contacts_used),
        }


# ---------------------------------------------------------------------------
# 后端接口与 MuJoCo 实现
# ---------------------------------------------------------------------------


class ClearanceBackend(Protocol):
    """检查器对仿真后端的最小需求，刻意保持很窄以便单测替身。"""

    def topology(self) -> ModelTopology:
        ...

    def apply(self, configuration: RobotConfiguration) -> None:
        ...

    def clearance(self, first: int, second: int, threshold: float) -> float:
        ...

    def geom_positions(self) -> tuple[tuple[float, float, float], ...]:
        ...


class CollisionBackendUnavailable(RuntimeError):
    """无法建立碰撞后端（缺 mujoco、缺官方 MJCF 等）。

    调用方**不得**把它当成「没有碰撞」，而应让 ``collision_free`` 保持 ``None``，
    由执行器继续拒绝。
    """


def _finger_qpos(opening: float, fully_open: float) -> float:
    """把 0~1 的夹爪开度换算成某根手指的关节位置（米，含符号）。

    ``fully_open`` 是这根手指全开时的关节值，由模型限位读出（见
    ``_read_qpos_addresses.finger``），左右指符号相反。

    超出 0~1 一律夹紧：控制指令本来就被 ``SafetyPolicy`` 限在 [0, 1]，这里再夹
    一次是为了让「用越界数值查碰撞」不可能悄悄把手指送到限位之外——那会让检查
    结果偏向「没碰撞」，是危险方向。
    """

    return max(0.0, min(1.0, float(opening))) * fully_open


class MujocoClearanceBackend:
    """用官方竞赛 MJCF 建立的只读碰撞后端。

    只做运动学与碰撞查询，不推进物理，因此不会、也不能影响正在运行的官方仿真。
    """

    # 官方服务器构造运行时 MJCF 的方式（material_sorting_server.py）：读取源 XML
    # 后把占位符替换成任务目录绝对路径。这里逐字复刻，保证几何完全一致。
    # 相对路径直接复用 official_assets 里的标志文件常量，避免两处各写一份、
    # 改了一处忘另一处。
    SOURCE_RELATIVE = SCENE_MARKER
    PLACEHOLDER = "__REPO_ROOT__"

    # 机器人各自由度对应的 MJCF 关节名。按名字查 qpos 地址，不写死下标。
    JOINT_NAMES = {
        "slide": "slide_joint",
        "head_yaw": "head_yaw_joint",
        "head_pitch": "head_pitch_joint",
        "left_arm": tuple(f"lft_arm_joint{index}" for index in range(1, 7)),
        "right_arm": tuple(f"rgt_arm_joint{index}" for index in range(1, 7)),
        "left_gripper": ("lft_finger_right_joint", "lft_finger_left_joint"),
        "right_gripper": ("rgt_finger_right_joint", "rgt_finger_left_joint"),
    }
    ROBOT_ROOT_BODY = "mmk2"

    def __init__(
        self,
        *,
        example_path: Path | str | None = None,
        runtime_xml: Path | str | None = None,
    ) -> None:
        try:
            import mujoco
        except ModuleNotFoundError as error:  # pragma: no cover - 取决于运行环境
            raise CollisionBackendUnavailable(
                "缺少 mujoco；碰撞检查不可用，collision_free 必须保持 None"
            ) from error

        self._mujoco = mujoco
        # 官方 Client 基座不自带任何场景资产（实测 find / 搜不到 MJCF 与 mesh），
        # 正常命中的是仓库 vendor/official_scene；候选顺序见 official_assets。
        # 资产缺失在这里转成 CollisionBackendUnavailable 而不是直接抛出：本模块的
        # 失败语义一贯是「碰撞无法判定」——collision_free 保持 None、执行器继续
        # 拒绝动作，而不是让 Client 崩溃。
        try:
            example = resolve_scene_root(example_path)
        except OfficialAssetNotFound as error:
            raise CollisionBackendUnavailable(str(error)) from error
        # resolve_scene_root 的存在性标志就是 SOURCE_RELATIVE，返回即代表文件在。
        source = example / self.SOURCE_RELATIVE

        text = source.read_text().replace(self.PLACEHOLDER, str(example))
        if runtime_xml is None:
            # from_xml_string 需要一个资产根目录来解析 mesh 相对路径。
            self._model = mujoco.MjModel.from_xml_string(text, {})
        else:
            path = Path(runtime_xml)
            path.write_text(text)
            self._model = mujoco.MjModel.from_xml_path(str(path))
        self._data = mujoco.MjData(self._model)
        # 检查器自己算间隙，不依赖 margin 生成接触，显式清零避免语义混淆。
        self._model.geom_margin[:] = 0.0
        self._topology = self._read_topology()
        self._qpos_address = self._read_qpos_addresses()

    # -- 建模信息读取 ---------------------------------------------------

    def _name(self, kind: Any, index: int) -> str:
        return self._mujoco.mj_id2name(self._model, kind, index) or ""

    def _read_topology(self) -> ModelTopology:
        mujoco = self._mujoco
        model = self._model
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.ROBOT_ROOT_BODY)
        if root < 0:
            raise CollisionBackendUnavailable(
                f"MJCF 中找不到机器人根 body：{self.ROBOT_ROOT_BODY}"
            )
        exclusions = set()
        for index in range(model.nexclude):
            signature = int(model.exclude_signature[index])
            exclusions.add((signature >> 16, signature & 0xFFFF))
        return ModelTopology(
            geom_names=tuple(
                self._name(mujoco.mjtObj.mjOBJ_GEOM, index)
                for index in range(model.ngeom)
            ),
            geom_bodies=tuple(int(value) for value in model.geom_bodyid),
            geom_contype=tuple(int(value) for value in model.geom_contype),
            geom_conaffinity=tuple(int(value) for value in model.geom_conaffinity),
            body_names=tuple(
                self._name(mujoco.mjtObj.mjOBJ_BODY, index)
                for index in range(model.nbody)
            ),
            body_parentid=tuple(int(value) for value in model.body_parentid),
            body_weldid=tuple(int(value) for value in model.body_weldid),
            exclusions=frozenset(exclusions),
            robot_root_body=int(root),
        )

    def _read_qpos_addresses(self) -> dict[str, Any]:
        mujoco = self._mujoco
        model = self._model

        def joint_index(joint_name: str) -> int:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if index < 0:
                raise CollisionBackendUnavailable(
                    f"MJCF 中找不到关节：{joint_name}"
                )
            return index

        def address(joint_name: str) -> int:
            return int(model.jnt_qposadr[joint_index(joint_name)])

        def finger(joint_name: str) -> tuple[int, float]:
            """把手指关节读成「地址 + 全开位置」。

            官方模型里夹爪不是一个 0~1 的关节：左右两指各是一个 **滑动关节**，
            单位是米，行程互为镜像（右指 [0, 0.04]，左指 [-0.04, 0]），再由一个
            equality 约束绑成同步开合。而 19 维控制指令里夹爪是归一化的 0~1。

            所以必须在这里做单位换算。早期版本把 0~1 直接写进 qpos，等于命令
            「张开 1 米」——mj_kinematics 不夹紧关节限位，手指会飞到一米开外，
            于是张开夹爪的构型全都「查不到碰撞」。那是**偏危险方向**的错误：
            检查通过，实物却会撞。行程上限从模型读，不写死数值。
            """

            index = joint_index(joint_name)
            if not bool(model.jnt_limited[index]):
                raise CollisionBackendUnavailable(
                    f"手指关节没有限位，无法把 0~1 开度换算成米：{joint_name}"
                )
            lower = float(model.jnt_range[index][0])
            upper = float(model.jnt_range[index][1])
            # 两端必有一端是 0（= 完全闭合），另一端就是全开位置，含符号。
            if min(abs(lower), abs(upper)) > 1e-9:
                raise CollisionBackendUnavailable(
                    f"手指关节行程不含闭合位 0：{joint_name} range=[{lower}, {upper}]"
                )
            return (
                int(model.jnt_qposadr[index]),
                upper if abs(upper) >= abs(lower) else lower,
            )

        root_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.ROBOT_ROOT_BODY
        )
        root_joint = int(model.body_jntadr[root_body])
        if root_joint < 0 or int(model.jnt_type[root_joint]) != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            raise CollisionBackendUnavailable("机器人根 body 上没有自由关节")
        return {
            "base": int(model.jnt_qposadr[root_joint]),
            "slide": address(self.JOINT_NAMES["slide"]),
            "head_yaw": address(self.JOINT_NAMES["head_yaw"]),
            "head_pitch": address(self.JOINT_NAMES["head_pitch"]),
            "left_arm": tuple(address(name) for name in self.JOINT_NAMES["left_arm"]),
            "right_arm": tuple(address(name) for name in self.JOINT_NAMES["right_arm"]),
            "left_gripper": tuple(
                finger(name) for name in self.JOINT_NAMES["left_gripper"]
            ),
            "right_gripper": tuple(
                finger(name) for name in self.JOINT_NAMES["right_gripper"]
            ),
        }

    # -- ClearanceBackend 实现 -------------------------------------------

    def topology(self) -> ModelTopology:
        return self._topology

    def set_free_body_pose(
        self,
        body_name: str,
        position: Sequence[float],
        quaternion: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        """写入箱体等自由物体的位姿；不调用时沿用 MJCF 里的作者位姿。

        真实比赛中箱体位置由视觉给出，运动层不得凭空假设，因此这里做成显式接口。
        """

        mujoco = self._mujoco
        body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body < 0:
            raise ValueError(f"MJCF 中找不到 body：{body_name}")
        joint = int(self._model.body_jntadr[body])
        if joint < 0 or int(self._model.jnt_type[joint]) != int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(f"body 没有自由关节，无法直接设定位姿：{body_name}")
        address = int(self._model.jnt_qposadr[joint])
        self._data.qpos[address : address + 3] = list(position)
        self._data.qpos[address + 3 : address + 7] = list(quaternion)

    def apply(self, configuration: RobotConfiguration) -> None:
        data = self._data
        address = self._qpos_address

        base = address["base"]
        data.qpos[base + 0] = configuration.base.x
        data.qpos[base + 1] = configuration.base.y
        # z 与官方初始高度保持一致：底盘不会自己浮起来，检查关心的是水平位姿。
        data.qpos[base + 2] = self._model.qpos0[base + 2]
        half_yaw = 0.5 * configuration.base.yaw
        data.qpos[base + 3] = math.cos(half_yaw)
        data.qpos[base + 4] = 0.0
        data.qpos[base + 5] = 0.0
        data.qpos[base + 6] = math.sin(half_yaw)

        data.qpos[address["slide"]] = configuration.slide
        data.qpos[address["head_yaw"]] = configuration.head_yaw
        data.qpos[address["head_pitch"]] = configuration.head_pitch
        for slot, value in zip(address["left_arm"], configuration.left_arm):
            data.qpos[slot] = value
        for slot, value in zip(address["right_arm"], configuration.right_arm):
            data.qpos[slot] = value
        # 官方模型左右两指各有独立关节，行程互为镜像，单位是米；
        # 控制指令里的开度是 0~1，必须换算后再写，换算表见 _read_qpos_addresses。
        for slot, fully_open in address["left_gripper"]:
            data.qpos[slot] = _finger_qpos(configuration.left_gripper, fully_open)
        for slot, fully_open in address["right_gripper"]:
            data.qpos[slot] = _finger_qpos(configuration.right_gripper, fully_open)

        # 只更新位姿，不推进物理：碰撞查询只需要 geom_xpos / geom_xmat。
        self._mujoco.mj_kinematics(self._model, data)
        # 相机与灯光的世界位姿**不在** mj_kinematics 里算，要再走一遍 mj_camlight。
        # 少了这一步，data.cam_xpos / cam_xmat 会一直是全 0——而全 0 不报错，
        # 只是让 camera_frame 返回零向量，于是所有方位角都算成 atan2(0,0)=0，
        # 看起来像"每个目标都正好在画面正中"。这正是 2026-07-31 头部覆盖标定
        # 里所有角度打印成 0.00° 的原因。代价是 O(相机数+灯数)，与遍历全部
        # 刚体的 mj_kinematics 相比可以忽略，因此无条件调用，不做惰性判断。
        self._mujoco.mj_camlight(self._model, data)

    def clearance(self, first: int, second: int, threshold: float) -> float:
        return float(
            self._mujoco.mj_geomDistance(
                self._model, self._data, first, second, threshold, None
            )
        )

    def geom_positions(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            (float(row[0]), float(row[1]), float(row[2]))
            for row in self._data.geom_xpos
        )

    # -- 建模信息读取（供前置观测使用，不参与碰撞判定） ---------------------

    def joint_range(self, name: str) -> tuple[float, float]:
        """读一个关节的行程。

        前置观测要判断"头转到极限还看不看得见"，行程必须来自官方模型现场读取，
        不能在别处写死一个常量——模型一改，写死的常量就成了一个看起来权威的
        过期数字，而且不会报错。

        没有名字或没有限位的关节直接抛错：那种情况下调用方拿到的任何"行程"
        都是编的，静默返回一个默认区间比抛错危险得多。
        """

        mujoco = self._mujoco
        joint = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise KeyError(f"model has no joint named {name!r}")
        if not bool(self._model.jnt_limited[joint]):
            raise ValueError(f"joint {name!r} is not limited; it has no travel range")
        low, high = self._model.jnt_range[joint]
        return float(low), float(high)

    def camera_fovy_deg(self, name: str) -> float:
        """读一台相机的**垂直**视场全角（度）。

        MuJoCo 只存垂直视场；水平视场取决于渲染画幅的宽高比，必须由调用方按
        实际渲染尺寸换算（官方 material_sorting_server.py 用的是 640×480）。
        本方法刻意不替调用方猜画幅——那正是之前把水平半视场误算成 36.56°
        （按 16:9）而真值是 29.10°（按 4:3）的原因。
        """

        mujoco = self._mujoco
        camera = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera < 0:
            raise KeyError(f"model has no camera named {name!r}")
        return float(self._model.cam_fovy[camera])

    def camera_frame(
        self, name: str
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """读一台相机在当前构型下的世界系位姿，返回 ``(光心, 前, 左, 上)``。

        MuJoCo 相机的约定是**沿自身 −z 方向看**，+x 向右、+y 向上。这里把它翻译
        成"前/左/上"，因为 :mod:`dg202612.head_aim` 的角偏离符号是按关节转向定义
        的（左为正 → 增大 yaw），用相机原生的 ±x 会在符号上反复出错。

        调用前必须先 :meth:`apply` 一个构型：``mj_kinematics`` 刷新刚体位姿、
        紧随其后的 ``mj_camlight`` 才刷新相机坐标系，两步都不做则读到的是全 0。

        读出来的旋转矩阵如果是退化的（列向量不是单位长），本方法**直接抛错**而
        不是返回零向量。理由是这个错误的表现极具迷惑性：零向量会让所有方位角
        变成 ``atan2(0, 0) = 0``，打印出来是一张"每个目标都正好居中"的漂亮表格，
        没有任何异常迹象，而结论全错。宁可炸掉。
        """

        mujoco = self._mujoco
        camera = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera < 0:
            raise KeyError(f"model has no camera named {name!r}")
        position = self._data.cam_xpos[camera]
        rotation = self._data.cam_xmat[camera].reshape(3, 3)
        # cam_xmat 的列是相机自身的 x/y/z 轴在世界系里的方向。
        right = tuple(float(rotation[row][0]) for row in range(3))
        up = tuple(float(rotation[row][1]) for row in range(3))
        backward = tuple(float(rotation[row][2]) for row in range(3))
        for axis_name, axis in (("x", right), ("y", up), ("z", backward)):
            length = math.sqrt(sum(value * value for value in axis))
            if abs(length - 1.0) > 1e-6:
                raise RuntimeError(
                    f"camera {name!r} has a degenerate {axis_name} axis "
                    f"(length {length:.6f}); apply() must run before camera_frame()"
                )
        return (
            (float(position[0]), float(position[1]), float(position[2])),
            tuple(-value for value in backward),
            tuple(-value for value in right),
            up,
        )


# ---------------------------------------------------------------------------
# 检查器
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryClearanceChecker:
    """对单个构型或一整条轨迹做间隙检查。

    ``required_clearance_m`` 同时承担两个角色：几何体对的安全间隙阈值，以及
    轨迹采样密度的判据（相邻采样的几何体位移必须小于它，否则可能扫过障碍）。
    """

    backend: ClearanceBackend
    required_clearance_m: float = 0.01
    allowed_contacts: tuple[AllowedContact, ...] = DEFAULT_ALLOWED_CONTACTS
    # 采样加密上限：每轮把段数翻倍，8 轮即 256 段（257 个采样点）。
    # 上限取 8 是实测结果：官方模型下单个构型检查约 0.6 ms、1134 组几何体对，
    # 257 点合计约 0.15 s，对「规划阶段跑一次」完全可接受；而 6 轮（64 段）在
    # 关节大幅运动时步长仍有 0.018 m，压不到 0.01 m 的间隙要求之下。
    max_refinements: int = 8
    _pairs: tuple[GeomPair, ...] = field(init=False)
    _allow_reasons: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.required_clearance_m) or self.required_clearance_m <= 0.0:
            raise ValueError("required_clearance_m must be a positive finite value")
        if self.max_refinements < 0:
            raise ValueError("max_refinements must not be negative")
        topology = self.backend.topology()
        allowed: list[str] = []
        pairs: list[GeomPair] = []
        for pair in candidate_geom_pairs(topology):
            rule = self._allowing_rule(topology, pair)
            if rule is None:
                pairs.append(pair)
            else:
                allowed.append(
                    f"{pair.first_label} <-> {pair.second_label}：{rule.reason}"
                )
        self._pairs = tuple(pairs)
        self._allow_reasons = tuple(sorted(set(allowed)))

    def _allowing_rule(
        self, topology: ModelTopology, pair: GeomPair
    ) -> AllowedContact | None:
        first_body = topology.body_names[topology.geom_bodies[pair.first]]
        second_body = topology.body_names[topology.geom_bodies[pair.second]]
        first_geom = topology.geom_names[pair.first]
        second_geom = topology.geom_names[pair.second]
        for rule in self.allowed_contacts:
            if rule.matches(first_body, first_geom, second_body, second_geom):
                return rule
        return None

    @property
    def checked_pair_count(self) -> int:
        return len(self._pairs)

    @property
    def allowed_contact_reasons(self) -> tuple[str, ...]:
        return self._allow_reasons

    # -- 单构型 -----------------------------------------------------------

    def _violations_at(self, index: int) -> list[ClearanceViolation]:
        threshold = self.required_clearance_m
        found: list[ClearanceViolation] = []
        for pair in self._pairs:
            distance = self.backend.clearance(pair.first, pair.second, threshold)
            # 阈值语义：返回值达到阈值即代表真实距离在阈值之外（安全）。
            # 用相对容差吸收浮点误差，避免恰好等于阈值时反复误报。
            if distance >= threshold - 1e-12:
                continue
            found.append(
                ClearanceViolation(
                    sample_index=index,
                    first=pair.first_label,
                    second=pair.second_label,
                    clearance_lower_bound_m=distance,
                    required_m=threshold,
                )
            )
        return found

    def check_configuration(
        self, configuration: RobotConfiguration
    ) -> ClearanceReport:
        """检查单个静止构型；不回答「怎么走过去」。"""

        self.backend.apply(configuration)
        violations = tuple(self._violations_at(0))
        return ClearanceReport(
            collision_free=not violations,
            required_clearance_m=self.required_clearance_m,
            sample_count=1,
            checked_pair_count=len(self._pairs),
            max_sample_step_m=0.0,
            discretization_resolved=True,
            violations=violations,
            allowed_contacts_used=self._allow_reasons,
        )

    # -- 轨迹 -------------------------------------------------------------

    def _max_geom_step(
        self, samples: Sequence[RobotConfiguration]
    ) -> float:
        """相邻采样之间任一几何体中心的最大位移。

        这是「采样够不够密」的直接证据：位移小于所需间隙时，两点之间不可能
        整体越过一个本来满足间隙的障碍。
        """

        largest = 0.0
        previous: tuple[tuple[float, float, float], ...] | None = None
        for sample in samples:
            self.backend.apply(sample)
            current = self.backend.geom_positions()
            if previous is not None:
                for before, after in zip(previous, current):
                    step = math.dist(before, after)
                    if step > largest:
                        largest = step
            previous = current
        return largest

    def check_path(
        self, waypoints: Sequence[RobotConfiguration]
    ) -> ClearanceReport:
        """检查一整条轨迹，必要时自动加密采样。

        ``waypoints`` 应当包含起点，否则起点姿态本身不会被检查。
        """

        samples = list(waypoints)
        if not samples:
            raise ValueError("waypoints must not be empty")
        if len(samples) == 1:
            return self.check_configuration(samples[0])

        step = self._max_geom_step(samples)
        refinements = 0
        while step >= self.required_clearance_m and refinements < self.max_refinements:
            samples = _densify(samples)
            step = self._max_geom_step(samples)
            refinements += 1
        resolved = step < self.required_clearance_m

        violations: list[ClearanceViolation] = []
        for index, sample in enumerate(samples):
            self.backend.apply(sample)
            violations.extend(self._violations_at(index))

        return ClearanceReport(
            # 采样不够密时不给「安全」结论：宁可挡住，也不能放行没查清的轨迹。
            collision_free=resolved and not violations,
            required_clearance_m=self.required_clearance_m,
            sample_count=len(samples),
            checked_pair_count=len(self._pairs),
            max_sample_step_m=step,
            discretization_resolved=resolved,
            violations=tuple(violations),
            allowed_contacts_used=self._allow_reasons,
        )

    # -- 精确测量（仅供人工复核，不参与判定） -------------------------------

    def measure_clearance(
        self,
        configuration: RobotConfiguration,
        first: int,
        second: int,
        *,
        upper_bound: float = 2.0,
        iterations: int = 40,
    ) -> float:
        """二分出一对几何体的真实间隙。

        为什么不能直接传一个大 distmax：见模块文档「结论二」，box-box 组合在
        distmax 越过真实距离后会返回 0。二分则只依赖那个已验证的单调性质，
        对所有几何类型都成立。
        """

        self.backend.apply(configuration)
        low, high = 0.0, float(upper_bound)
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if middle <= 0.0:
                break
            if self.backend.clearance(first, second, middle) >= middle - 1e-12:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)


def _densify(samples: Sequence[RobotConfiguration]) -> list[RobotConfiguration]:
    """在每两个相邻采样之间插入一个中点。"""

    dense: list[RobotConfiguration] = [samples[0]]
    for before, after in zip(samples, samples[1:]):
        dense.append(interpolate_configuration(before, after, 0.5))
        dense.append(after)
    return dense


def grasp_contact_allowance(
    target_body: str, reason: str
) -> tuple[AllowedContact, ...]:
    """抓取阶段临时允许 link6 主接触面和四片指爪接触目标箱体。

    做成显式函数而不是塞进默认白名单，是为了让「什么时候允许碰箱子」这件事
    出现在调用点上，可被审核；PREGRASP 等不该接触的动作绝不会误用。

    ``arm_link5`` 不在赛事裁判的左右夹持链接集合里，也不是本方案的接触面，
    因此绝不豁免；它靠近箱体应继续被碰撞检查拦下。
    """

    contact_bodies = (
        "lft_arm_link6",
        "lft_finger_right_link",
        "lft_finger_left_link",
        "rgt_arm_link6",
        "rgt_finger_right_link",
        "rgt_finger_left_link",
    )
    return tuple(
        AllowedContact(body, target_body, reason) for body in contact_bodies
    )


def configurations_from_joint_path(
    base: Pose2D,
    frames: Iterable[Any],
    *,
    fallback_slide: float,
    left_gripper: float = 0.0,
    right_gripper: float = 0.0,
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
) -> tuple[RobotConfiguration, ...]:
    """把 ``kinematics.synchronized_joint_path`` 的输出转成待检查构型序列。

    ``frames`` 里的元素只需具备 ``left`` / ``right`` / ``slide`` 三个属性，
    因此 :class:`~dg202612.kinematics.JointPair` 可以直接传入而不产生循环依赖。
    """

    result = []
    for frame in frames:
        slide = getattr(frame, "slide", None)
        result.append(
            RobotConfiguration(
                base=base,
                slide=fallback_slide if slide is None else float(slide),
                left_arm=frame.left,
                right_arm=frame.right,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                head_yaw=head_yaw,
                head_pitch=head_pitch,
            )
        )
    return tuple(result)
