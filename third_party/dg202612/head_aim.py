"""前置观测控制：把头部相机对准目标，或在不知道目标在哪时做有界扫视。

这个模块解决的问题
------------------
上游视觉层交回来的 ``SceneState`` 里有物体位姿，运动层照着去抓。但这里有一个
从来没被写下来的前提：**视觉要先看得见。** 官方示例是靠人工排好的观察位加固定
低头角把这件事绕过去的（``INITIAL_OBSERVE_HEAD``、``look_pitch`` −0.50/−0.45/
−0.25），一旦场景随机化、或者目标在开局视野之外，那套写死的角度就不成立。

所以运动层需要一个"前置动作"：在请求视觉识别之前，先把头转到能看见目标的角度；
如果头部关节转到极限仍然看不见，就明确报出"还差多少，需要底盘补多少偏航"，
而不是默默交一个看不见目标的视角给视觉层，让对方在图里找不到东西。

为什么要闭环，不能一次算完
--------------------------
相机不在头部两个旋转轴的交点上（``head_cam`` 挂在 ``head_pitch_link`` 上，
偏航轴、俯仰轴、光心三者互不重合）。转头会同时改变相机的**位置**和**朝向**，
所以"目标当前偏离光轴 10°，那就把头转 10°"只是一阶近似，转完之后残差不为零。
本模块用不动点迭代把残差压下去：每轮实测偏离角、按它修正指令、再测。实测由调用
方注入（仿真里用 MuJoCo 正运动学，真机上可以用检测框中心），本模块只管迭代、
限位和收敛判据，因此可以在没有 MuJoCo 的机器上跑单元测试。

关节转到极限时的正确行为
------------------------
写这个模块的直接教训：诊断脚本里第一版把"指令角"和"实际角"混在一个累加器里，
指令被限位夹住之后实测偏离角不再变化，累加器却继续积分，最后算出 +725.99° 这种
数。那种数字在真机上会变成一条"看起来很确定"的错误指令。
正确做法是——**被夹住就停止积分，把剩下的偏离角当作结论上报**：那正是"头部单独
办不到，需要底盘转多少"的答案。本模块的 :class:`HeadAimResult` 因此一定会带
``base_yaw_delta_rad``，而不是只回一个 yaw/pitch。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

__all__ = [
    "AngularOffset",
    "HeadAimResult",
    "HeadAimStep",
    "HeadLimits",
    "HeadPose",
    "aim_head",
    "coverage_bounds",
    "plan_scan",
    "visible_from",
]


@dataclass(frozen=True)
class HeadPose:
    """一组头部关节角，单位弧度。"""

    yaw: float
    pitch: float


@dataclass(frozen=True)
class AngularOffset:
    """目标相对相机光轴的角偏离，符号与关节转向一致。

    符号约定（和 MMK2 的关节正方向对齐，避免调用方靠猜）：

    * ``horizontal`` > 0 —— 目标在光轴**左侧**，需要**增大** ``head_yaw``；
    * ``vertical``   > 0 —— 目标在光轴**上方**，需要**增大** ``head_pitch``。

    也就是说修正量就是偏离量本身，一阶意义上 ``yaw += horizontal``。官方低头看桌面
    用的是负 pitch（``look_pitch`` −0.50 等），与"向下为负"一致，可作为交叉验证。

    ``distance`` 是相机光心到目标的直线距离，只用于记录与排序，不参与迭代。
    """

    horizontal: float
    vertical: float
    distance: float = 0.0


@dataclass(frozen=True)
class HeadLimits:
    """头部关节行程与相机视场角，全部是半角（弧度）。

    刻意不在本模块里写死任何官方数值：这些量必须从官方 MJCF 现场读出来再传进来，
    否则模型一改，这里就成了一个看起来权威的过期常量。取值方法见
    ``dev/motion_lab.py`` 的 ``official_head_limits()``。
    """

    yaw_range: tuple[float, float]
    pitch_range: tuple[float, float]
    half_fov_horizontal: float
    half_fov_vertical: float

    def __post_init__(self) -> None:
        for name in ("yaw_range", "pitch_range"):
            low, high = getattr(self, name)
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(f"{name} must be finite")
            if low > high:
                raise ValueError(f"{name} must be ordered (low, high)")
        for name in ("half_fov_horizontal", "half_fov_vertical"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite half-angle")

    def clamp(self, pose: HeadPose) -> tuple[HeadPose, tuple[str, ...]]:
        """把一组指令角夹进行程，并报出哪几个轴被夹住了。

        返回"被夹住的轴名"而不是只返回夹完的角度：调用方需要区分"刚好停在
        限位上"和"想要的角度超出了限位"，前者正常，后者意味着头部办不到。
        """

        yaw = min(max(pose.yaw, self.yaw_range[0]), self.yaw_range[1])
        pitch = min(max(pose.pitch, self.pitch_range[0]), self.pitch_range[1])
        saturated = []
        if not math.isclose(yaw, pose.yaw, rel_tol=0.0, abs_tol=1e-12):
            saturated.append("yaw")
        if not math.isclose(pitch, pose.pitch, rel_tol=0.0, abs_tol=1e-12):
            saturated.append("pitch")
        return HeadPose(yaw, pitch), tuple(saturated)


@dataclass(frozen=True)
class HeadAimStep:
    """一轮迭代的完整留痕，供人工复核"为什么最后停在这个角度"。"""

    index: int
    commanded: HeadPose
    applied: HeadPose
    saturated: tuple[str, ...]
    offset: AngularOffset


@dataclass(frozen=True)
class HeadAimResult:
    """一次对准的结论。

    三个布尔量刻意分开，它们回答的是不同的问题，不能互相替代：

    * ``measured`` —— 测量本身是否可用（注入的测量函数返回 ``None`` 时为假，
      对应真机上目标被遮挡、检测丢失）；
    * ``in_view``  —— 目标是否落在（收缩了安全边距的）视场内，即"看得见"；
    * ``centered`` —— 目标是否已被拉到光轴附近的容差内，即"看得清、可用于估位姿"。

    只要 ``base_yaw_delta_rad`` 不为 0，就说明头部行程不够，需要底盘补这么多偏航；
    它是一阶估计（相机不在底盘旋转轴上），底盘转完必须重测一次，不能当作精确值。
    """

    pose: HeadPose
    measured: bool
    in_view: bool
    centered: bool
    offset: AngularOffset
    iterations: int
    saturated: tuple[str, ...]
    base_yaw_delta_rad: float
    trace: tuple[HeadAimStep, ...]

    @property
    def needs_base_rotation(self) -> bool:
        return self.base_yaw_delta_rad != 0.0


# 视场边缘的安全边距：默认 5°。不是保守偏好，而是两个具体理由——
#   1) 靠图像边框的检测框容易被画幅截断，位姿估计的方差显著变大；
#   2) 本模块的迭代是一阶的，判"刚好在视场边缘"时留不出修正余量。
# 调用方可以按需调小，但请连同理由一起写在调用处。
DEFAULT_FOV_MARGIN_RAD = math.radians(5.0)

# 收敛容差：默认 2°。相机 640×480、水平半视场 29.1°，一个像素约 0.09°，
# 2° 约 22 像素——比检测框中心的抖动量级大，再压下去是在追噪声。
DEFAULT_CENTER_TOLERANCE_RAD = math.radians(2.0)


def aim_head(
    measure: Callable[[HeadPose], AngularOffset | None],
    limits: HeadLimits,
    *,
    initial: HeadPose = HeadPose(0.0, 0.0),
    center_tolerance_rad: float = DEFAULT_CENTER_TOLERANCE_RAD,
    fov_margin_rad: float = DEFAULT_FOV_MARGIN_RAD,
    max_iterations: int = 8,
    gain: float = 1.0,
) -> HeadAimResult:
    """闭环把头部对准目标。

    参数
    ----
    measure
        注入的测量函数：给一组头部关节角，返回目标相对光轴的角偏离，看不到时返回
        ``None``。仿真里由 MuJoCo 正运动学算出，真机上可由检测框中心换算。
        本模块不关心它怎么来的，因此可以在没有 MuJoCo 的机器上做单元测试。
    limits
        关节行程与视场半角，见 :class:`HeadLimits`。
    initial
        起始角。给当前实测头部角，而不是 0：从当前位置起步的迭代步数更少，
        也避免"先回零再转过去"这种没必要的大幅摆头。
    gain
        每轮吃掉偏离角的比例，默认 1.0（一阶完全修正）。若实测发现某台机器上
        1.0 会过冲，可调小；调小只会变慢，不会改变收敛点。
    max_iterations
        迭代上限。到上限仍未收敛时不抛异常，而是把当轮残差如实报出来——
        运动层需要据此决定"重观测还是换站位"，抛异常会把这个决策权拿走。

    ``gain`` 必须在 (0, 1] 内；``max_iterations`` 至少 1。这两个是编程错误而不是
    运行时状况，所以直接抛 ``ValueError``。
    """

    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be within (0, 1]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    if center_tolerance_rad <= 0.0:
        raise ValueError("center_tolerance_rad must be positive")
    if fov_margin_rad < 0.0:
        raise ValueError("fov_margin_rad must not be negative")

    pose, _ = limits.clamp(initial)
    steps: list[HeadAimStep] = []
    offset = AngularOffset(0.0, 0.0, 0.0)
    saturated: tuple[str, ...] = ()
    measured = False
    # 残差必须是"在最终交出的这组角上"测到的，否则报告里的角度和残差对不上号。
    # 这个变量就是用来保证这一点的：循环结束时若姿态已经动过而没再测，补测一次。
    measured_at: HeadPose | None = None

    for index in range(max_iterations):
        sample = measure(pose)
        if sample is None:
            # 测量不可用：如实上报，不猜、不沿用上一轮的偏离角继续转头。
            return HeadAimResult(
                pose=pose,
                measured=False,
                in_view=False,
                centered=False,
                offset=offset,
                iterations=index,
                saturated=saturated,
                base_yaw_delta_rad=0.0,
                trace=tuple(steps),
            )
        measured = True
        measured_at = pose
        offset = sample
        if (
            abs(sample.horizontal) <= center_tolerance_rad
            and abs(sample.vertical) <= center_tolerance_rad
        ):
            return HeadAimResult(
                pose=pose,
                measured=True,
                in_view=_within_fov(sample, limits, fov_margin_rad),
                centered=True,
                offset=sample,
                iterations=index + 1,
                saturated=saturated,
                base_yaw_delta_rad=0.0,
                trace=tuple(steps),
            )

        commanded = HeadPose(
            pose.yaw + gain * sample.horizontal,
            pose.pitch + gain * sample.vertical,
        )
        applied, saturated = limits.clamp(commanded)
        steps.append(
            HeadAimStep(
                index=index,
                commanded=commanded,
                applied=applied,
                saturated=saturated,
                offset=sample,
            )
        )
        if applied == pose:
            # 指令没能让姿态实际改变——两轴都顶死，或顶死的轴之外已经到位。
            # 再迭代只会原地打转（当初算出 +725.99° 的坑正是继续积分导致的）：
            # 停下来，把残差当作结论交出去。
            break
        pose = applied

    if measured_at is not None and measured_at != pose:
        sample = measure(pose)
        if sample is not None:
            offset = sample

    return HeadAimResult(
        pose=pose,
        measured=measured,
        in_view=_within_fov(offset, limits, fov_margin_rad),
        centered=(
            abs(offset.horizontal) <= center_tolerance_rad
            and abs(offset.vertical) <= center_tolerance_rad
        ),
        offset=offset,
        iterations=len(steps),
        saturated=saturated,
        # 只有偏航顶死才需要底盘帮忙；俯仰顶死是底盘转不出来的，得靠升降或换站位，
        # 那是上层的决策，本模块不越权给建议。
        base_yaw_delta_rad=(offset.horizontal if "yaw" in saturated else 0.0),
        trace=tuple(steps),
    )


def _within_fov(
    offset: AngularOffset, limits: HeadLimits, margin: float
) -> bool:
    """目标是否落在收缩了 ``margin`` 的视场内。

    边距把视场缩小到不小于零；给一个大于半视场的边距时判定恒为假，而不是抛错——
    调用方那样传就是想要"必须非常居中"，让它自然为假比抛异常更好用。
    """

    horizontal = max(limits.half_fov_horizontal - margin, 0.0)
    vertical = max(limits.half_fov_vertical - margin, 0.0)
    return abs(offset.horizontal) <= horizontal and abs(offset.vertical) <= vertical


def coverage_bounds(limits: HeadLimits) -> tuple[tuple[float, float], tuple[float, float]]:
    """头部单独（底盘不动）能覆盖的角度范围，返回 ``((yaw_lo, yaw_hi), (pitch_lo, pitch_hi))``。

    覆盖范围是"关节行程 ⊕ 视场半角"：头转到极限时，视场边缘还能再往外看半个视场。
    这是回答"某个东西开局到底看不看得见"最直接的判据——如果它的方位角落在这个
    范围之外，那么**任何头部角度都看不见它**，必须动底盘。
    """

    return (
        (
            limits.yaw_range[0] - limits.half_fov_horizontal,
            limits.yaw_range[1] + limits.half_fov_horizontal,
        ),
        (
            limits.pitch_range[0] - limits.half_fov_vertical,
            limits.pitch_range[1] + limits.half_fov_vertical,
        ),
    )


def plan_scan(
    limits: HeadLimits,
    *,
    overlap: float = 0.2,
    start: HeadPose | None = None,
) -> tuple[HeadPose, ...]:
    """规划一遍有界扫视，覆盖整个头部可视范围。

    ``aim_head`` 解决的是"已经知道目标大致在哪、把它拉到画面中央"；开局什么都不
    知道时需要的是另一件事：**把头部行程扫一遍，保证可视范围内没有漏看的角落。**

    覆盖方式是标准的栅格扫视：相邻视场之间保留 ``overlap`` 比例的重叠，
    因此步长 = 2 × 半视场 × (1 − overlap)。重叠不是浪费——落在两幅画面接缝上的
    物体会被两边各截掉一半，重叠让它至少在一幅画面里是完整的。

    行序按"蛇形"（boustrophedon）来回走，而不是每行都回到起点：头部是实体关节，
    每行归位要多转一整个行程，蛇形把总转动量减到最小。

    ``start`` 给当前头部角时，会选择离它更近的那一端作为扫视起点，同样是为了少转。
    """

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be within [0, 1)")

    yaws = _scan_axis(limits.yaw_range, limits.half_fov_horizontal, overlap)
    pitches = _scan_axis(limits.pitch_range, limits.half_fov_vertical, overlap)
    if start is not None:
        # 起点靠哪端就从哪端开始扫。
        if abs(start.yaw - yaws[-1]) < abs(start.yaw - yaws[0]):
            yaws = tuple(reversed(yaws))
        if abs(start.pitch - pitches[-1]) < abs(start.pitch - pitches[0]):
            pitches = tuple(reversed(pitches))

    poses: list[HeadPose] = []
    for row, pitch in enumerate(pitches):
        columns = yaws if row % 2 == 0 else tuple(reversed(yaws))
        poses.extend(HeadPose(yaw, pitch) for yaw in columns)
    return tuple(poses)


def _scan_axis(
    joint_range: tuple[float, float], half_fov: float, overlap: float
) -> tuple[float, ...]:
    """单轴上的扫视角序列：等间距、间距不超过步长、两端都取到。"""

    low, high = joint_range
    span = high - low
    step = 2.0 * half_fov * (1.0 - overlap)
    if span <= 0.0 or step <= 0.0:
        return (0.5 * (low + high),)
    if span <= step:
        # 一幅画面（连同重叠余量）就盖住了整个行程，扫两端反而多转。
        return (0.5 * (low + high),)
    count = int(math.ceil(span / step)) + 1
    return tuple(low + span * index / (count - 1) for index in range(count))


def visible_from(
    limits: HeadLimits, bearings: Sequence[AngularOffset]
) -> tuple[bool, ...]:
    """一组方位角（相对头部零位光轴）里，哪些是头部单独能看到的。

    用来回答"开局位姿下货架/箱子看不看得见"这类问题：只做范围判断，不迭代、
    不需要注入测量函数，因此可以对着一张离线的方位角表直接跑。
    """

    (yaw_low, yaw_high), (pitch_low, pitch_high) = coverage_bounds(limits)
    return tuple(
        yaw_low <= item.horizontal <= yaw_high
        and pitch_low <= item.vertical <= pitch_high
        for item in bearings
    )
