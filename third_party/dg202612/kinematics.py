"""双臂 IK 的可验证接口与同步轨迹工具。

本文件故意没有伪造 MMK2 的逆解。真正适配官方镜像时，只能由已在官方环境验证过的
求解器实现 ``DualArmSolver``；没有该实现时，候选会被明确拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
from typing import Any, Callable, Protocol, Sequence

from .contracts import JointVector, Pose2D, Pose3D, joint_vector
from .manipulation import DualArmHugPlan
from .official_assets import resolve_kdl_root


def world_pose_to_base_target(
    pose: Pose3D,
    base: Pose2D,
) -> tuple[
    tuple[float, float, float],
    tuple[tuple[float, float, float], ...],
]:
    """把世界坐标中的末端位姿转换成官方 KDL 使用的底盘坐标目标。"""

    cr, sr = math.cos(pose.roll), math.sin(pose.roll)
    cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
    cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
    world_rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    cosine = math.cos(base.yaw)
    sine = math.sin(base.yaw)
    world_to_base = (
        (cosine, sine, 0.0),
        (-sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )
    dx = pose.x - base.x
    dy = pose.y - base.y
    position = (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        pose.z,
    )
    rotation = tuple(
        tuple(
            sum(world_to_base[row][k] * world_rotation[k][column] for k in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    return position, rotation


@dataclass(frozen=True)
class JointPair:
    left: JointVector
    right: JointVector
    # ``residual`` 是双臂前向运动学复核后的最大末端位置误差（米）。
    residual: float
    # None 表示只求出了运动学解，尚未接入碰撞检查；不能据此执行。
    collision_free: bool | None
    slide: float | None = None
    orientation_error: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", joint_vector(self.left, "left"))
        object.__setattr__(self, "right", joint_vector(self.right, "right"))
        if not math.isfinite(self.residual) or self.residual < 0.0:
            raise ValueError("residual must be a non-negative finite value")
        if self.slide is not None and not math.isfinite(self.slide):
            raise ValueError("slide must be finite")
        if (
            not math.isfinite(self.orientation_error)
            or self.orientation_error < 0.0
        ):
            raise ValueError(
                "orientation_error must be a non-negative finite value"
            )


class DualArmSolver(Protocol):
    """官方 KDL/碰撞检查适配器未来要实现的最小接口。"""

    def solve(self, plan: DualArmHugPlan, seed: JointPair | None) -> tuple[JointPair, ...]:
        ...


# 碰撞判定回调：输入一个运动学候选，返回 True/False/None。
# None 表示「无法判定」（碰撞后端缺失等），调用方必须继续拒绝，不得当作安全。
CollisionEvaluator = Callable[["JointPair"], bool | None]


@dataclass(frozen=True)
class KinematicCheck:
    feasible: bool
    solution: JointPair | None
    reason: str
    candidate_count: int = 0
    collision_checked: bool = False


@dataclass(frozen=True)
class HugSequenceCheck:
    """一串笛卡尔抱持采样的联合 IK 结果。"""

    feasible: bool
    solutions: tuple[JointPair, ...]
    reason: str
    failed_index: int | None = None


def _travel(current: JointPair, candidate: JointPair) -> tuple[float, float]:
    deltas = [
        abs(after - before)
        for before, after in zip(
            current.left + current.right,
            candidate.left + candidate.right,
        )
    ]
    if current.slide is not None and candidate.slide is not None:
        deltas.append(abs(candidate.slide - current.slide))
    return max(deltas), sum(deltas)


def choose_joint_pair(
    candidates: tuple[JointPair, ...],
    current: JointPair,
    *,
    require_collision_check: bool = True,
) -> JointPair | None:
    """按碰撞、残差、最大关节变化、总关节变化的可解释字典序筛选。"""

    valid = [
        item
        for item in candidates
        if item.collision_free is True
        or (not require_collision_check and item.collision_free is not False)
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda item: (
            item.residual,
            item.orientation_error,
            *_travel(current, item),
        ),
    )


def check_dual_arm_hug(
    solver: DualArmSolver,
    plan: DualArmHugPlan,
    current: JointPair,
) -> KinematicCheck:
    candidates = solver.solve(plan, current)
    chosen = choose_joint_pair(candidates, current)
    if chosen is None:
        if any(item.collision_free is None for item in candidates):
            return KinematicCheck(
                False,
                None,
                "dual-arm IK exists, but collision checking is not connected",
                len(candidates),
                False,
            )
        return KinematicCheck(
            False,
            None,
            "no collision-free dual-arm IK solution",
            len(candidates),
            True,
        )
    return KinematicCheck(
        True,
        chosen,
        "dual-arm IK solution selected",
        len(candidates),
        True,
    )


def synchronized_joint_path(
    start: JointPair,
    goal: JointPair,
    max_joint_step: float,
    max_slide_step: float | None = None,
) -> tuple[JointPair, ...]:
    """以同一节拍插补左右臂，避免一只臂先到位而另一只仍大幅运动。"""

    if max_joint_step <= 0.0:
        raise ValueError("max_joint_step must be positive")
    if (start.slide is None) != (goal.slide is None):
        raise ValueError(
            "start and goal must either both include slide or both omit it"
        )
    if start.slide is not None and (
        max_slide_step is None or max_slide_step <= 0.0
    ):
        raise ValueError(
            "max_slide_step must be positive when the path includes slide"
        )
    values_start = start.left + start.right
    values_goal = goal.left + goal.right
    largest_joint_delta = max(
        abs(end - begin)
        for begin, end in zip(values_start, values_goal)
    )
    steps = math.ceil(largest_joint_delta / max_joint_step)
    if start.slide is not None and goal.slide is not None:
        assert max_slide_step is not None
        steps = max(
            steps,
            math.ceil(abs(goal.slide - start.slide) / max_slide_step),
        )
    steps = max(1, steps)
    result = []
    for index in range(1, steps + 1):
        ratio = index / steps
        values = tuple(begin + (end - begin) * ratio for begin, end in zip(values_start, values_goal))
        slide = (
            None
            if start.slide is None or goal.slide is None
            else start.slide + (goal.slide - start.slide) * ratio
        )
        result.append(
            JointPair(
                values[:6],
                values[6:],
                goal.residual,
                goal.collision_free,
                slide,
                goal.orientation_error,
            )
        )
    return tuple(result)


def solve_hug_sequence(
    solver: DualArmSolver,
    plans: Sequence[DualArmHugPlan],
    current: JointPair,
    *,
    require_collision_check: bool = False,
) -> HugSequenceCheck:
    """按采样顺序求联合 IK，并把上一解作为下一采样的种子。

    该函数只保证运动学连续性。正式执行前仍须把返回的完整构型序列交给场景碰撞
    检查器；不能拿最终一点安全替代整条路径安全。
    """

    if not plans:
        return HugSequenceCheck(False, (), "hug sequence is empty", 0)
    result: list[JointPair] = []
    seed = current
    for index, plan in enumerate(plans):
        candidates = solver.solve(plan, seed)
        chosen = choose_joint_pair(
            candidates,
            seed,
            require_collision_check=require_collision_check,
        )
        if chosen is None:
            return HugSequenceCheck(
                False,
                tuple(result),
                f"no valid dual-arm IK at Cartesian sample {index}",
                index,
            )
        result.append(chosen)
        seed = chosen
    return HugSequenceCheck(True, tuple(result), "all Cartesian samples solved")


def synchronized_joint_sequence(
    start: JointPair,
    goals: Sequence[JointPair],
    max_joint_step: float,
    max_slide_step: float | None = None,
) -> tuple[JointPair, ...]:
    """连接一串已求出的笛卡尔采样，不跨过任何中间采样抄近道。"""

    result: list[JointPair] = []
    previous = start
    for goal in goals:
        result.extend(
            synchronized_joint_path(
                previous,
                goal,
                max_joint_step,
                max_slide_step,
            )
        )
        previous = goal
    return tuple(result)


class UnverifiedDualArmSolver:
    """干运行默认使用它，明确说明当前还没有经过官方环境验证的 IK 适配。"""

    def solve(self, plan: DualArmHugPlan, seed: JointPair | None) -> tuple[JointPair, ...]:
        del plan, seed
        return ()


class OfficialMMK2DualArmSolver:
    """官方 MMK2Kdl.inverse_kinematics 的薄适配器。

    官方接口只回答「这个末端位姿有没有逆解」，不返回任何碰撞结论。因此本适配器
    默认把候选的 ``collision_free`` 置为 ``None``，执行器据此继续拒绝。

    传入 ``collision_evaluator`` 后，每个通过前向运动学复核的候选会再交给它判定，
    返回值直接写进 ``collision_free``：``True`` 可执行、``False`` 拒绝、``None``
    表示无法判定（例如碰撞后端不可用）——三种情况都不允许被静默当成安全。

    回调而不是直接持有检查器，是因为碰撞判定还需要夹爪开度、头部角度、底盘位姿
    以及「从哪个姿态走过去」这些 JointPair 里没有的信息。由组装方（dev/motion_lab.py）
    在知道整机状态的地方构造闭包，本模块就不必反向依赖 collision 模块。
    """

    def __init__(
        self,
        base: Pose2D,
        target_slide: float,
        *,
        example_path: Path | None = None,
        backend_factory: Callable[[], Any] | None = None,
        max_position_error: float = 1e-4,
        max_orientation_error: float = 1e-3,
        collision_evaluator: CollisionEvaluator | None = None,
    ) -> None:
        if not math.isfinite(target_slide):
            raise ValueError("target_slide must be finite")
        if max_position_error <= 0.0 or max_orientation_error <= 0.0:
            raise ValueError("forward-kinematics tolerances must be positive")
        self.base = base
        self.target_slide = float(target_slide)
        # None 表示「按 official_assets 的候选顺序自动定位」，等到真正要用后端时
        # 才解析：构造求解器本身不该触发文件系统探测。
        self.example_path = None if example_path is None else Path(example_path)
        self.backend_factory = backend_factory
        self.max_position_error = float(max_position_error)
        self.max_orientation_error = float(max_orientation_error)
        self.collision_evaluator = collision_evaluator

    def _backend(self) -> Any:
        if self.backend_factory is not None:
            return self.backend_factory()
        # 官方 Client 基座里没有 mmk2_kdl（实测 find / 无结果），靠仓库 vendor
        # 自带。绝不能把 Server 镜像里的路径写死成唯一来源：那样交付镜像一跑就是
        # ModuleNotFoundError，Client 异常退出，按官方 Q&A Q16/Q38 直接判 0 分。
        # 候选顺序与理由见 dg202612.official_assets 的模块说明。
        root = str(resolve_kdl_root(self.example_path))
        if root not in sys.path:
            sys.path.insert(0, root)
        from mmk2_kdl import MMK2Kdl

        return MMK2Kdl()

    @staticmethod
    def _rotation_from_rpy(pose: Pose3D) -> tuple[tuple[float, float, float], ...]:
        cr, sr = math.cos(pose.roll), math.sin(pose.roll)
        cp, sp = math.cos(pose.pitch), math.sin(pose.pitch)
        cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )

    @staticmethod
    def _multiply(
        left: Sequence[Sequence[float]],
        right: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            tuple(sum(float(left[row][k]) * float(right[k][column]) for k in range(3)) for column in range(3))
            for row in range(3)
        )

    def _world_target(
        self,
        pose: Pose3D,
    ) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
        return world_pose_to_base_target(pose, self.base)

    def solve_base_targets(
        self,
        left_position: Sequence[float],
        left_rotation: Sequence[Sequence[float]],
        right_position: Sequence[float],
        right_rotation: Sequence[Sequence[float]],
        seed: JointPair | None,
    ) -> tuple[JointPair, ...]:
        """供官方 baseline 姿态标定使用；输入明确位于底盘坐标系。"""

        if seed is None:
            return ()
        import numpy as np

        left_transform = np.eye(4)
        right_transform = np.eye(4)
        left_transform[:3, :3] = np.asarray(left_rotation, dtype=float)
        right_transform[:3, :3] = np.asarray(right_rotation, dtype=float)
        left_transform[:3, 3] = np.asarray(left_position, dtype=float)
        right_transform[:3, 3] = np.asarray(right_position, dtype=float)
        reference = np.asarray(
            (self.target_slide, *seed.left, *seed.right),
            dtype=float,
        )
        backend = self._backend()
        raw_solutions = backend.inverse_kinematics(
            T_left=left_transform,
            T_right=right_transform,
            ref_pos=reference,
            target_height=self.target_slide,
        )
        if raw_solutions is None:
            return ()

        candidates: list[JointPair] = []
        for raw_solution in raw_solutions:
            values = np.asarray(raw_solution, dtype=float).reshape(-1)
            if values.size != 13 or not np.isfinite(values).all():
                continue
            actual_left, actual_right = backend.forward_kinematics(values)
            position_error = max(
                float(
                    np.linalg.norm(
                        actual_left[:3, 3] - left_transform[:3, 3]
                    )
                ),
                float(
                    np.linalg.norm(
                        actual_right[:3, 3] - right_transform[:3, 3]
                    )
                ),
            )
            orientation_error = max(
                _rotation_error(
                    actual_left[:3, :3],
                    left_transform[:3, :3],
                ),
                _rotation_error(
                    actual_right[:3, :3],
                    right_transform[:3, :3],
                ),
            )
            if (
                position_error > self.max_position_error
                or orientation_error > self.max_orientation_error
            ):
                continue
            candidates.append(
                JointPair(
                    tuple(float(value) for value in values[1:7]),
                    tuple(float(value) for value in values[7:13]),
                    position_error,
                    None,
                    float(values[0]),
                    orientation_error,
                )
            )
        if self.collision_evaluator is None:
            return tuple(candidates)
        # 逐个候选做碰撞判定。判定失败（抛异常）等同于「无法判定」而不是「安全」，
        # 所以这里不吞掉异常——让它冒到规划层，由规划层记录并保持候选不可执行。
        return tuple(
            replace(candidate, collision_free=self.collision_evaluator(candidate))
            for candidate in candidates
        )

    def forward_base_targets(
        self,
        joints: JointPair,
    ) -> tuple[
        tuple[tuple[float, ...], tuple[tuple[float, ...], ...]],
        tuple[tuple[float, ...], tuple[tuple[float, ...], ...]],
    ]:
        """正运动学：给一组关节角，回答两臂末端此刻在底盘坐标系的哪里、朝哪。

        返回 ``((左位置, 左旋转), (右位置, 右旋转))``，位置是三元组（米），旋转是
        3×3 行主序矩阵。升降取 ``joints.slide``（为 ``None`` 时退回构造时的
        ``target_slide``）——升降是躯干高度，它和六个臂关节一起决定末端在哪，
        少算它等于把手臂挂在了错误的高度上。

        为什么要有这个方法：逆解只回答「要到那儿关节角该是多少」，回答不了
        「现在实际到哪了」。而「手臂是不是已经在某个位姿上」这类判断必须由
        末端位姿来定——同一个末端位姿在不同躯干高度下对应不同的关节角，拿
        关节角直接比会把躯干高度差混进结论里。
        """

        import numpy as np

        slide = self.target_slide if joints.slide is None else float(joints.slide)
        values = np.asarray((slide, *joints.left, *joints.right), dtype=float)
        left_transform, right_transform = self._backend().forward_kinematics(values)
        return tuple(  # type: ignore[return-value]
            (
                tuple(float(value) for value in transform[:3, 3]),
                tuple(
                    tuple(float(value) for value in transform[row, :3])
                    for row in range(3)
                ),
            )
            for transform in (left_transform, right_transform)
        )

    def solve(self, plan: DualArmHugPlan, seed: JointPair | None) -> tuple[JointPair, ...]:
        left_position, left_rotation = self._world_target(plan.left.contact)
        right_position, right_rotation = self._world_target(plan.right.contact)
        return self.solve_base_targets(
            left_position,
            left_rotation,
            right_position,
            right_rotation,
            seed,
        )


def _rotation_error(
    actual: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
) -> float:
    """返回两个旋转矩阵之间的最短旋转角（弧度）。"""

    relative_trace = sum(
        sum(
            float(target[k][row]) * float(actual[k][row])
            for k in range(3)
        )
        for row in range(3)
    )
    cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
    return math.acos(cosine)


#: 公开别名。求解器内部用它复核逆解，调用方也需要它来回答「实际姿态离目标姿态
#: 差了多少」——例如 APPROACH 出发前核对手臂是否真的停在预抓取位。同一个量在
#: 两处用同一份实现，才不会出现「求解器认为对齐了、调用方认为没有」这类分歧。
rotation_error = _rotation_error
