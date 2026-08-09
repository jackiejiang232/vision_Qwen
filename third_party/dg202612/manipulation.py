"""MMK2 双臂抱持几何。

本模块只负责一件事：把箱体侧面上的期望接触区域，换算成官方 KDL 所需的左右
末端位姿。夹持主接触面是 ``arm_link6`` 上的长方体侧面；三角指爪保持张开，
只作为上下防转辅助，不把指尖参考点冒充成接触面。

输入 ``ObjectState.pose.yaw`` 表示本次抱持坐标系的前向轴，``size.width`` 表示左右
两接触面之间的箱体宽度。视觉层以后只需给出箱体位姿和尺寸，运动层不依赖固定
世界坐标。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .contracts import GraspProfile, ObjectState, Pose3D


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


@dataclass(frozen=True)
class ToolContactSurface:
    """官方 KDL 末端坐标系中的一个物理接触面。"""

    center_local: Vector3
    inward_normal_local: Vector3
    half_length: float
    half_height: float


# 来源：官方 arm_left.xml / arm_right.xml 的 arm_link6 box geom。
# MJCF body 坐标换到 lft/rgt_endpoint(KDL) 坐标后，长方体侧面中心为
# (-0.07, 0, -0.025)，面内尺寸为 30 mm × 160 mm。
MMK2_LINK6_SIDE_SURFACE = ToolContactSurface(
    center_local=(-0.070, 0.0, -0.025),
    inward_normal_local=(0.0, 0.0, -1.0),
    half_length=0.015,
    half_height=0.080,
)


@dataclass(frozen=True)
class HugProfileGeometry:
    """一种抱持方式中具有明确物理含义的参数。"""

    name: GraspProfile
    # 接触面目标进入箱体表面的命令虚位移；真实刚体会挡住手臂并产生夹持力。
    contact_press: float
    # 预抓取时，link6 主接触面离箱体侧面的空气间隙。
    pregrasp_gap: float
    # 接触面中心在抱持坐标系前向轴上的偏移。
    contact_longitudinal_offset: float
    # 接触面中心相对箱体中心的高度偏移。
    contact_height_offset: float
    # 官方基线在抱持期间保持 1.0（全开）；开度不参与横向挤压。
    gripper_opening: float = 1.0

    def __post_init__(self) -> None:
        if self.contact_press < 0.0:
            raise ValueError("contact_press must be non-negative")
        if self.pregrasp_gap <= 0.0:
            raise ValueError("pregrasp_gap must be positive")
        if not 0.0 <= self.gripper_opening <= 1.0:
            raise ValueError("gripper_opening must be within [0, 1]")


@dataclass(frozen=True)
class ArmContact:
    arm: str
    # 下面两个 Pose3D 是 KDL endpoint 目标，可直接交给官方 IK。
    pregrasp: Pose3D
    contact: Pose3D
    # 下面两个 Pose3D 是真正要贴近箱体的 link6 面中心，供审核和反馈判断。
    pregrasp_surface: Pose3D
    contact_surface: Pose3D
    outward_normal: Vector3
    surface_tangent: Vector3


@dataclass(frozen=True)
class DualArmHugPlan:
    profile: GraspProfile
    left: ArmContact
    right: ArmContact
    gripper_opening: float = 1.0


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> Matrix3:
    """标准 ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``。"""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector3:
    return tuple(  # type: ignore[return-value]
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )


def ideal_tool_rpy(arm: str, grasp_yaw: float) -> Vector3:
    """让 link6 宽面与箱体侧面平行的解析姿态。"""

    if arm == "left":
        return (-math.pi / 2.0, 0.0, grasp_yaw)
    if arm == "right":
        return (math.pi / 2.0, 0.0, grasp_yaw)
    raise ValueError(f"unknown arm: {arm}")


def endpoint_pose_for_surface(
    surface_pose: Pose3D,
    *,
    surface: ToolContactSurface = MMK2_LINK6_SIDE_SURFACE,
) -> Pose3D:
    """将物理面中心目标换算为 KDL endpoint 目标。"""

    rotation = _rpy_matrix(surface_pose.roll, surface_pose.pitch, surface_pose.yaw)
    offset = _matvec(rotation, surface.center_local)
    return Pose3D(
        surface_pose.x - offset[0],
        surface_pose.y - offset[1],
        surface_pose.z - offset[2],
        surface_pose.roll,
        surface_pose.pitch,
        surface_pose.yaw,
    )


def contact_surface_from_endpoint(
    endpoint_position: Sequence[float],
    endpoint_rotation: Sequence[Sequence[float]],
    *,
    surface: ToolContactSurface = MMK2_LINK6_SIDE_SURFACE,
) -> tuple[Vector3, Vector3]:
    """由 FK 末端位姿还原 ``(link6 面中心, 面的向内法向)``。"""

    offset = _matvec(endpoint_rotation, surface.center_local)
    normal = _matvec(endpoint_rotation, surface.inward_normal_local)
    center = tuple(  # type: ignore[assignment]
        float(endpoint_position[index]) + offset[index] for index in range(3)
    )
    return center, normal


def _world_xy(pose: Pose3D, local_x: float, local_y: float) -> tuple[float, float]:
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    return (
        pose.x + cosine * local_x - sine * local_y,
        pose.y + sine * local_x + cosine * local_y,
    )


def _surface_pose(
    target: ObjectState,
    profile: HugProfileGeometry,
    arm: str,
    clearance: float,
) -> tuple[Pose3D, Vector3, Vector3]:
    """生成某一侧在指定表面间隙下的 link6 面中心位姿。"""

    side = 1.0 if arm == "left" else -1.0
    pose = target.pose
    half_width = target.size.width / 2.0
    x, y = _world_xy(
        pose,
        profile.contact_longitudinal_offset,
        side * (half_width + clearance),
    )
    roll, pitch, yaw = ideal_tool_rpy(arm, pose.yaw)
    tangent = (math.cos(pose.yaw), math.sin(pose.yaw), 0.0)
    outward = (-side * math.sin(pose.yaw), side * math.cos(pose.yaw), 0.0)
    return (
        Pose3D(
            x,
            y,
            pose.z + profile.contact_height_offset,
            roll,
            pitch,
            yaw,
        ),
        outward,
        tangent,
    )


def _arm_contact(
    target: ObjectState,
    profile: HugProfileGeometry,
    arm: str,
) -> ArmContact:
    pre_surface, outward, tangent = _surface_pose(
        target, profile, arm, profile.pregrasp_gap
    )
    contact_surface, _, _ = _surface_pose(
        target, profile, arm, -profile.contact_press
    )
    return ArmContact(
        arm=arm,
        pregrasp=endpoint_pose_for_surface(pre_surface),
        contact=endpoint_pose_for_surface(contact_surface),
        pregrasp_surface=pre_surface,
        contact_surface=contact_surface,
        outward_normal=outward,
        surface_tangent=tangent,
    )


def dual_arm_hug(target: ObjectState, profile: HugProfileGeometry) -> DualArmHugPlan:
    """生成左右 link6 宽面平行、法向相反的预抓取与抱持目标。"""

    return DualArmHugPlan(
        profile=profile.name,
        left=_arm_contact(target, profile, "left"),
        right=_arm_contact(target, profile, "right"),
        gripper_opening=profile.gripper_opening,
    )


def hug_approach_sequence(
    target: ObjectState,
    profile: HugProfileGeometry,
    max_surface_step: float,
) -> tuple[DualArmHugPlan, ...]:
    """沿箱体法向生成从预抓取间隙到夹持虚位移的笛卡尔采样。"""

    if max_surface_step <= 0.0:
        raise ValueError("max_surface_step must be positive")
    start = profile.pregrasp_gap
    end = -profile.contact_press
    steps = max(1, math.ceil((start - end) / max_surface_step))
    result: list[DualArmHugPlan] = []
    nominal = dual_arm_hug(target, profile)
    for index in range(steps + 1):
        clearance = start + (end - start) * index / steps
        arms: list[ArmContact] = []
        for arm, reference in (("left", nominal.left), ("right", nominal.right)):
            surface_pose, outward, tangent = _surface_pose(
                target, profile, arm, clearance
            )
            endpoint = endpoint_pose_for_surface(surface_pose)
            arms.append(
                ArmContact(
                    arm=arm,
                    pregrasp=reference.pregrasp,
                    contact=endpoint,
                    pregrasp_surface=reference.pregrasp_surface,
                    contact_surface=surface_pose,
                    outward_normal=outward,
                    surface_tangent=tangent,
                )
            )
        result.append(
            DualArmHugPlan(
                profile.name,
                arms[0],
                arms[1],
                profile.gripper_opening,
            )
        )
    return tuple(result)


def side_faces_are_parallel(plan: DualArmHugPlan, tolerance: float = 1e-9) -> bool:
    """校验真实 link6 面法向，而不是校验人为填入的标签。"""

    measured = []
    for contact in (plan.left, plan.right):
        pose = contact.contact
        rotation = _rpy_matrix(pose.roll, pose.pitch, pose.yaw)
        measured.append(_matvec(rotation, MMK2_LINK6_SIDE_SURFACE.inward_normal_local))
    normals_opposite = math.dist(measured[0], tuple(-v for v in measured[1])) <= tolerance
    expected = (
        tuple(-v for v in plan.left.outward_normal),
        tuple(-v for v in plan.right.outward_normal),
    )
    aligned = all(math.dist(actual, wanted) <= tolerance for actual, wanted in zip(measured, expected))
    return normals_opposite and aligned
