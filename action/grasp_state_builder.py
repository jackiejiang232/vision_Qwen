import math
import time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from third_party.dg202612.contracts import (
    BoxSize,
    CameraId,
    CameraObservation,
    GraspProfile,
    ObjectState,
    PickPlaceGoal,
    Pose2D,
    Pose3D,
    RobotState,
    RobotTargets,
    SceneState,
    TaskId,
)
from third_party.dg202612.executor import PerceptionMotionLimits
from third_party.dg202612.kinematics import OfficialMMK2DualArmSolver
from third_party.dg202612.manipulation import HugProfileGeometry


class MultiSlideDualArmSolver:
    def __init__(self, base, slides, config, collision_evaluator):
        self.base = base
        self.slides = tuple(slides)
        self.config = config
        self.collision_evaluator = collision_evaluator

    def solve(self, plan, seed):
        results = []
        for slide in self.slides:
            solver = OfficialMMK2DualArmSolver(
                self.base,
                slide,
                max_position_error=float(getattr(self.config, "grasp_ik_max_position_error", 0.015)),
                max_orientation_error=float(getattr(self.config, "grasp_ik_max_orientation_error", 0.20)),
                collision_evaluator=self.collision_evaluator,
            )
            results.extend(solver.solve(plan, seed))
        return tuple(results)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def joint_value(message: JointState, name: str, default=None):
    if message is None:
        return default
    try:
        index = message.name.index(name)
    except ValueError:
        return default
    if index >= len(message.position):
        return default
    return float(message.position[index])


def joint_tuple(message: JointState, names):
    values = []
    missing = []
    for name in names:
        value = joint_value(message, name)
        if value is None:
            missing.append(name)
        else:
            values.append(value)
    if missing:
        raise ValueError(f"/joint_states missing joints: {missing}")
    return tuple(values)


def robot_state_from_ros(odom: Odometry, joints: JointState, config, now=None):
    if odom is None:
        raise ValueError("missing odom")
    if joints is None:
        raise ValueError("missing joint_states")
    stamp = time.time() if now is None else float(now)
    pose = odom.pose.pose
    twist = odom.twist.twist
    return RobotState(
        base=Pose2D(
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        ),
        base_linear=float(twist.linear.x),
        base_angular=float(twist.angular.z),
        slide=float(joint_value(joints, config.slide_joint_name, 0.0)),
        head_yaw=float(joint_value(joints, config.head_yaw_joint_name, 0.0)),
        head_pitch=float(joint_value(joints, config.head_pitch_joint_name, 0.0)),
        left_arm=joint_tuple(joints, config.left_arm_joint_names),
        left_gripper=float(joint_value(joints, config.left_gripper_joint_name, 1.0)),
        right_arm=joint_tuple(joints, config.right_arm_joint_names),
        right_gripper=float(joint_value(joints, config.right_gripper_joint_name, 1.0)),
        observed_at=stamp,
    )


def measured_targets(robot: RobotState):
    return RobotTargets(
        base_linear=0.0,
        base_angular=0.0,
        slide=robot.slide,
        head_yaw=robot.head_yaw,
        head_pitch=robot.head_pitch,
        left_arm=robot.left_arm,
        left_gripper=robot.left_gripper,
        right_arm=robot.right_arm,
        right_gripper=robot.right_gripper,
    )


def task_id_from_ready(value):
    if int(value) == 1:
        return TaskId.TASK_1
    if int(value) == 2:
        return TaskId.TASK_2
    if int(value) == 3:
        return TaskId.TASK_3
    raise ValueError(f"unknown task_id: {value}")


def profile_from_ready(ready):
    raw = ready.get("motion_grasp_profile") or ""
    if raw == GraspProfile.TABLE_SIDE_HUG.value:
        return GraspProfile.TABLE_SIDE_HUG
    if raw == GraspProfile.SHELF_EXTRACT_HUG.value:
        return GraspProfile.SHELF_EXTRACT_HUG
    if raw == GraspProfile.TABLE_TOP_HUG.value:
        return GraspProfile.TABLE_TOP_HUG
    raise ValueError(f"unsupported motion_grasp_profile: {raw!r}")


def pose3d_from_dict(value):
    if not isinstance(value, dict):
        raise ValueError("pose must be a dict")
    return Pose3D(
        float(value["x"]),
        float(value["y"]),
        float(value["z"]),
        float(value.get("roll", 0.0)),
        float(value.get("pitch", 0.0)),
        float(value.get("yaw", 0.0)),
    )


def box_size_from_ready(ready):
    size = ready.get("target_size_3d")
    if not isinstance(size, dict):
        raise ValueError("ready_for_grasp missing target_size_3d")
    return BoxSize(
        float(size["length"]),
        float(size["width"]),
        float(size["height"]),
    )


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def normalize_table_side_target_pose_and_size(pose, size, ready, robot):
    if ready.get("motion_grasp_profile") != GraspProfile.TABLE_SIDE_HUG.value:
        return pose, size
    # DG202612 的桌边抱持标定使用官方粉色箱尺寸，且 pose.z 要传箱体中心。
    # 当前视觉 z 常落在箱体上表面附近，直接给 IK 会整体偏高。
    # pose.yaw 不是检测框 yaw，而是抱持坐标系前向；官方桌边基线取底盘抓取朝向。
    normalized_size = BoxSize(0.160, 0.240, 0.190)
    z = pose.z
    if z > 0.82:
        z = z - normalized_size.height / 2.0
    yaw = normalize_angle(robot.base.yaw)
    normalized_pose = Pose3D(
        pose.x,
        pose.y,
        z,
        pose.roll,
        pose.pitch,
        yaw,
    )
    return normalized_pose, normalized_size


def scene_goal_from_ready(ready, robot: RobotState, now=None):
    stamp = time.time() if now is None else float(now)
    if not ready.get("ready_for_grasp"):
        raise ValueError("ready_for_grasp is false")
    if ready.get("motion_source_area") is None:
        raise ValueError("ready_for_grasp missing motion_source_area")
    if ready.get("motion_grasp_profile") is None:
        raise ValueError("ready_for_grasp missing motion_grasp_profile")

    target_pose = pose3d_from_dict(ready["target_pose_world"])
    target_size = box_size_from_ready(ready)
    target_pose, target_size = normalize_table_side_target_pose_and_size(
        target_pose,
        target_size,
        ready,
        robot,
    )
    target_id = str(ready["target_object_id"])
    label = str(ready.get("target_label") or "box")
    color = label.split()[0] if label.split() else label
    confidence = float(ready.get("confidence", 1.0))
    if confidence > 1.0:
        confidence = 1.0

    target = ObjectState(
        object_id=target_id,
        color=color,
        pose=target_pose,
        size=target_size,
        observed_at=stamp,
        confidence=confidence,
        source_cameras=(CameraId.HEAD_RGBD,),
        position_std_m=ready.get("position_std_m", 0.02) or 0.02,
        yaw_std_rad=ready.get("yaw_std_rad", 0.05) or 0.05,
    )
    scene = SceneState(
        timestamp=stamp,
        robot=robot,
        objects=(target,),
        camera_observations=(
            CameraObservation(CameraId.HEAD_RGBD, stamp),
            CameraObservation(CameraId.LEFT_WRIST_RGB, stamp),
            CameraObservation(CameraId.RIGHT_WRIST_RGB, stamp),
        ),
    )
    goal = PickPlaceGoal(
        task_id=task_id_from_ready(ready.get("task_id", 1)),
        target_id=target.object_id,
        target_color=target.color,
        target_pose=target.pose,
        target_size=target.size,
        source_area=str(ready["motion_source_area"]),
        grasp_profile=profile_from_ready(ready),
        place_type=ready.get("place_type"),
    )
    return scene, goal


def table_side_hug_profile(config):
    return HugProfileGeometry(
        GraspProfile.TABLE_SIDE_HUG,
        contact_press=float(config.table_side_contact_press),
        pregrasp_gap=float(config.table_side_pregrasp_gap),
        contact_longitudinal_offset=float(config.table_side_contact_longitudinal_offset),
        contact_height_offset=float(config.table_side_contact_height_offset),
        gripper_opening=float(config.table_side_gripper_opening),
    )


def perception_limits(config):
    return PerceptionMotionLimits(
        coarse_max_age=float(config.grasp_coarse_max_age),
        coarse_min_confidence=float(config.grasp_coarse_min_confidence),
        coarse_position_std_m=float(config.grasp_coarse_position_std_m),
        fine_max_age=float(config.grasp_fine_max_age),
        fine_min_confidence=float(config.grasp_fine_min_confidence),
        fine_position_std_m=float(config.grasp_fine_position_std_m),
        fine_yaw_std_rad=float(config.grasp_fine_yaw_std_rad),
        approach_max_age=float(config.grasp_approach_max_age),
        max_centered_error_m=float(config.grasp_max_centered_error_m),
        redock_position_tolerance_m=float(config.grasp_redock_position_tolerance_m),
        redock_yaw_tolerance_rad=float(config.grasp_redock_yaw_tolerance_rad),
        observation_extra_standoff_m=float(config.grasp_observation_extra_standoff_m),
    )


def make_solver(robot: RobotState, config):
    collision_evaluator = None
    if bool(config.grasp_allow_unchecked_ik):
        collision_evaluator = lambda _candidate: True
    center = float(robot.slide)
    slide_min = float(getattr(config, "table_pregrasp_spine_min", -0.04))
    slide_max = float(getattr(config, "table_pregrasp_spine_max", 0.87))
    # slide 数值越大，腰部越低。桌边抱箱优先保持当前高度或继续降低腰部，
    # 避免抓取 IK 覆盖导航阶段的预抓取腰部设置后又把身体抬高。
    offsets = (0.0, 0.02, 0.04, 0.06, 0.08)
    slides = []
    for offset in offsets:
        value = min(slide_max, max(slide_min, center + offset))
        if all(abs(value - old) > 1e-4 for old in slides):
            slides.append(value)
    return MultiSlideDualArmSolver(
        robot.base,
        slides,
        config,
        collision_evaluator=collision_evaluator,
    )
