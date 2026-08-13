import math
from dataclasses import dataclass


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


SEARCH_WAYPOINTS = {
    "table_front": Pose2D(
        x=-0.70,
        # 搜索点要比桌面预抓取点更靠后。固定在 1.55 m 时，
        # 官方场景中桌面目标会贴到相机底边，GroundingDINO/SAM
        # 只能得到裁剪框，导航的完整框安全门槛会拒绝它。
        y=1.30,
        yaw=math.pi / 2.0,
    ),
    # 随机任务的桌面侧边目标可能位于 x=-1.00 或 x=-0.18。
    # 从桌前中点观察时，另一侧目标会贴到图像边缘甚至完全出框。
    # 两个点仍位于桌面南侧安全线外，只改变 x，不改变接近方向。
    "table_front_left": Pose2D(
        x=-1.05,
        y=1.30,
        yaw=math.pi / 2.0,
    ),
    "table_front_right": Pose2D(
        # 桌面右边界约为 x=0.29；在 y=1.30 的桌前角度下，
        # x=-0.35 仍会让随机右侧箱落在图像右边缘。移到 -0.05
        # 后目标居中，同时与桌角保持足够的斜向间距。
        x=-0.05,
        y=1.30,
        yaw=math.pi / 2.0,
    ),
    "shelf_front": Pose2D(
        # 低层货架目标在原站位近距离会顶到相机上沿，
        # 向后退约 10 cm 给完整框和深度反投影留余量。
        x=-1.65,
        y=0.78,
        yaw=math.pi,
    ),
}


def yaw_to_face_target(base_x, base_y, target_x, target_y):
    return math.atan2(
        target_y - base_y,
        target_x - base_x,
    )


def clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _build_table_approach_goal(
    target_x,
    target_y,
    config,
    robot_pose,
    distance_extra,
):
    distance = config.approach_distance_table + distance_extra

    goal_x = clamp(
        target_x,
        *config.table_pick_x_range,
    )

    goal_y = clamp(
        target_y - distance,
        *config.table_pick_y_range,
    )

    return Pose2D(
        x=goal_x,
        y=goal_y,
        yaw=float(config.table_front_yaw),
    )


def build_approach_goal_from_target(
    target,
    config,
    robot_pose=None,
    distance_extra=0.0,
):
    pose = target.get("pose_world") or {}
    surface = target.get("support_surface")
    if surface is None and target.get("on_shelf"):
        surface = "shelf"
    if surface is None:
        surface = "table"

    target_x = float(pose["x"])
    target_y = float(pose["y"])
    extra = max(0.0, float(distance_extra))

    if surface == "shelf":
        # 货架目标的Pose3D在近距离/裁边时会漂到货架内部。
        # 底盘站位必须固定在货架前安全线，只允许沿货架方向小幅对齐。
        shelf_front = SEARCH_WAYPOINTS["shelf_front"]
        goal_x = float(shelf_front.x)
        goal_y = clamp(target_y, 0.55, 1.05)
        yaw = float(shelf_front.yaw)
    else:
        return _build_table_approach_goal(
            target_x,
            target_y,
            config,
            robot_pose,
            extra,
        )

    return Pose2D(x=goal_x, y=goal_y, yaw=yaw)

def pose2d_from_motion_handoff(pose):
    return Pose2D(
        x=float(pose.x),
        y=float(pose.y),
        yaw=float(pose.yaw),
    )

def build_search_goal(area_name):
    return SEARCH_WAYPOINTS.get(
        area_name,
        SEARCH_WAYPOINTS["table_front"],
    )
