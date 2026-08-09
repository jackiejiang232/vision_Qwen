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
        y=1.55,
        yaw=math.pi / 2.0,
    ),
    "shelf_front": Pose2D(
        x=-1.75,
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
        distance = config.approach_distance_shelf + extra
        # 货架可接近面位于世界坐标x增大的一侧。
        goal_x = target_x + distance
        goal_y = target_y
        yaw = yaw_to_face_target(
            goal_x,
            goal_y,
            target_x,
            target_y,
        )
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
