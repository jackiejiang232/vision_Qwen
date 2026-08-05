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


def _approach_direction(target_x, target_y, robot_pose, surface):
    if robot_pose is not None:
        dx = float(robot_pose.x) - target_x
        dy = float(robot_pose.y) - target_y
        length = math.hypot(dx, dy)

        if length > 0.20:
            return dx / length, dy / length

    # 无有效里程计方向时才使用赛场固定物体的前方方向。
    if surface == "shelf":
        return 1.0, 0.0

    return 0.0, -1.0


def build_approach_goal_from_target(target, config, robot_pose=None):
    pose = target.get("pose_world") or {}
    surface = target.get("support_surface")
    if surface is None and target.get("on_shelf"):
        surface = "shelf"
    if surface is None:
        surface = "table"

    target_x = float(pose["x"])
    target_y = float(pose["y"])

    if surface == "shelf":
        distance = config.approach_distance_shelf
    else:
        distance = config.approach_distance_table

    direction_x, direction_y = _approach_direction(
        target_x,
        target_y,
        robot_pose,
        surface,
    )
    goal_x = target_x + direction_x * distance
    goal_y = target_y + direction_y * distance
    yaw = yaw_to_face_target(
        goal_x,
        goal_y,
        target_x,
        target_y,
    )

    return Pose2D(x=goal_x, y=goal_y, yaw=yaw)


def build_search_goal(area_name):
    return SEARCH_WAYPOINTS.get(
        area_name,
        SEARCH_WAYPOINTS["table_front"],
    )