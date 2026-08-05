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


def build_approach_goal_from_target(target, config):
    pose = target.get("pose_world") or {}
    surface = target.get("support_surface")

    target_x = float(pose["x"])
    target_y = float(pose["y"])

    if surface == "shelf" or target.get("on_shelf"):
        distance = config.approach_distance_shelf
        # 货架大致在 x=-2.67, y=0.778，机器人应停在货架前方偏 x 增大侧。
        goal_x = target_x + distance
        goal_y = target_y
    else:
        distance = config.approach_distance_table
        # 桌子在 y 约 2.0~2.7，机器人从桌子前方靠近。
        goal_x = target_x
        goal_y = target_y - distance

    yaw = yaw_to_face_target(
        goal_x,
        goal_y,
        target_x,
        target_y,
    )

    return Pose2D(
        x=goal_x,
        y=goal_y,
        yaw=yaw,
    )


def build_search_goal(area_name):
    return SEARCH_WAYPOINTS.get(
        area_name,
        SEARCH_WAYPOINTS["table_front"],
    )