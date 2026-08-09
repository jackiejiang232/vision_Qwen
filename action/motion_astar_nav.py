import json
from pathlib import Path

from third_party.dg202612.navigation import (
    AxisAlignedRect,
    BaseFootprint,
    DockController,
    PathFollower,
    StaticAStarPlanner,
    StaticMap,
    plan_dock_route,
)
from third_party.dg202612.contracts import Pose2D as DGPose2D


def _rect_from_list(values):
    # DG202612 的 AxisAlignedRect 顺序是 min_x, min_y, max_x, max_y
    return AxisAlignedRect(
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def load_static_planner(scene_geometry_path):
    data = json.loads(Path(scene_geometry_path).read_text())
    bounds = _rect_from_list(data["map"]["bounds"])
    obstacles = tuple(
        _rect_from_list(item)
        for item in data.get("static_obstacles", [])
    )
    static_map = StaticMap(
        bounds=bounds,
        obstacles=obstacles,
        resolution=float(data["map"]["resolution"]),
        safety_margin=float(data["map"]["safety_margin"]),
    )
    return StaticAStarPlanner(static_map)


def load_base_footprint(robot_geometry_path):
    data = json.loads(Path(robot_geometry_path).read_text())
    bounds = data["base_footprint"]["bounds_in_base"]
    return BaseFootprint(
        min_x=float(bounds[0]),
        min_y=float(bounds[1]),
        max_x=float(bounds[2]),
        max_y=float(bounds[3]),
    )


def to_dg_pose2d(pose):
    return DGPose2D(
        x=float(pose.x),
        y=float(pose.y),
        yaw=float(pose.yaw),
    )


def make_dock_route(robot_pose, goal_pose, config):
    planner = load_static_planner(config.dg_scene_geometry_path)
    footprint = load_base_footprint(config.dg_robot_geometry_path)
    return plan_dock_route(
        planner,
        to_dg_pose2d(robot_pose),
        to_dg_pose2d(goal_pose),
        footprint,
        final_approach_distance=float(config.astar_final_approach_distance),
        footprint_clearance=float(config.astar_footprint_clearance),
    )


def make_dock_controller(route, config):
    follower = PathFollower(
        position_tolerance=float(config.astar_position_tolerance),
        yaw_tolerance=float(config.astar_yaw_tolerance),
        max_linear=float(config.astar_max_linear_speed),
        max_angular=float(config.astar_max_angular_speed),
    )
    return DockController(
        route,
        follower,
        stable_duration=float(config.astar_dock_stable_sec),
    )