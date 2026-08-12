import math

from .shelf_config import all_shelves, default_shelf_approach_pose, default_shelf_id, shelf_by_id


TABLE_APPROACH_POSE = {"x": -0.70, "y": 1.55, "yaw": math.pi / 2.0}


def _clamp(value, limits):
    if not limits or len(limits) < 2:
        return float(value)
    return max(float(limits[0]), min(float(limits[1]), float(value)))


def table_approach_pose_for_reference(reference_pose_world, config=None):
    """Build a table approach pose aligned with a historical target pose."""
    reference_pose = reference_pose_world or {}
    if not all(axis in reference_pose for axis in ("x", "y")):
        return dict(TABLE_APPROACH_POSE)

    distance = float(
        getattr(config, "approach_distance_table", 0.55)
        if config is not None
        else 0.55
    )
    x_range = getattr(config, "table_pick_x_range", None) if config else None
    y_range = getattr(config, "table_pick_y_range", None) if config else None
    table_yaw = float(
        getattr(config, "table_front_yaw", TABLE_APPROACH_POSE["yaw"])
        if config is not None
        else TABLE_APPROACH_POSE["yaw"]
    )

    return {
        "x": _clamp(float(reference_pose["x"]), x_range),
        "y": _clamp(float(reference_pose["y"]) - distance, y_range),
        "yaw": table_yaw,
    }


def _pose_xyz(pose):
    if not pose:
        return None
    if all(k in pose for k in ("x", "y", "z")):
        return float(pose["x"]), float(pose["y"]), float(pose["z"])
    return None


def resolve_shelf_level_from_reference(reference_pose_world):
    xyz = _pose_xyz(reference_pose_world)
    if xyz is None:
        return None

    x, y, z = xyz
    for shelf_id, shelf in all_shelves():
        x_min, x_max = shelf["x_range"]
        y_min, y_max = shelf["y_range"]
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue

        for level in shelf["levels"]:
            if level["z_min"] <= z <= level["z_max"]:
                return {
                    "type": "shelf_level",
                    "shelf_id": shelf_id,
                    "level_id": int(level["id"]),
                    "reference_pose_world": reference_pose_world,
                    "approach_pose": dict(shelf["approach_pose"]),
                }

    return None


def shelf_for_pose(pose_world):
    xyz = _pose_xyz(pose_world)
    if xyz is None:
        return None

    x, y, _ = xyz
    for shelf_id, shelf in all_shelves():
        x_min, x_max = shelf["x_range"]
        y_min, y_max = shelf["y_range"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return shelf_id, shelf
    return None


def is_shelf_place_type(place_type):
    place_type = str(place_type or "").lower()
    return place_type.startswith("shelf") or "shelf" in place_type


def resolve_destination(context, config=None):
    relation = str(context.destination.place_relation or "").lower()
    place_type = str(context.destination.place_type or "").lower()

    if context.destination.place_pose:
        context.destination.type = place_type or "explicit_pose"
        if not context.destination.approach_pose:
            shelf_match = shelf_for_pose(context.destination.place_pose)
            if shelf_match is not None:
                shelf_id, shelf = shelf_match
                context.destination.shelf_id = context.destination.shelf_id or shelf_id
                context.destination.approach_pose = dict(shelf["approach_pose"])
            elif is_shelf_place_type(place_type):
                context.destination.shelf_id = (
                    context.destination.shelf_id
                    or default_shelf_id()
                )
                context.destination.approach_pose = default_shelf_approach_pose()
            else:
                context.destination.approach_pose = table_approach_pose_for_reference(
                    context.destination.place_pose,
                    config,
                )
        return True, "explicit_place_pose_resolved"

    if relation == "original_position_of" and context.place_reference.pose_world:
        context.destination.type = place_type or "table_point"
        context.destination.reference_object_id = context.place_reference.object_id
        context.destination.reference_pose_world = context.place_reference.pose_world
        context.destination.approach_pose = table_approach_pose_for_reference(
            context.place_reference.pose_world,
            config,
        )
        return True, "original_position_destination_resolved"

    result = resolve_shelf_level_from_reference(
        context.place_reference.pose_world
    )
    if result is None:
        return False, "reference_object_not_inside_configured_shelf"

    if relation in ("left_of", "right_of", "front_of", "behind"):
        context.destination.type = place_type or "shelf_object_side"
    else:
        context.destination.type = result["type"]
    context.destination.shelf_id = result["shelf_id"]
    context.destination.level_id = result["level_id"]
    context.destination.reference_object_id = context.place_reference.object_id
    context.destination.reference_pose_world = result["reference_pose_world"]
    context.destination.approach_pose = result["approach_pose"]
    return True, "destination_resolved"


def destination_level(context):
    shelf = shelf_by_id(context.destination.shelf_id)
    if shelf is None:
        return None
    for level in shelf["levels"]:
        if int(level["id"]) == int(context.destination.level_id):
            return shelf, level
    return None
