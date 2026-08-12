import math

from .destination_resolver import destination_level


def _clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


def _object_radius(size_3d, config):
    if not size_3d:
        length = config.place_default_object_length_m
        width = config.place_default_object_width_m
    else:
        length = float(size_3d.get("length") or config.place_default_object_length_m)
        width = float(size_3d.get("width") or config.place_default_object_width_m)
    return 0.5 * max(length, width)


def plan_place_pose(context, scene, config):
    """
    第一版：在目标货架层内有限候选选点。
    不做复杂 3D packing，只避开参考物体邻近，并保证点在货架范围内。
    """
    relation = str(context.destination.place_relation or "").lower()
    explicit_pose = context.destination.place_pose or {}
    if explicit_pose and all(key in explicit_pose for key in ("x", "y")):
        return {
            "x": float(explicit_pose["x"]),
            "y": float(explicit_pose["y"]),
            "z": float(
                explicit_pose.get("z")
                or config.place_default_object_height_m
            ),
            "yaw": float(explicit_pose.get("yaw") or 0.0),
            "frame_id": "world",
        }, "official_place_world_selected"

    resolved = destination_level(context)
    if context.destination.type in ("table_point", "explicit_pose"):
        reference_pose = (
            context.destination.place_pose
            or context.destination.reference_pose_world
            or context.place_reference.pose_world
            or {}
        )
        if not all(key in reference_pose for key in ("x", "y")):
            return None, "table_place_reference_pose_missing"
        return {
            "x": float(reference_pose["x"]),
            "y": float(reference_pose["y"]),
            "z": float(
                reference_pose.get("z")
                or config.place_default_object_height_m
            ),
            "yaw": float(reference_pose.get("yaw") or 0.0),
            "frame_id": "world",
        }, "table_place_pose_selected"

    if resolved is None:
        return None, "destination_level_not_found"

    shelf, level = resolved
    x_min, x_max = shelf["x_range"]
    y_min, y_max = shelf["y_range"]
    margin = float(config.place_level_edge_margin_m)
    radius = _object_radius(context.pick_target.size_3d, config)
    clearance = float(config.place_object_clearance_m) + radius

    x_candidates = [
        x_min + margin + radius,
        (x_min + x_max) * 0.5,
        x_max - margin - radius,
    ]
    y_candidates = [
        y_min + margin + radius,
        (y_min + y_max) * 0.5,
        y_max - margin - radius,
    ]

    reference_pose = context.place_reference.pose_world or {}
    ref_x = reference_pose.get("x")
    ref_y = reference_pose.get("y")

    if relation in ("left_of", "right_of", "front_of", "behind"):
        if ref_x is None or ref_y is None:
            return None, "side_relation_reference_pose_missing"

        side_gap = float(config.place_object_clearance_m) + radius * 2.0
        shelf_yaw = float(shelf["pose"].get("yaw") or 0.0)
        front = (math.cos(shelf_yaw), math.sin(shelf_yaw))
        left = (-math.sin(shelf_yaw), math.cos(shelf_yaw))

        if relation == "left_of":
            dx, dy = left
        elif relation == "right_of":
            dx, dy = -left[0], -left[1]
        elif relation == "front_of":
            dx, dy = front
        else:
            dx, dy = -front[0], -front[1]

        x = _clamp(float(ref_x) + dx * side_gap, x_min + margin + radius, x_max - margin - radius)
        y = _clamp(float(ref_y) + dy * side_gap, y_min + margin + radius, y_max - margin - radius)
        return {
            "x": float(x),
            "y": float(y),
            "z": float(level["z_place"]),
            "yaw": float(shelf["pose"]["yaw"]),
            "frame_id": "world",
        }, f"{relation}_place_pose_selected"

    candidates = []
    for x in x_candidates:
        for y in y_candidates:
            if not (x_min + margin <= x <= x_max - margin):
                continue
            if not (y_min + margin <= y <= y_max - margin):
                continue
            if ref_x is not None and ref_y is not None:
                dist2 = (x - float(ref_x)) ** 2 + (y - float(ref_y)) ** 2
                if dist2 < clearance ** 2:
                    continue
            candidates.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": float(level["z_place"]),
                    "yaw": float(shelf["pose"]["yaw"]),
                    "frame_id": "world",
                }
            )

    if not candidates:
        return None, "no_free_place_candidate"

    # 第一版选离参考物体最近但不碰撞的候选，便于满足“同层”语义。
    if ref_x is not None and ref_y is not None:
        candidates.sort(
            key=lambda p: (p["x"] - float(ref_x)) ** 2 + (p["y"] - float(ref_y)) ** 2
        )

    return candidates[0], "place_pose_selected"
