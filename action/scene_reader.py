import json


def parse_scene_message(message_data):
    return json.loads(message_data)


def get_active_task(scene):
    active_task_id = scene.get("active_task_id")
    task_queue = scene.get("task_queue") or []

    if active_task_id is None and task_queue:
        return task_queue[0]

    for task in task_queue:
        if task.get("task_id") == active_task_id:
            return task

    return task_queue[0] if task_queue else None


def target_is_detected(task):
    if not task:
        return False

    target = task.get("target") or {}

    return bool(
        target.get("object_id")
        and target.get("pose_world")
        and not target.get("requires_reobserve")
    )


def choose_search_areas(task):
    if not task:
        return ["table_front", "shelf_front"]

    target = task.get("target") or {}
    support_surface = target.get("support_surface")
    source_location = str(
        target.get("source_location") or ""
    ).lower()

    if support_surface == "table" or "table" in source_location:
        return ["table_front"]
    if support_surface == "shelf" or "shelf" in source_location:
        return ["shelf_front"]

    # Qwen尚未绑定目标实例时，保留其任务语义，只从“抓取”子句推断来源。
    # 不能读取“放到”之后的货架/桌面，否则会把目的地误当成抓取来源。
    instruction = str(task.get("original_instruction") or "")
    pickup_clause = instruction.split("放到", 1)[0].lower()
    if "桌面" in pickup_clause or "桌子" in pickup_clause or "table" in pickup_clause:
        return ["table_front"]
    if "货架" in pickup_clause or "shelf" in pickup_clause:
        return ["shelf_front"]

    # 目标来源未知时必须搜索两个区域，不能用放置目的地猜抓取来源。
    return ["table_front", "shelf_front"]


def choose_search_area(task):
    return choose_search_areas(task)[0]


def merge_task_target_from_scene(scene, task):
    if not scene or not task:
        return None

    target = dict(task.get("target") or {})
    object_id = target.get("object_id")

    if not object_id:
        return target

    for item in scene.get("objects") or []:
        if item.get("object_id") == object_id:
            merged = dict(target)
            merged.update(item)
            return merged

    return target


def get_servo_vertical_target(target, config):
    is_shelf = (
        target.get("support_surface") == "shelf"
        or bool(target.get("on_shelf"))
    )
    if is_shelf:
        return (
            float(config.servo_target_v_shelf),
            float(config.servo_v_tolerance_shelf),
        )

    return (
        float(config.servo_target_v_table),
        float(config.servo_v_tolerance_table),
    )


def target_is_visible_for_servo(scene, task, config):
    target = merge_task_target_from_scene(scene, task)

    if not target or not target.get("object_id"):
        return False
    if not target.get("pose_world"):
        return False
    if target.get("requires_reobserve"):
        return False

    centroid_uv = target.get("centroid_uv")
    box_xyxy = target.get("box_xyxy")
    if not centroid_uv or len(centroid_uv) < 2:
        return False
    if not box_xyxy or len(box_xyxy) < 4:
        return False

    u = float(centroid_uv[0])
    v = float(centroid_uv[1])
    target_v, v_tolerance = get_servo_vertical_target(
        target,
        config,
    )
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    margin = float(config.servo_bbox_margin_px)

    box_inside = (
        x1 >= margin
        and y1 >= margin
        and x2 <= config.image_width - margin
        and y2 <= config.image_height - margin
    )
    box_large_enough = (
        x2 - x1 >= config.servo_min_bbox_width_px
        and y2 - y1 >= config.servo_min_bbox_height_px
    )
    centroid_inside = (
        0.0 <= u < config.image_width
        and config.servo_min_v <= v <= config.servo_max_v
        and abs(v - target_v) <= v_tolerance
    )

    # 进入视觉伺服时允许目标轻微贴边，因为伺服本身可以
    # 通过转动底盘把目标带回画面。最终ready仍由VisualServo
    # 的完整框检查把关，不能把裁边目标直接交给抓取。
    return box_large_enough and centroid_inside


KNOWN_COLORS = ("pink", "brown", "yellow", "white")


def _canonical_object_label(obj):
    """Return the post-processed label without mixing in stale raw labels."""
    return str(
        obj.get("corrected_label")
        or obj.get("label")
        or obj.get("raw_label")
        or ""
    ).lower()


def _resolved_object_color(obj, canonical_label):
    estimated_color = str(
        obj.get("estimated_color") or ""
    ).lower()
    if estimated_color in KNOWN_COLORS:
        return estimated_color

    for candidate in KNOWN_COLORS:
        if candidate in canonical_label:
            return candidate

    return ""


def _object_matches_task(obj, target):
    label = _canonical_object_label(obj)
    color = str(target.get("color") or "").lower()
    category = str(target.get("category") or "").lower()
    resolved_color = _resolved_object_color(obj, label)

    if label in ("table", "shelf"):
        return -999.0

    if color:
        if resolved_color and resolved_color != color:
            return -999.0
        if not resolved_color:
            return -999.0

    box_family = ("box", "cube", "cuboid", "block")
    if category in box_family:
        if not any(word in label for word in box_family):
            return -999.0
    elif category and category not in label:
        return -999.0

    score = float(
        obj.get("confidence")
        or obj.get("dino_score")
        or 0.0
    )
    if color:
        score += 3.0
    if category:
        score += 2.0

    return score


def _copy_detection_to_target(target, detection, score):
    target.update(
        {
            "object_id": detection.get("object_id"),
            "label": (
                detection.get("corrected_label")
                or detection.get("label")
            ),
            "box_xyxy": detection.get("box_xyxy"),
            "centroid_uv": detection.get("centroid_uv"),
            "mask_area": detection.get("mask_area"),
            "dino_score": detection.get("dino_score"),
            "sam_score": detection.get("sam_score"),
            "raw_label": detection.get("raw_label"),
            "corrected_label": detection.get("corrected_label"),
            "estimated_color": detection.get("estimated_color"),
            "pose_world": detection.get("pose_world"),
            "size_3d": detection.get("size_3d"),
            "support_surface": detection.get("support_surface"),
            "on_table": detection.get("on_table"),
            "on_shelf": detection.get("on_shelf"),
            "shelf_layer": detection.get("shelf_layer"),
            "shelf_layer_confidence": detection.get(
                "shelf_layer_confidence"
            ),
            "confidence": float(score),
            "requires_reobserve": False,
        }
    )
    return target


def _canonical_support_surface(target):
    surface = str(target.get("support_surface") or "").lower()
    if surface in ("table", "table_candidate") or target.get("on_table"):
        return "table"
    if surface == "shelf" or target.get("on_shelf"):
        return "shelf"
    return "unknown"


def _value_in_range(value, limits):
    return float(limits[0]) <= float(value) <= float(limits[1])


def target_is_plausible_for_search(task, area_name, config):
    """Validate one search candidate before it can influence base motion."""
    if not target_is_detected(task):
        return False, "target_not_detected"

    target = task.get("target") or {}
    expected_surface = "shelf" if area_name == "shelf_front" else "table"
    actual_surface = _canonical_support_surface(target)
    if actual_surface != expected_surface:
        return False, f"surface_{actual_surface}_expected_{expected_surface}"

    dino_score = float(target.get("dino_score") or 0.0)
    if dino_score < float(config.search_target_min_dino_score):
        return False, "dino_score_too_low"

    pose = target.get("pose_world") or {}
    if not all(axis in pose for axis in ("x", "y", "z")):
        return False, "pose_world_incomplete"

    bounds = (
        config.shelf_target_bounds_xyz
        if expected_surface == "shelf"
        else config.table_target_bounds_xyz
    )
    if not all(
        _value_in_range(pose[axis], limits)
        for axis, limits in zip(("x", "y", "z"), bounds)
    ):
        return False, "pose_world_outside_workspace"

    box = target.get("box_xyxy") or []
    if len(box) < 4:
        return False, "bbox_missing"
    x1, y1, x2, y2 = [float(value) for value in box]
    # 搜索阶段允许目标贴边：这里只用Pose3D生成较远的预抓取点。
    # 到达后target_is_visible_for_servo()仍会严格要求完整检测框。
    if not (
        x2 > 0.0
        and y2 > 0.0
        and x1 < float(config.image_width)
        and y1 < float(config.image_height)
        and x2 - x1 >= float(config.servo_min_bbox_width_px)
        and y2 - y1 >= float(config.servo_min_bbox_height_px)
    ):
        return False, "bbox_not_search_usable"

    return True, "candidate_valid"


def _clear_stale_target(target):
    for key in (
        "object_id",
        "label",
        "box_xyxy",
        "centroid_uv",
        "mask_area",
        "pose_world",
        "size_3d",
        "support_surface",
        "on_table",
        "on_shelf",
        "shelf_layer",
        "shelf_layer_confidence",
    ):
        target[key] = None

    target["confidence"] = 0.0
    target["requires_reobserve"] = True
    return target


def target_is_servo_usable(task, config):
    if not target_is_detected(task):
        return False

    target = task.get("target") or {}
    centroid_uv = target.get("centroid_uv")
    box_xyxy = target.get("box_xyxy")
    if not centroid_uv or not box_xyxy:
        return False

    v = float(centroid_uv[1])
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    margin = float(config.servo_bbox_margin_px)

    return (
        config.servo_min_v <= v <= config.servo_max_v
        and x1 >= margin
        and y1 >= margin
        and x2 <= config.image_width - margin
        and y2 <= config.image_height - margin
    )


def bind_task_target_from_objects(scene, task):
    if not scene or not task:
        return task

    task = dict(task)
    target = dict(task.get("target") or {})
    objects = scene.get("objects") or []

    if not objects:
        task["target"] = _clear_stale_target(target)
        return task

    existing_id = target.get("object_id")
    if existing_id:
        for obj in objects:
            if obj.get("object_id") != existing_id:
                continue

            existing_score = _object_matches_task(obj, target)
            if existing_score >= 1.0:
                task["target"] = _copy_detection_to_target(
                    target,
                    obj,
                    existing_score,
                )
                return task
            break

    best = max(
        objects,
        key=lambda obj: _object_matches_task(obj, target),
    )
    best_score = _object_matches_task(best, target)

    if best_score < 1.0:
        task["target"] = _clear_stale_target(target)
        return task

    task["target"] = _copy_detection_to_target(
        target,
        best,
        best_score,
    )
    return task
