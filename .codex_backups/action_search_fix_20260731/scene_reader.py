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


def choose_search_area(task):
    if not task:
        return "table_front"

    target = task.get("target") or {}
    place_goal = task.get("place_goal") or {}

    support_surface = target.get("support_surface")
    if support_surface == "shelf":
        return "shelf_front"

    if support_surface == "table":
        return "table_front"

    place_type = place_goal.get("type")
    if place_type in ("shelf_layer", "relative_position"):
        return "shelf_front"

    return "table_front"

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
        and abs(v - config.image_center_v)
        <= config.servo_v_tolerance
    )

    return box_inside and box_large_enough and centroid_inside


def _object_label_text(obj):
    return " ".join(
        str(obj.get(key) or "").lower()
        for key in (
            "label",
            "corrected_label",
            "raw_label",
            "estimated_color",
        )
    )


def _object_matches_task(obj, target):
    label = _object_label_text(obj)
    corrected_label = str(
        obj.get("corrected_label")
        or obj.get("label")
        or ""
    ).lower()
    color = str(target.get("color") or "").lower()
    category = str(target.get("category") or "").lower()
    estimated_color = str(
        obj.get("estimated_color") or ""
    ).lower()

    if corrected_label in ("table", "shelf"):
        return -999.0

    if color:
        if estimated_color and estimated_color != color:
            return -999.0
        if not estimated_color and color not in label:
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
