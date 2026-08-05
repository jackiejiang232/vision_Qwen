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

    if not target:
        return False

    if not target.get("object_id"):
        return False

    if not target.get("pose_world"):
        return False

    if target.get("requires_reobserve"):
        return False

    centroid_uv = target.get("centroid_uv")
    if not centroid_uv or len(centroid_uv) < 2:
        return False

    v = float(centroid_uv[1])

    # 目标太靠上/太靠下，都不适合直接进入视觉伺服。
    # 底盘旋转只能调水平中心，不能解决目标在图像最下方的问题。
    return (
        config.servo_min_v
        <= v
        <= config.servo_max_v
    )
def _text_contains(text, keyword):
    return keyword and keyword in str(text).lower()


def _object_matches_task(obj, target):
    label = str(obj.get("label") or "").lower()
    corrected_label = str(
        obj.get("corrected_label") or ""
    ).lower()
    raw_label = str(obj.get("raw_label") or "").lower()

    full_label = " ".join(
        [label, corrected_label, raw_label]
    )

    color = str(target.get("color") or "").lower()
    category = str(target.get("category") or "").lower()

    score = 0.0

    if color and color in full_label:
        score += 1.0

    if category and category in full_label:
        score += 0.8

    if category == "box" and any(
        word in full_label
        for word in ("box", "cube", "cuboid")
    ):
        score += 0.5

    score += float(obj.get("confidence") or 0.0)

    return score

def target_is_servo_usable(task, config):
    if not target_is_detected(task):
        return False

    target = task.get("target") or {}
    centroid_uv = target.get("centroid_uv")

    if not centroid_uv or len(centroid_uv) < 2:
        return False

    u = float(centroid_uv[0])
    v = float(centroid_uv[1])

    u_ok = abs(u - config.image_center_u) < 180.0
    v_ok = config.servo_min_v <= v <= config.servo_max_v

    return u_ok and v_ok

def bind_task_target_from_objects(scene, task):
    if not scene or not task:
        return task

    task = dict(task)
    target = dict(task.get("target") or {})

    if (
        target.get("object_id")
        and target.get("pose_world")
        and not target.get("requires_reobserve")
    ):
        return task

    objects = scene.get("objects") or []
    if not objects:
        return task

    best = max(
        objects,
        key=lambda obj: _object_matches_task(obj, target),
    )

    best_score = _object_matches_task(best, target)

    if best_score < 1.0:
        return task

    target.update(
        {
            "object_id": best.get("object_id"),
            "label": best.get("corrected_label")
            or best.get("label"),
            "pose_world": best.get("pose_world"),
            "size_3d": best.get("size_3d"),
            "box_xyxy": best.get("box_xyxy"),
            "centroid_uv": best.get("centroid_uv"),
            "support_surface": best.get("support_surface"),
            "on_table": best.get("on_table"),
            "on_shelf": best.get("on_shelf"),
            "shelf_layer": best.get("shelf_layer"),
            "confidence": best_score,
            "requires_reobserve": False,
        }
    )

    task["target"] = target
    return task