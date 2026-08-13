import json

def _surface_from_flat_scene(scene):
    source = str(scene.get("source_location") or "").lower()

    if scene.get("on_shelf") or "shelf" in source:
        return "shelf"
    if scene.get("on_table") or "table" in source:
        return "table"

    return None

def _normalize_flat_scene(scene):
    """
    兼容新的扁平 scene_understanding 输出。

    新格式示例：
    {
      "task_id": 1,
      "target_object_id": "...",
      "target_label": "pink box",
      "target_pose_world": {...},
      "source_location": "table",
      "place_type": "shelf_layer",
      ...
    }

    转成动作节点原来需要的：
    active_task_id + task_queue + objects + grounding
    """
    task_id = scene.get("task_id")
    target_object_id = scene.get("target_object_id")
    target_label = scene.get("target_label")
    target_pose_world = scene.get("target_pose_world")
    reference_pose_world = (
        scene.get("reference_pose_world")
        or scene.get("place_pose_world")
    )
    requires_reobserve = bool(scene.get("requires_reobserve", True))
    support_surface = _surface_from_flat_scene(scene)

    target = {
        "object_id": target_object_id,
        "label": target_label,
        "color": scene.get("target_color"),
        "category": scene.get("target_category"),
        "pose_world": target_pose_world,
        "size_3d": scene.get("target_size_3d") or scene.get("size_3d"),
        "source_location": scene.get("source_location"),
        "support_surface": support_surface,
        "on_table": scene.get("on_table"),
        "on_shelf": scene.get("on_shelf"),
        "shelf_layer": scene.get("shelf_layer"),
        "confidence": scene.get("confidence", 0.0),
        "requires_reobserve": requires_reobserve,
    }

    place_goal = {
        "type": scene.get("place_type"),
        "place_type": scene.get("place_type"),
        "reference_object_id": scene.get("reference_object_id"),
        "reference_label": scene.get("reference_label"),
        "reference_pose_world": reference_pose_world,
        "temporal_reference": scene.get("temporal_reference"),
        "spatial_relation": scene.get("spatial_relation"),
        "pose_world": scene.get("place_pose_world"),
        "requires_planning": True,
    }

    objects = []
    if target_object_id and target_pose_world:
        objects.append(
            {
                "object_id": target_object_id,
                "label": target_label,
                "semantic_role": "active_task_target",
                "location": support_surface or "unknown",
                "pose_world": target_pose_world,
                "size_3d": target.get("size_3d"),
                "support_surface": support_surface,
                "on_table": scene.get("on_table"),
                "on_shelf": scene.get("on_shelf"),
                "shelf_layer": scene.get("shelf_layer"),
                "confidence": scene.get("confidence", 0.0),
                "dino_score": scene.get("confidence", 0.0),
                "requires_reobserve": requires_reobserve,
            }
        )

    # 连续任务需要在任务开始时同时 Ground 放置参考物体（例如白色圆柱）。
    reference_object_id = scene.get("reference_object_id")
    reference_label = scene.get("reference_label")

    if reference_object_id and reference_pose_world:
        objects.append(
            {
                "object_id": reference_object_id,
                "label": reference_label,
                "semantic_role": "place_reference",
                "location": "shelf",
                "pose_world": reference_pose_world,
                "support_surface": "shelf",
                "on_shelf": True,
                "confidence": scene.get("reference_confidence", 0.0),
                "dino_score": scene.get("reference_confidence", 0.0),
                "requires_reobserve": False,
            }
        )

    return {
        "schema_version": "flat_compat_1.0",
        "source_stamp_sec": scene.get("source_stamp_sec", 0),
        "source_stamp_nanosec": scene.get("source_stamp_nanosec", 0),
        "active_task_id": task_id,
        "scene_summary": (
            f"兼容扁平scene：任务{task_id}，目标{target_label}"
        ),
        "objects": objects,
        "grounding": {
            "selected_object_id": target_object_id,
            "selected_label": target_label,
            "reason": "flat_scene_compat",
            "confidence": scene.get("confidence", 0.0),
            "requires_reobserve": requires_reobserve,
        },
        "task_queue": [
            {
                "task_id": task_id,
                "status": "pending",
                "original_instruction": scene.get("original_instruction", ""),
                "target": target,
                "place_goal": place_goal,
                "uncertainties": (
                    ["target_requires_reobserve"]
                    if requires_reobserve
                    else []
                ),
            }
        ],
    }


def parse_scene_message(message_data):
    scene = json.loads(message_data)

    # 旧格式：直接返回
    if "task_queue" in scene or "active_task_id" in scene:
        return scene

    # 新扁平格式：自动补成动作模块需要的旧格式
    if "task_id" in scene and "target_object_id" in scene:
        return _normalize_flat_scene(scene)

    return scene


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
        instruction = str(task.get("original_instruction") or "")
        pickup_clause = instruction.split("放到", 1)[0].lower()
        # 官方随机任务会在抓取子句中给出“桌面左侧/右侧”。
        # 先去对应侧的观测点，避免桌前中点把侧边目标推到相机边缘；
        # 另一侧仍保留为泛化兜底，适应没有侧别提示的任务。
        if "右侧" in pickup_clause or "右边" in pickup_clause:
            return ["table_front_right", "table_front_left"]
        if "左侧" in pickup_clause or "左边" in pickup_clause:
            return ["table_front_left", "table_front_right"]
        return ["table_front_right", "table_front_left"]
    if support_surface == "shelf" or "shelf" in source_location:
        return ["shelf_front"]

    # Qwen尚未绑定目标实例时，保留其任务语义，只从“抓取”子句推断来源。
    # 不能读取“放到”之后的货架/桌面，否则会把目的地误当成抓取来源。
    instruction = str(task.get("original_instruction") or "")
    pickup_clause = instruction.split("放到", 1)[0].lower()
    if "桌面" in pickup_clause or "桌子" in pickup_clause or "table" in pickup_clause:
        if "右侧" in pickup_clause or "右边" in pickup_clause:
            return ["table_front_right", "table_front_left"]
        if "左侧" in pickup_clause or "左边" in pickup_clause:
            return ["table_front_left", "table_front_right"]
        return ["table_front_right", "table_front_left"]
    if "货架" in pickup_clause or "shelf" in pickup_clause:
        return ["shelf_front"]

    # 目标来源未知时必须搜索两个区域，不能用放置目的地猜抓取来源。
    # 比赛场景里桌面目标通常在初始/回原点视角已可见；真正缺失位姿的
    # 目标更常见于货架遮挡，因此优先近距离看货架，再回桌面兜底。
    return ["shelf_front", "table_front"]


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
    bbox_area_ratio = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
        / float(config.image_width * config.image_height)
    )
    target_surface = str(target.get("support_surface") or "").lower()
    is_shelf_target = target_surface == "shelf" or bool(target.get("on_shelf"))
    is_table_dual_target = (
        not is_shelf_target
        and target.get("selected_arm") == "dual"
    )

    # 目标框覆盖过大时通常是箱子与桌面/相邻箱体被SAM合并，
    # 不能用它计算视觉伺服误差。
    if bbox_area_ratio > 0.60:
        return False
    if is_shelf_target and y2 > float(config.image_height) - margin:
        return False

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
        and (
            is_table_dual_target
            or (
                config.servo_min_v <= v <= config.servo_max_v
                and abs(v - target_v) <= v_tolerance
            )
        )
    )

    # 进入视觉伺服前必须先拿到完整目标框。贴边或越界框不能
    # 再驱动底盘，否则机器人会追着桌面/墙面等合并区域运动。
    return box_inside and box_large_enough and centroid_inside


KNOWN_COLORS = ("pink", "brown", "yellow", "white")

# Official scene geometry keeps the table height fixed while object XY
# positions are randomized. A clipped RGB-D mask can sample the tabletop and
# produce a near-zero world Z; use this only when the task already says that
# the pick source is the table.
OFFICIAL_TABLE_OBJECT_CENTER_Z = 0.834


def _canonical_object_label(obj):
    """Return the post-processed label without mixing in stale raw labels."""
    return str(
        obj.get("corrected_label")
        or obj.get("label")
        or obj.get("raw_label")
        or ""
    ).lower()


def _resolved_object_color(obj, canonical_label):
    # 视觉节点一旦提供 estimated_color，就以图像颜色估计为准。
    # 即使结果为 unknown，也不能退回使用可能错误的 DINO 文字标签。
    if "estimated_color" in obj:
        estimated_color = str(obj.get("estimated_color") or "").lower()
        return estimated_color if estimated_color in KNOWN_COLORS else ""

    estimated_color = str(obj.get("estimated_color") or "").lower()
    if estimated_color in KNOWN_COLORS:
        return estimated_color

    for candidate in KNOWN_COLORS:
        if candidate in canonical_label:
            return candidate

    return ""


def _object_matches_task(obj, target, *, allow_partial_bbox=False):
    label = _canonical_object_label(obj)
    target_label = str(target.get("label") or "").lower()
    color = str(target.get("color") or "").lower()
    category = str(target.get("category") or "").lower()

    if not color:
        for candidate in KNOWN_COLORS:
            if candidate in target_label:
                color = candidate
                break

    if not category:
        if any(word in target_label for word in ("box", "cube", "cuboid", "block")):
            category = "box"
        elif "cylinder" in target_label:
            category = "cylinder"

    resolved_color = _resolved_object_color(obj, label)

    if label in ("table", "shelf"):
        return -999.0

    if color:
        if resolved_color and resolved_color != color:
            return -999.0
        if not resolved_color:
            return -999.0

    # 桌面抓取目标必须是完整可见的独立物体。贴边或异常大的框通常
    # 是叠放箱体与桌面/相邻物体被SAM合并，不能让它替换上一帧目标。
    box = obj.get("box_xyxy") or []
    if (
        len(box) >= 4
        and _canonical_support_surface(target) == "table"
        and not allow_partial_bbox
    ):
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        margin = 20.0
        area_ratio = (
            max(0.0, x2 - x1)
            * max(0.0, y2 - y1)
            / (640.0 * 480.0)
        )
        if area_ratio > 0.60:
            return -999.0
        if (
            x1 < margin
            or y1 < margin
            or x2 > 640.0 - margin
            or y2 > 480.0 - margin
        ):
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
    expected_surface = _canonical_support_surface(target)
    detected_surface = _canonical_support_surface(detection)
    resolved_surface = detection.get("support_surface")
    resolved_on_table = detection.get("on_table")
    resolved_on_shelf = detection.get("on_shelf")
    pose_world = detection.get("pose_world")

    # The task contract is the stronger source-location cue. Keep a table
    # candidate as a table target when its XY/depth estimate is otherwise
    # usable; do not let one bad Z sample redirect the robot to another area.
    if expected_surface == "table" and detected_surface != "shelf":
        resolved_surface = "table"
        resolved_on_table = True
        resolved_on_shelf = False
        if isinstance(pose_world, dict):
            pose_world = dict(pose_world)
            z = float(pose_world.get("z", 0.0))
            if z < 0.55 or z > 1.35:
                pose_world["z"] = OFFICIAL_TABLE_OBJECT_CENTER_Z
    elif expected_surface == "shelf" and detected_surface != "table":
        resolved_surface = "shelf"
        resolved_on_table = False
        resolved_on_shelf = True
        # 近距离低层货架的 SAM 掩膜常会包含板面，导致深度中位数
        # 落到 z≈0。任务合同已经明确目标来自货架，且检测的 XY
        # 仍在货架范围内时，使用已推断出的层板高度恢复箱体中心 Z。
        # 只修复明显无效的高度，不覆盖正常的深度结果，也不影响桌面任务。
        if isinstance(pose_world, dict):
            raw_z = float(pose_world.get("z", 0.0))
            shelf_surface_z = detection.get("shelf_surface_z")
            if (
                shelf_surface_z is not None
                and raw_z < 0.35
            ):
                pose_world = dict(pose_world)
                pose_world["z"] = (
                    float(shelf_surface_z)
                    + 0.095
                    + 0.010
                )

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
            "color_consistent": detection.get("color_consistent"),
            "color_scores": detection.get("color_scores"),
            "pose_world": pose_world,
            "size_3d": detection.get("size_3d"),
            "support_surface": resolved_surface,
            "on_table": resolved_on_table,
            "on_shelf": resolved_on_shelf,
            "shelf_layer": detection.get("shelf_layer"),
            "support_surface_index": detection.get(
                "support_surface_index"
            ),
            "shelf_layer_confidence": detection.get(
                "shelf_layer_confidence"
            ),
            "shelf_memory_fused": bool(
                detection.get("shelf_memory_fused")
            ),
            "confidence": float(score),
            "requires_reobserve": False,
            "yaw_world_rad": detection.get("yaw_world_rad"),
            "position_std_m": detection.get("position_std_m"),
            "yaw_std_rad": detection.get("yaw_std_rad"),
            "source_cameras": detection.get("source_cameras"),
            "observed_at": detection.get("observed_at"),
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
    target_label = str(target.get("label") or "").lower()
    task_threshold = float(
        getattr(
            config,
            "search_target_min_dino_score_task",
            config.search_target_min_dino_score,
        )
    )
    threshold = (
        task_threshold
        if target_label and target.get("color")
        else float(config.search_target_min_dino_score)
    )
    if dino_score < threshold:
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
    bbox_area_ratio = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
        / float(config.image_width * config.image_height)
    )
    if bbox_area_ratio > 0.65:
        return False, "bbox_too_large_for_search"

    # 在桌面搜索位看到的货架目标经常只露出画面底边一小块，
    # 深度反投影会把它误算成桌面附近目标。桌面候选必须完整落入画面，
    # 否则继续搜索下一个区域，避免把货架物体当成桌面物体去抓。
    if expected_surface == "table" and not bool(
        getattr(config, "search_allow_partial_table_bbox", False)
    ):
        margin = float(config.servo_bbox_margin_px)
        if (
            x1 < margin
            or y1 < margin
            or x2 > float(config.image_width) - margin
            or y2 > float(config.image_height) - margin
        ):
            return False, "table_candidate_bbox_clipped"

    # 搜索阶段允许目标贴边：这里只用Pose3D生成较远的预抓取点。
    # 到达后target_is_visible_for_servo()仍会严格要求完整检测框。
    if expected_surface == "shelf" and target.get("shelf_memory_fused"):
        # 低层货架目标可能只露出画面边缘，但其世界坐标已经由同一
        # 检测消息中的 shelf_object_levels 校正。允许用这个小框触发
        # 货架预站位；到站后的视觉伺服仍要求完整目标框。
        if not (
            x2 > 0.0
            and y2 > 0.0
            and x1 < float(config.image_width)
            and y1 < float(config.image_height)
            and x2 - x1 >= 12.0
            and y2 - y1 >= 12.0
        ):
            return False, "bbox_not_search_usable"
    elif not (
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
    # Keep the task contract (label/color/category and expected source area)
    # when the current camera frame has no matching detection.  Only the
    # frame-specific geometry is stale.  Clearing support_surface here makes
    # a known table task look location-unknown and can incorrectly send the
    # robot to shelf_front.
    for key in (
        "object_id",
        "box_xyxy",
        "centroid_uv",
        "mask_area",
        "pose_world",
        "size_3d",
        "shelf_layer",
        "shelf_layer_confidence",
    ):
        target[key] = None

    source_surface = _canonical_support_surface(target)
    if source_surface == "table":
        target["support_surface"] = "table"
        target["source_location"] = "table"
        target["on_table"] = True
        target["on_shelf"] = False
    elif source_surface == "shelf":
        target["support_surface"] = "shelf"
        target["source_location"] = "shelf"
        target["on_table"] = False
        target["on_shelf"] = True

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


def bind_task_target_from_objects(scene, task, *, allow_partial_bbox=False):
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

            existing_score = _object_matches_task(
                obj,
                target,
                allow_partial_bbox=allow_partial_bbox,
            )
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
        key=lambda obj: _object_matches_task(
            obj,
            target,
            allow_partial_bbox=allow_partial_bbox,
        ),
    )
    best_score = _object_matches_task(
        best,
        target,
        allow_partial_bbox=allow_partial_bbox,
    )

    if best_score < 1.0:
        task["target"] = _clear_stale_target(target)
        return task

    task["target"] = _copy_detection_to_target(
        target,
        best,
        best_score,
    )
    return task
