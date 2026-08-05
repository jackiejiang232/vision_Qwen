import ast
import json
import re

from .schemas import (
    DinoQuery,
    VLMSceneUnderstanding,
)


def validate_model(model_class, payload):
    if hasattr(model_class, "model_validate"):
        return model_class.model_validate(payload)

    return model_class.parse_obj(payload)


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


SURFACE_FIELDS = (
    "support_surface",
    "on_table",
    "on_shelf",
    "support_surface_index",
    "shelf_layer",
    "shelf_surface_z",
    "shelf_layer_confidence",
    "table_height_confidence",
    "pose_world",
    "size_3d",
)


def _surface_fields(detection):
    detection = detection or {}
    return {
        field: detection.get(field)
        for field in SURFACE_FIELDS
    }


def _task_surface_fields(detection):
    detection = detection or {}
    return {
        "support_surface": detection.get("support_surface"),
        "on_table": detection.get("on_table"),
        "on_shelf": detection.get("on_shelf"),
        "shelf_layer": detection.get("shelf_layer"),
        "shelf_layer_confidence": detection.get(
            "shelf_layer_confidence"
        ),
    }


def _location_from_detection(detection):
    support_surface = str(
        (detection or {}).get("support_surface") or ""
    )

    if support_surface in ("table", "shelf"):
        return support_surface

    return "unknown"

def _target_name_from_task(target):
    target = target or {}

    label = target.get("label")
    if label:
        return label

    color = target.get("color")
    category = target.get("category") or "object"

    if color:
        return f"{color} {category}"

    return category


def build_scene_summary_from_task_queue(
    detections,
    task_queue,
):
    detection_count = len(detections or [])

    if not task_queue:
        return (
            f"检测到{detection_count}个候选物体。"
            "当前没有有效active_task。"
        )

    active_task = task_queue[0]
    active_task_id = active_task.get("task_id")
    active_target = active_task.get("target") or {}
    active_target_name = _target_name_from_task(
        active_target
    )

    if active_target.get("object_id"):
        active_sentence = (
            f"当前active_task为任务{active_task_id}，"
            f"目标{active_target_name}已检测到。"
        )
    else:
        active_sentence = (
            f"当前active_task为任务{active_task_id}，"
            f"目标{active_target_name}未检测到，"
            "需要重新观察。"
        )

    other_sentences = []

    for task in task_queue[1:]:
        task_id = task.get("task_id")
        target = task.get("target") or {}
        target_name = _target_name_from_task(target)

        if target.get("object_id"):
            other_sentences.append(
                f"{target_name}为任务{task_id}目标，已检测到"
            )
        else:
            other_sentences.append(
                f"任务{task_id}目标{target_name}未检测到"
            )

    summary = (
        f"检测到{detection_count}个候选物体。"
        f"{active_sentence}"
    )

    if other_sentences:
        summary += "；" + "；".join(other_sentences) + "。"

    return summary

def align_object_roles_with_task_queue(objects, task_queue):
    if not task_queue:
        return objects

    active_target_id = (
        (task_queue[0].get("target") or {}).get("object_id")
    )

    future_target_ids = {
        (task.get("target") or {}).get("object_id")
        for task in task_queue[1:]
    }
    future_target_ids.discard(None)

    for item in objects:
        object_id = item.get("object_id")

        if active_target_id and object_id == active_target_id:
            item["semantic_role"] = "target_object"
        elif object_id in future_target_ids:
            item["semantic_role"] = "future_task_target"
        else:
            item["semantic_role"] = "context_object"

    return objects

def extract_json_object(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError("Qwen输出中没有JSON")

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(candidate)
    except Exception as error:
        raise ValueError(
            f"Qwen输出不是合法JSON: {error}"
        ) from error
# def validate_dino_query_output(raw_text):
#     result = validate_model(
#         DinoQuery,
#         extract_json_object(raw_text),
#     )
#
#     prompt = result.grounding_prompt.strip()
#     if not prompt:
#         raise ValueError("grounding_prompt为空")
#
#     if not prompt.endswith("."):
#         prompt += " ."
#
#     result.grounding_prompt = prompt
#     return model_to_dict(result)
def _normalize_grounding_prompt(prompt):
    prompt = (prompt or "").strip()

    if not prompt:
        raise ValueError("grounding_prompt为空")

    parts = [
        item.strip().lower()
        for item in prompt.split(".")
        if item.strip()
    ]

    cleaned = []

    for item in parts:
        item = item.replace("方块", "box")
        item = item.replace("箱子", "box")
        item = item.replace("正方体", "cube")
        item = item.replace("长方体", "cuboid")
        item = item.replace("圆柱", "cylinder")
        item = item.replace("货架", "shelf")
        item = item.replace("桌子", "table")

        if item not in cleaned:
            cleaned.append(item)

    if not cleaned:
        raise ValueError("grounding_prompt没有有效短语")

    return " . ".join(cleaned) + " ."


def _extend_prompt_from_query(result, parts):
    def add_phrase(phrase):
        phrase = str(phrase or "").strip().lower()
        if phrase and phrase not in parts:
            parts.append(phrase)

    for task in getattr(result, "tasks", []) or []:
        if task.target_object:
            for phrase in task.target_object.query_phrases:
                add_phrase(phrase)

        if task.reference_object:
            for phrase in task.reference_object.query_phrases:
                add_phrase(phrase)

        for item in task.context_objects:
            add_phrase(item)

    if result.target_object:
        for phrase in result.target_object.query_phrases:
            add_phrase(phrase)

    if result.reference_object:
        for phrase in result.reference_object.query_phrases:
            add_phrase(phrase)

    for item in result.context_objects:
        add_phrase(str(item))

    return parts


def validate_dino_query_output(raw_text, instruction_text=""):
    result = validate_model(
        DinoQuery,
        extract_json_object(raw_text),
    )

    prompt = _normalize_grounding_prompt(
        result.grounding_prompt
    )

    parts = [
        item.strip()
        for item in prompt.split(".")
        if item.strip()
    ]

    parts = _extend_prompt_from_query(result, parts)

    result.grounding_prompt = (
        " . ".join(parts) + " ."
    )

    return model_to_dict(result)

def enrich_scene_geometry(
    result,
    detection_payload,
):
    detection_by_id = {
        item["object_id"]: item
        for item in detection_payload.get(
            "detections",
            [],
        )
    }

    for scene_object in result.objects:
        detection = detection_by_id.get(
            scene_object.object_id
        )

        if detection is None:
            continue

        scene_object.box_xyxy = detection.get(
            "box_xyxy"
        )
        scene_object.centroid_uv = detection.get(
            "centroid_uv"
        )
        scene_object.mask_area = detection.get(
            "mask_area"
        )
        scene_object.dino_score = detection.get(
            "dino_score"
        )
        scene_object.sam_score = detection.get(
            "sam_score"
        )

        for field, value in _surface_fields(detection).items():
            setattr(scene_object, field, value)

        location = _location_from_detection(detection)
        if location != "unknown":
            scene_object.location = location

    selected_id = result.grounding.selected_object_id

    if selected_id and selected_id in detection_by_id:
        result.future_action.target_object_id = selected_id

    return result

def build_scene_from_decision(
    decision_payload,
    detection_payload,
    instruction_text="",
    query_payload=None,
):
    source_stamp = detection_payload.get("source_stamp") or {}
    detections = detection_payload.get("detections", [])

    detection_by_id = {
        item.get("object_id"): item
        for item in detections
    }

    selected_id = decision_payload.get("selected_object_id")
    if selected_id not in detection_by_id:
        selected_id = None

    selected = (
        detection_by_id.get(selected_id)
        if selected_id
        else None
    )

    objects = []

    for detection in detections[:8]:
        object_id = detection.get("object_id")
        label = (
            detection.get("corrected_label")
            or detection.get("label")
            or "unknown"
        )
        label_lower = str(label).lower()

        attributes = []
        for item in ("pink", "brown", "yellow", "white"):
            if item in label_lower:
                attributes.append(item)

        for item in ("box", "cube", "cuboid", "block", "cylinder"):
            if item in label_lower:
                attributes.append(item)

        confidence = max(
            float(detection.get("dino_score") or 0.0),
            float(detection.get("sam_score") or 0.0),
        )

        objects.append(
            {
                "object_id": object_id,
                "label": label,
                "raw_label": detection.get("label"),
                "corrected_label": detection.get("corrected_label"),
                "estimated_color": detection.get("estimated_color"),
                "semantic_role": (
                    "target_object"
                    if object_id == selected_id
                    else "context_object"
                ),
                "location": _location_from_detection(detection),
                "attributes": attributes,
                "relations": [],
                "confidence": confidence,
                "box_xyxy": detection.get("box_xyxy"),
                "centroid_uv": detection.get("centroid_uv"),
                "mask_area": detection.get("mask_area"),
                "dino_score": detection.get("dino_score"),
                "sam_score": detection.get("sam_score"),
                **_surface_fields(detection),
            }
        )

    color, category = _infer_target_from_query_or_instruction(
        instruction_text,
        query_payload=query_payload,
    )

    target_object = {}
    destination = {}
    reference_object = {}

    if query_payload:
        target_object = query_payload.get("target_object") or {}
        destination = query_payload.get("destination") or {}
        reference_object = query_payload.get("reference_object") or {}

    instruction_items = []

    for task in (query_payload or {}).get("tasks", []):
        task_target = task.get("target_object") or {}
        task_destination = task.get("destination") or {}
        task_reference = task.get("reference_object") or {}

        reference_label = (
            " ".join(
                item for item in [
                    task_reference.get("color"),
                    task_reference.get("category"),
                ]
                if item
            )
            or None
        )

        instruction_items.append(
            {
                "task_id": task.get("task_id"),
                "target_category": (
                    task_target.get("category")
                    or "object"
                ),
                "target_color": task_target.get("color"),
                "source_location": None,
                "destination_type": task_destination.get("type"),
                "reference_object": reference_label,
                "spatial_relation": task_destination.get(
                    "spatial_relation"
                ),
                "original_instruction": task.get(
                    "original_instruction",
                    "",
                ),
            }
        )

    if not instruction_items:
        instruction_items = [
            {
                "task_id": 1,
                "target_category": (
                    target_object.get("category")
                    or category
                    or "object"
                ),
                "target_color": (
                    target_object.get("color")
                    or color
                ),
                "source_location": None,
                "destination_type": destination.get("type"),
                "reference_object": (
                    " ".join(
                        item for item in [
                            reference_object.get("color"),
                            reference_object.get("category"),
                        ]
                        if item
                    )
                    or None
                ),
                "spatial_relation": destination.get(
                    "spatial_relation"
                ),
                "original_instruction": instruction_text or "",
            }
        ]

    task_queue = build_task_queue_from_query(
        query_payload=query_payload,
        detection_payload=detection_payload,
    )

    active_task_id = (
        task_queue[0]["task_id"]
        if task_queue
        else None
)
    active_task = task_queue[0] if task_queue else None

    active_target = (
        active_task.get("target")
        if active_task
        else {}
    )

    grounding_selected_id = active_target.get("object_id")
    grounding_selected_label = active_target.get("label")

    grounding_requires_reobserve = bool(
        active_target.get("requires_reobserve")
    )
    objects = align_object_roles_with_task_queue(
        objects=objects,
        task_queue=task_queue,
)
    scene_summary = build_scene_summary_from_task_queue(
        detections=detections,
        task_queue=task_queue,
)
    
    payload = {
        "schema_version": "1.0",
        "source_stamp_sec": int(source_stamp.get("sec") or 0),
        "source_stamp_nanosec": int(source_stamp.get("nanosec") or 0),
        "scene_summary": scene_summary,
        "instruction_understanding": instruction_items,
        "objects": objects,
        
        "grounding": {
            "selected_object_id": grounding_selected_id,
            "selected_label": grounding_selected_label,
            "reason": (
                "顶层grounding已对齐active_task.target，"
                "参考物不会作为抓取目标。"
            ),
            "confidence": float(
                active_target.get("confidence") or 0.0
            ),
            "requires_reobserve": grounding_requires_reobserve,
        },
        "uncertainties": [],
        "future_action": {
            "interface_version": "1.0",
            "enabled": False,
            "target_object_id": grounding_selected_id,
            "skills": [],
        },
        "task_queue": task_queue,
        "active_task_id": active_task_id,
        "execution_policy": {
            "order": "ascending_task_id",
            "allow_parallel": False,
            "require_action_feedback": True,
        },
        }
    

    return validate_model(
        VLMSceneUnderstanding,
        payload,
    )

def validate_vlm_output(
    raw_text,
    detection_payload,
    instruction_text="",
    query_payload=None,
):
    payload = extract_json_object(raw_text)

    if "objects" not in payload:
        return build_scene_from_decision(
            decision_payload=payload,
            detection_payload=detection_payload,
            instruction_text=instruction_text,
            query_payload=query_payload,
        )

    result = validate_model(
        VLMSceneUnderstanding,
        payload,
    )

    allowed_ids = {
        item["object_id"]
        for item in detection_payload.get(
            "detections",
            [],
        )
    }

    for item in result.objects:
        if item.object_id not in allowed_ids:
            raise ValueError(f"非法object_id: {item.object_id}")

    selected_id = result.grounding.selected_object_id
    if selected_id is not None and selected_id not in allowed_ids:
        raise ValueError(f"目标不存在: {selected_id}")

    if result.future_action.enabled:
        raise ValueError("当前禁止动作输出")

    if result.future_action.skills:
        raise ValueError("当前skills必须为空")

    return enrich_scene_geometry(
        result,
        detection_payload,
    )


def _infer_target_from_instruction(instruction_text):
    color = None
    category = None

    if "粉" in instruction_text or "pink" in instruction_text.lower():
        color = "pink"
    elif "棕" in instruction_text or "brown" in instruction_text.lower():
        color = "brown"
    elif "黄" in instruction_text or "yellow" in instruction_text.lower():
        color = "yellow"
    elif "白" in instruction_text or "white" in instruction_text.lower():
        color = "white"

    if "长方体" in instruction_text or "cuboid" in instruction_text.lower():
        category = "cuboid"
    elif "正方体" in instruction_text or "cube" in instruction_text.lower():
        category = "cube"
    elif "方块" in instruction_text or "箱" in instruction_text:
        category = "box"
    elif "圆柱" in instruction_text or "cylinder" in instruction_text.lower():
        category = "cylinder"
    else:
        category = "object"

    return color, category

def _infer_target_from_query_or_instruction(
    instruction_text,
    query_payload=None,
):
    if query_payload:
        target = query_payload.get("target_object") or {}

        color = target.get("color")
        category = target.get("category")

        if color or category:
            return color, category

        old_target = query_payload.get("target") or {}

        color = old_target.get("color")
        category = old_target.get("category")

        if color or category:
            return color, category

    return _infer_target_from_instruction(
        instruction_text or ""
    )

def _score_detection_for_instruction(detection, color, category):
    label = str(detection.get("label", "")).lower()
    score = float(detection.get("dino_score") or 0.0)

    if color and color in label:
        score += 1.0

    if category and category in label:
        score += 0.6

    if category in ("box", "cuboid", "cube") and any(
            word in label for word in ("box", "block", "cube", "cuboid")
    ):
        score += 0.5

    return score

def _score_detection_for_query(detection, query_object):
    if not query_object:
        return 0.0

    label = str(
        detection.get("corrected_label")
        or detection.get("label")
        or ""
    ).lower()

    estimated_color = str(
        detection.get("estimated_color") or ""
    ).lower()

    dino_score = float(detection.get("dino_score") or 0.0)
    sam_score = float(detection.get("sam_score") or 0.0)

    color = str(query_object.get("color") or "").lower()
    category = str(query_object.get("category") or "").lower()
    phrases = [
        str(item).lower()
        for item in query_object.get("query_phrases", [])
    ]

    if color:
        detection_color = estimated_color
        label_color_match = color in label

        if detection_color:
            if detection_color != color:
                return -999.0
        elif not label_color_match:
            return -999.0
    strict_categories = ("cube", "cuboid", "cylinder")
    if category in strict_categories:
        if category not in label:
            return -999.0
    if category == "cuboid" and "cylinder" in label:
        return -999.0

    if category == "cylinder" and "cuboid" in label:
        return -999.0

    if category == "cube" and "cuboid" in label:
        return -999.0
    score = dino_score + 0.3 * sam_score

    if color and color in label:
        score += 1.5

    if color and color == estimated_color:
        score += 1.0

    if category and category in label:
        score += 1.0

    for phrase in phrases:
        if phrase and phrase in label:
            score += 2.0

    if category in ("box", "cube", "cuboid") and any(
        word in label
        for word in ("box", "cube", "cuboid", "block")
    ):
        score += 0.6

    return score


def _select_detection_for_query(detections, query_object, used_ids=None):
    used_ids = used_ids or set()
    candidates = [
        item for item in detections
        if item.get("object_id") not in used_ids
    ]

    if not candidates:
        return None, 0.0

    best = max(
        candidates,
        key=lambda item: _score_detection_for_query(
            item,
            query_object,
        ),
    )
    score = _score_detection_for_query(best, query_object)

    if score < 0:
        return None, score

    if score < 1.0:
        return None, score

    return best, score

def build_fallback_vlm_output(
    instruction_text,
    detection_payload,
    reason="qwen_output_invalid",
    query_payload=None,
):
    source_stamp = detection_payload.get("source_stamp") or {}
    detections = detection_payload.get("detections", [])
    color, category = _infer_target_from_query_or_instruction(
        instruction_text or "",
        query_payload=query_payload,
    )
    destination = {}
    reference_object = {}

    if query_payload:
        destination = query_payload.get("destination") or {}
        reference_object = query_payload.get("reference_object") or {}

    reference_label = (
            " ".join(
                item for item in [
                    reference_object.get("color"),
                    reference_object.get("category"),
                ]
                if item
            )
            or None
    )
    selected = None
    if detections:
        selected = max(
            detections,
            key=lambda item: _score_detection_for_instruction(
                item,
                color,
                category,
            ),
        )

    selected_id = selected.get("object_id") if selected else None
    selected_label = (
    selected.get("corrected_label")
    or selected.get("label")
) if selected else None

    objects = []
    for detection in detections[:8]:
        object_id = detection.get("object_id")
        label = (
            detection.get("corrected_label")
            or detection.get("label")
            or "unknown"
        )
        attributes = []

        label_lower = str(label).lower()
        for item in ("pink", "brown", "yellow", "white"):
            if item in label_lower:
                attributes.append(item)

        for item in ("box", "cube", "cuboid", "block", "cylinder"):
            if item in label_lower:
                attributes.append(item)

        confidence = max(
            float(detection.get("dino_score") or 0.0),
            float(detection.get("sam_score") or 0.0),
        )

        objects.append(
            {
                "object_id": object_id,
                "label": label,
                "raw_label": detection.get("label"),
                "corrected_label": detection.get("corrected_label"),
                "estimated_color": detection.get("estimated_color"),
                "semantic_role": (
                    "target_object"
                    if object_id == selected_id
                    else "context_object"
                ),
                "location": _location_from_detection(detection),
                "attributes": attributes,
                "relations": [],
                "confidence": confidence,
                "box_xyxy": detection.get("box_xyxy"),
                "centroid_uv": detection.get("centroid_uv"),
                "mask_area": detection.get("mask_area"),
                "dino_score": detection.get("dino_score"),
                "sam_score": detection.get("sam_score"),
                **_surface_fields(detection),
            }
        )
    task_queue = build_task_queue_from_query(
        query_payload=query_payload,
        detection_payload=detection_payload,
    )
    active_task_id = (
        task_queue[0]["task_id"]
        if task_queue
        else 1
)
    objects = align_object_roles_with_task_queue(
        objects=objects,
        task_queue=task_queue,
)
    scene_summary = build_scene_summary_from_task_queue(
        detections=detections,
        task_queue=task_queue,
    )
    payload = {
        "schema_version": "1.0",
        "source_stamp_sec": int(source_stamp.get("sec") or 0),
        "source_stamp_nanosec": int(source_stamp.get("nanosec") or 0),
        "scene_summary": scene_summary,
        "instruction_understanding": [
            {
                "task_id": 1,
                "target_category": category or "object",
                "target_color": color,
                "source_location": None,
                "destination_type": destination.get("type"),
                "reference_object": reference_label,
                "spatial_relation": destination.get("spatial_relation"),
                "original_instruction": instruction_text or "",
            }
        ],
        "objects": objects,
        "grounding": {
            "selected_object_id": selected_id,
            "selected_label": selected_label,
            "reason": (
                "Qwen输出未通过JSON校验，已根据任务颜色、类别和检测置信度选择候选目标。"
            ),
            "confidence": float(
                selected.get("dino_score") or 0.0
            ) if selected else 0.0,
            "requires_reobserve": selected is None,
        },
        "uncertainties": [reason],
        "future_action": {
            "interface_version": "1.0",
            "enabled": False,
            "target_object_id": selected_id,
            "skills": [],
        },
        "task_queue": task_queue,
         "active_task_id": active_task_id,
         "execution_policy": {
             "order": "ascending_task_id",
             "allow_parallel": False,
             "require_action_feedback": True,
        },
    }

    return validate_model(
        VLMSceneUnderstanding,
        payload,
    )
def build_task_queue_from_query(
    query_payload,
    detection_payload,
):
    detections = detection_payload.get("detections", [])
    scene_memory = detection_payload.get("scene_memory") or {}
    table_original_poses = scene_memory.get("table_original_poses") or {}
    tasks = []

    if query_payload:
        tasks = query_payload.get("tasks") or []

    if not tasks and query_payload:
        tasks = [
            {
                "task_id": 1,
                "original_instruction": query_payload.get(
                    "original_instruction",
                    "",
                ),
                "target_object": query_payload.get(
                    "target_object"
                ),
                "destination": query_payload.get(
                    "destination"
                ),
                "reference_object": query_payload.get(
                    "reference_object"
                ),
                "context_objects": query_payload.get(
                    "context_objects",
                    [],
                ),
            }
        ]

    task_queue = []
    used_target_ids = set()

    for task in sorted(
        tasks,
        key=lambda item: int(item.get("task_id") or 999),
    ):
        task_id = int(task.get("task_id") or 0)
        target_query = task.get("target_object") or {}
        reference_query = task.get("reference_object") or {}
        destination = task.get("destination") or {}

        target_detection, target_score = _select_detection_for_query(
            detections,
            target_query,
            used_ids=used_target_ids,
        )

        if target_detection:
            used_target_ids.add(target_detection.get("object_id"))

        reference_detection, _ = _select_detection_for_query(
            detections,
            reference_query,
            used_ids=set(),
        )

        target_id = (
            target_detection.get("object_id")
            if target_detection
            else None
        )
        target_label = (
            target_detection.get("label")
            if target_detection
            else None
        )

        reference_id = (
            reference_detection.get("object_id")
            if reference_detection
            else None
        )
        reference_label = (
            reference_detection.get("label")
            if reference_detection
            else None
        )
        memory_pose_world = None

        if destination.get("type") == "original_position":
            expected_reference_label = " ".join(
                item for item in [
                    reference_query.get("color"),
                    reference_query.get("category"),
                ]
                if item
            )

            memory_record = table_original_poses.get(
                expected_reference_label
            )

            if isinstance(memory_record, dict):
                memory_pose_world = memory_record.get("pose_world")
            elif isinstance(memory_record, dict) is False:
                memory_pose_world = memory_record

        uncertainties = []
        if target_id is None:
            uncertainties.append("target_not_detected")
        if reference_query and reference_id is None:
            uncertainties.append("reference_not_detected")

        place_type = destination.get("type")
        spatial_relation = destination.get("spatial_relation")

        task_queue.append(
            {
                "task_id": task_id,
                "status": "pending",
                "original_instruction": task.get(
                    "original_instruction",
                    "",
                ),
                "target": {
                    "object_id": target_id,
                    "label": target_label,
                    "category": target_query.get("category"),
                    "color": target_query.get("color"),
                    "pose_world": (
                        target_detection.get("pose_world")
                        if target_detection
                        else None
                    ),
                    "size_3d": (
                        target_detection.get("size_3d")
                        if target_detection
                        else None
                    ),
                    **_task_surface_fields(target_detection),
                    "confidence": float(target_score or 0.0),
                    "requires_reobserve": target_id is None,
                },
                "place_goal": {
                    "type": place_type,
                    "reference_object_id": reference_id,
                    "reference_label": reference_label,
                    "spatial_relation": spatial_relation,
                    "pose_world": (
                        memory_pose_world
                        or (
                            reference_detection.get("pose_world")
                            if reference_detection
                            else None
                        )
                    ),
                    **_task_surface_fields(reference_detection),
                    "requires_planning": True,
                    "requires_scene_memory": (
                        place_type == "original_position"
                        and memory_pose_world is None
                    ),
                },
                "uncertainties": uncertainties,
            }
        )

    return task_queue