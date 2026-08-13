import json
import re
import time


def _loads_maybe_nested_json(message_data):
    try:
        value = json.loads(message_data)
    except json.JSONDecodeError:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _label_from_instruction(text, keywords):
    for key, label in keywords:
        if key in text:
            return label
    return None


def _place_relation_from_instruction(text, payload):
    relation = payload.get("place_relation") or payload.get("spatial_relation")
    direction = payload.get("direction")
    place_type = payload.get("place_type")

    if relation:
        return str(relation), direction, place_type

    if "货架层" in text or ("货架" in text and "层" in text):
        relation = "same_shelf_level_as"
    elif "原" in text and ("位置" in text or "所在" in text):
        relation = "original_position_of"
    elif "左边" in text or "左侧" in text:
        relation = "left_of"
        direction = direction or "left"
    elif "右边" in text or "右侧" in text:
        relation = "right_of"
        direction = direction or "right"
    elif "前面" in text or "前方" in text:
        relation = "front_of"
        direction = direction or "front"
    elif "后面" in text or "后方" in text:
        relation = "behind"
        direction = direction or "behind"
    else:
        relation = "same_shelf_level_as"

    if place_type is None:
        if relation == "original_position_of" and ("桌" in text or "table" in text.lower()):
            place_type = "table_point"
        elif relation in ("left_of", "right_of", "front_of", "behind"):
            place_type = "shelf_prop_side" if "货架" in text or "shelf" in text.lower() else "object_side"
        elif relation == "same_shelf_level_as" and ("货架" in text or "shelf" in text.lower()):
            place_type = "shelf_layer"
        elif "货架" in text or "shelf" in text.lower():
            place_type = "shelf_point"
        elif "桌" in text or "table" in text.lower():
            place_type = "table_point"

    return relation, direction, place_type


def _box_label_from_color(color):
    color = str(color or "").lower()
    if color in ("pink", "粉色"):
        return "pink box"
    if color in ("brown", "棕色", "褐色"):
        return "brown box"
    if color in ("yellow", "黄色"):
        return "yellow box"
    return None


def _pose_to_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        pose = {
            "x": float(value[0]),
            "y": float(value[1]),
            "z": float(value[2]) if len(value) >= 3 else 0.0,
        }
        if len(value) >= 4:
            pose["yaw"] = float(value[3])
        return pose
    return value


def _label_from_body_or_instruction(body, color, instruction):
    text = " ".join(str(value or "") for value in (body, color, instruction)).lower()
    if "pink" in text or "粉色" in text:
        return "pink box"
    if "brown" in text or "棕色" in text or "褐色" in text:
        return "brown box"
    if "yellow" in text or "黄色" in text:
        return "yellow box"
    if "white cylinder" in text or "白色圆柱" in text:
        return "white cylinder"
    if "packaging" in text or "长方体" in text:
        return "packaging box"
    return None


def _parse_task_payload(payload, original_message):
    if not isinstance(payload, dict):
        payload = {}
    instruction = str(
        payload.get("instruction")
        or payload.get("raw_instruction")
        or original_message
    )
    pickup_clause, _, place_clause = instruction.partition("放到")

    pick_label = _label_from_body_or_instruction(
        payload.get("target_body") or payload.get("target_kind"),
        payload.get("target_color"),
        "",
    ) or _label_from_instruction(
        pickup_clause or instruction,
        (
            ("粉色方块", "pink box"),
            ("粉色盒子", "pink box"),
            ("粉色箱子", "pink box"),
            ("棕色方块", "brown box"),
            ("棕色盒子", "brown box"),
            ("褐色方块", "brown box"),
            ("褐色盒子", "brown box"),
            ("黄色方块", "yellow box"),
            ("黄色盒子", "yellow box"),
            ("pink", "pink box"),
            ("brown", "brown box"),
            ("yellow", "yellow box"),
        ),
    )
    reference_label = _label_from_body_or_instruction(
        payload.get("ref_body") or payload.get("ref_prop_body") or payload.get("ref_prop"),
        payload.get("ref_color"),
        "",
    ) or _label_from_instruction(
        place_clause or instruction,
        (
            ("白色圆柱", "white cylinder"),
            ("white cylinder", "white cylinder"),
            ("粉色方块", "pink box"),
            ("pink box", "pink box"),
            ("白色长方体", "packaging box"),
            ("packaging", "packaging box"),
        ),
    )
    place_relation, direction, place_type = _place_relation_from_instruction(
        instruction,
        payload,
    )

    return {
        "task_id": str(
            payload.get("task_id")
            or payload.get("task")
            or payload.get("id")
            or f"task_{int(time.time())}"
        ),
        "instruction": instruction,
        "action": "pick_and_place",
        "pick_label": payload.get("pick_label") or pick_label,
        "reference_label": payload.get("reference_label") or reference_label,
        "place_relation": place_relation,
        "temporal_reference": payload.get("temporal_reference") or (
            "initial" if place_relation == "original_position_of" else "current"
        ),
        "return_home": bool(payload.get("return_home", True)),
        "place_type": place_type,
        "place_world": _pose_to_dict(payload.get("place_world")),
        "place_radius": payload.get("place_radius"),
        "direction": direction,
    }


def parse_task_command(message_data):
    """
    第一版支持两种输入：
    1. JSON: {"instruction": "..."}
    2. 纯文本: 抓取粉色方块，放到原白色圆柱所在的货架层
    """
    payload = _loads_maybe_nested_json(message_data)
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        payload = {}
    return _parse_task_payload(payload, message_data)


def parse_official_task_commands(message_data):
    payload = _loads_maybe_nested_json(message_data)
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        payload = payload["tasks"]
    if not isinstance(payload, list):
        text = str(message_data).strip()
        matches = list(
            re.finditer(r"(?:^|\s)(?:任务|task)\s*([0-9]+)\s*[:：]\s*", text, re.I)
        )
        if matches:
            tasks = []
            for index, match in enumerate(matches):
                start = match.end()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                instruction = text[start:end].strip()
                if instruction:
                    tasks.append(
                        {
                            "task_id": match.group(1),
                            "instruction": instruction,
                        }
                    )
            if tasks:
                return [
                    _parse_task_payload(item, item["instruction"])
                    for item in tasks
                ]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        task_lines = []
        for line in lines:
            match = re.match(r"^(?:任务|task)\s*([0-9]+)\s*[:：]\s*(.+)$", line, re.I)
            if match:
                task_lines.append(
                    {
                        "task_id": match.group(1),
                        "instruction": match.group(2).strip(),
                    }
                )
        if task_lines:
            return [
                _parse_task_payload(item, item["instruction"])
                for item in task_lines
            ]
        return [parse_task_command(message_data)]
    tasks = [
        _parse_task_payload(item, json.dumps(item, ensure_ascii=False))
        for item in payload
        if isinstance(item, dict)
    ]
    return tasks or [parse_task_command(message_data)]
