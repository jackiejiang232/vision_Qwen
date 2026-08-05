import json
from typing import Dict, List, Optional

COLOR_MAP = {
    "粉色": "pink",
    "粉": "pink",
    "棕色": "brown",
    "黄褐色": "brown",
    "黄色": "yellow",
    "白色": "white",
}

CATEGORY_MAP = {
    "彩色箱": "box",
    "箱子": "box",
    "正方体": "cube",
    "长方体": "cuboid",
    "货架": "shelf",
    "桌面": "table",
}

LOCATION_MAP = {
    "桌面侧边": "table_side",
    "原桌面位置": "original_table_position",
    "货架空层": "shelf_empty_layer",
    "白色长方体左侧": "left_of_white_cuboid",
    "货架": "shelf",
    "桌面": "table",
}

PROMPT_SYNONYMS = {
    "pink box": [
        "pink box",
        "magenta box",
        "pink cube box",
    ],
    "brown box": [
        "brown box",
        "dark brown box",
        "brown cube box",
    ],
    "yellow box": [
        "yellow box",
        "yellow cube box",
        "yellow container",
    ],
    "white cube": [
        "white cube",
        "white block",
    ],
    "white cuboid": [
        "white cuboid",
        "white rectangular block",
    ],
}

CONTEXT_PROMPTS = [
    "shelf",
    "table",
    "empty shelf layer",
    "white cube",
    "white cuboid",
]


def find_first(text: str, mapping: Dict[str, str]) -> Optional[str]:
    for chinese, english in mapping.items():
        if chinese in text:
            return english
    return None


def parse_locations(instruction: str) -> tuple[Optional[str], Optional[str]]:
    source_location = None
    destination_location = None

    if "桌面侧边" in instruction:
        source_location = "table_side"
    elif "货架" in instruction and "到第一个箱子原桌面位置" in instruction:
        source_location = "shelf"
    elif "白色正方体顶部" in instruction:
        source_location = "top_of_white_cube"

    if "货架空层" in instruction:
        destination_location = "shelf_empty_layer"
    elif "原桌面位置" in instruction:
        destination_location = "original_table_position"
    elif "白色长方体左侧" in instruction:
        destination_location = "left_of_white_cuboid"

    return source_location, destination_location


def normalize_instruction(instruction: str) -> Dict:
    color = find_first(instruction, COLOR_MAP)
    category = find_first(instruction, CATEGORY_MAP)
    source_location, destination_location = parse_locations(
        instruction,
    )

    if category is None:
        category = "box"

    target_label = " ".join(
        item for item in [color, category]
        if item
    )

    if not target_label:
        target_label = "box"

    target_prompts: List[str] = PROMPT_SYNONYMS.get(
        target_label,
        [target_label],
    )

    all_prompts = []
    for item in target_prompts + CONTEXT_PROMPTS:
        if item not in all_prompts:
            all_prompts.append(item)

    grounding_prompt = " . ".join(all_prompts) + " ."

    return {
        "schema_version": "1.0",
        "original_instruction": instruction,
        "target": {
            "category": category,
            "color": color,
            "source_location": source_location,
            "destination_location": destination_location,
        },
        "target_prompts": target_prompts,
        "context_prompts": CONTEXT_PROMPTS,
        "grounding_prompt": grounding_prompt,
    }
def _official_color(task):
    color = task.get("target_color")
    if color in ("pink", "brown", "yellow", "white"):
        return color

    body = str(task.get("target_body") or "")
    if "pink" in body:
        return "pink"
    if "brown" in body:
        return "brown"
    if "yellow" in body:
        return "yellow"
    if "white" in body:
        return "white"

    instruction = str(task.get("instruction") or "")
    if "粉" in instruction:
        return "pink"
    if "褐" in instruction or "棕" in instruction:
        return "brown"
    if "黄" in instruction:
        return "yellow"
    if "白" in instruction:
        return "white"

    return None


def _official_category(task):
    kind = str(task.get("target_kind") or "")
    body = str(task.get("target_body") or "")
    instruction = str(task.get("instruction") or "")

    if "cylinder" in kind or "圆柱" in instruction:
        return "cylinder"

    if "cuboid_box" in kind or body.startswith("box_"):
        return "box"

    if "长方体" in instruction:
        return "cuboid"

    if "正方体" in instruction:
        return "cube"

    if "方块" in instruction or "箱" in instruction:
        return "box"

    return "box"


def _query_phrase(color, category):
    if color:
        return f"{color} {category}"
    return category


def _destination_from_official(task):
    place_type = str(task.get("place_type") or "")
    instruction = str(task.get("instruction") or "")

    if place_type == "shelf_point" or "货架空层" in instruction:
        return {
            "type": "shelf_layer",
            "spatial_relation": "empty_layer",
        }

    if place_type == "table_point" or "原来在桌子上的位置" in instruction:
        return {
            "type": "original_position",
            "spatial_relation": "as_original_position",
        }

    if place_type == "shelf_prop_side":
        relation = "left_of" if task.get("direction") == "left" else "right_of"
        return {
            "type": "relative_position",
            "spatial_relation": relation,
        }

    return {
        "type": "unknown",
        "spatial_relation": None,
    }


def _reference_from_official(task):
    instruction = str(task.get("instruction") or "")

    if "白色长方体" in instruction:
        return {
            "category": "cuboid",
            "color": "white",
            "attributes": [],
            "query_phrases": ["white cuboid"],
        }

    if "白色正方体" in instruction:
        return {
            "category": "cube",
            "color": "white",
            "attributes": [],
            "query_phrases": ["white cube"],
        }

    if "粉色方块" in instruction and "原来在桌子上的位置" in instruction:
        return {
            "category": "box",
            "color": "pink",
            "attributes": [],
            "query_phrases": ["pink box"],
        }

    return None


def normalize_official_task_list(tasks):
    visual_tasks = []
    prompt_parts = []

    original_lines = []

    for item in tasks:
        task_id = int(item.get("task") or len(visual_tasks) + 1)
        instruction = str(item.get("instruction") or "")
        original_lines.append(instruction)

        color = _official_color(item)
        category = _official_category(item)
        phrase = _query_phrase(color, category)

        if phrase not in prompt_parts:
            prompt_parts.append(phrase)

        reference = _reference_from_official(item)
        if reference:
            for ref_phrase in reference.get("query_phrases", []):
                if ref_phrase not in prompt_parts:
                    prompt_parts.append(ref_phrase)

        context_objects = ["table", "shelf"]

        if "货架空层" in instruction:
            context_objects.append("empty shelf layer")

        for context in context_objects:
            if context not in prompt_parts:
                prompt_parts.append(context)

        visual_tasks.append(
            {
                "task_id": task_id,
                "original_instruction": instruction,
                "target_object": {
                    "category": category,
                    "color": color,
                    "attributes": [],
                    "query_phrases": [phrase],
                },
                "destination": _destination_from_official(item),
                "reference_object": reference,
                "context_objects": context_objects,
            }
        )

    for required in (
        "pink box",
        "brown box",
        "yellow box",
        "white cube",
        "white cuboid",
        "shelf",
        "table",
        "empty shelf layer",
    ):
        if required not in prompt_parts:
            prompt_parts.append(required)

    return {
        "schema_version": "1.0",
        "original_instruction": "\n".join(original_lines),
        "tasks": visual_tasks,
        "grounding_prompt": " . ".join(prompt_parts) + " .",
    }


def instruction_to_dino_query_json(instruction):
    if isinstance(instruction, list):
        return json.dumps(
            normalize_official_task_list(instruction),
            ensure_ascii=False,
        )

    return json.dumps(
        normalize_instruction(str(instruction)),
        ensure_ascii=False,
    )