import json
try:
    from .layout_context import (
        build_known_visual_objects,
        build_layout_context,
    )
except ImportError:
    from layout_context import (
        build_known_visual_objects,
        build_layout_context,
    )

def _instruction_to_text(instruction_payload):
    if instruction_payload is None:
        return ""

    if isinstance(instruction_payload, str):
        return instruction_payload.strip()

    if isinstance(instruction_payload, list):
        parts = [_instruction_to_text(item) for item in instruction_payload]
        parts = [item for item in parts if item]
        return chr(10).join(parts).strip()

    if isinstance(instruction_payload, dict):
        for key in (
            "original_instruction",
            "instruction",
            "text",
            "content",
        ):
            value = instruction_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if {
            "target_color",
            "target_body",
            "place_type",
        }.issubset(instruction_payload.keys()):
            target_color = instruction_payload.get("target_color", "")
            target_body = instruction_payload.get("target_body", "")
            place_type = instruction_payload.get("place_type", "")
            return f"抓取{target_color}{target_body}，放到{place_type}".strip()

        return json.dumps(instruction_payload, ensure_ascii=False)

    return str(instruction_payload).strip()


def _summarize_detections(detection_payload, max_items=6):
    detections = detection_payload.get("detections", [])
    summary = []

    for item in detections[:max_items]:
        summary.append(
            {
                # "object_id": item.get("object_id"),
                # "label": item.get("corrected_label") or item.get("label"),
                # "raw_label": item.get("label"),
                # "corrected_label": item.get("corrected_label"),
                # "estimated_color": item.get("estimated_color"),
                # "color_consistent": item.get("color_consistent"),
                # "box_xyxy": item.get("box_xyxy"),
                # "centroid_uv": item.get("centroid_uv"),
                # "mask_area": item.get("mask_area"),
                # "dino_score": item.get("dino_score"),
                # "sam_score": item.get("sam_score"),
                "pose_world": item.get("pose_world"),
                "size_3d": item.get("size_3d"),
                "support_surface": item.get("support_surface"),
                "on_table": item.get("on_table"),
                "on_shelf": item.get("on_shelf"),
                "shelf_layer": item.get("shelf_layer"),
                "contract_ready": item.get("contract_ready"),
            }
        )

    return summary

# INSTRUCTION_TO_DINO_SYSTEM_PROMPT = """
# 你是比赛任务指令解析模块。
#
# 目标：把中文任务转成 GroundingDINO 能直接使用的英文检测短语。
#
# 只输出一个 JSON 对象，格式必须严格如下：
# {"grounding_prompt":"pink box . table . shelf ."}
#
# 规则：
# 1. 只输出 JSON，不要 Markdown，不要代码块，不要解释。
# 2. grounding_prompt 只能使用英文。
# 3. 短语之间用 " . " 分隔，最后以 " ." 结尾。
# 4. 必须包含目标物体，也可以包含 table、shelf、white cube、white cuboid 等上下文。
# 5. 如果有颜色，颜色必须放在物体类别前，例如 pink box。
# """
INSTRUCTION_TO_DINO_SYSTEM_PROMPT = """
你是文旅机器人比赛任务解析模块。

你的唯一任务：
把中文任务指令解析成 GroundingDINO 视觉查询 JSON。

必须解析所有任务，不允许只解析第一条。
如果输入包含任务1、任务2、任务3，输出 tasks 必须包含3项。
只输出 JSON，不要 Markdown，不要解释。

物体命名规则：
1. 粉色方块、棕色方块、黄色方块都是 box，不是 cube，不是 cuboid。
2. 白色圆柱输出 white cylinder。
3. 白色长方体输出 white cuboid。
4. 桌子输出 table。
5. 货架输出 shelf。
6. grounding_prompt 必须由 tasks 中所有 target_object、reference_object、context_objects 组成。

输出 JSON 必须严格包含：
{
  "schema_version": "1.0",
  "original_instruction": "...",
  "tasks": [...],
  "grounding_prompt": "..."
}
"""
# def build_instruction_to_dino_prompt(instruction_text):
#     payload = {
#         "instruction_text": instruction_text,
#         "known_objects": [
#             "pink box",
#             "brown box",
#             "yellow box",
#             "white cube",
#             "white cuboid",
#             "white cylinder",
#             "table",
#             "shelf",
#             "empty shelf layer",
#         ],
#         "examples": [
#             {
#                 "instruction_text": "抓取粉色箱子放到货架空层",
#                 "output": {
#                     "grounding_prompt": "pink box . table . shelf ."
#                 },
#             },
#             {
#                 "instruction_text": "把白色正方体顶部的黄色箱子放到白色长方体左侧",
#                 "output": {
#                     "grounding_prompt": "yellow box . white cube . white cuboid . table ."
#                 },
#             },
#         ],
#         "required_output": {
#             "grounding_prompt": "英文短语 . 英文短语 ."
#         },
#     }
#
#     return json.dumps(
#         payload,
#         ensure_ascii=False,
#         indent=2,
#     )

def build_instruction_to_dino_prompt(instruction_text):
    payload = {
        "instruction_text": instruction_text,
        "task": (
            "请理解这条自然语言任务指令，提取所有任务中需要视觉模块检测的目标物体和参考物体，"
            "并生成 GroundingDINO 英文检测提示词。"
        ),

        "official_layout_context": build_layout_context(),
        "known_visual_objects": [
                "pink box",
                "brown box",
                "yellow box",
                "white cylinder",
                "white cuboid",
                "shelf",
                "table",
            ],

        "output_schema": {
            "schema_version": "1.0",
            "original_instruction": "完整原始任务文本",
            "tasks": [
        {
            "task_id": 1,
            "original_instruction": "单条任务原文",
            "target_object": {
                "category": "box/cube/cuboid/cylinder/object",
                "color": "pink/brown/yellow/white/null",
                "attributes": [],
                "query_phrases": ["pink box"]
            },
            "destination": {
                "type": "table/shelf/shelf_layer/original_position/relative_position/unknown",
                "spatial_relation": "left_of/right_of/on/under/same_layer/as_original_position/unknown/null"
            },
            "reference_object": {
                "category": "box/cuboid/cylinder/null",
                "color": "pink/brown/yellow/white/null",
                "attributes": [],
                "query_phrases": [
                    "white cylinder"
                ]
            },
            "context_objects": ["shelf"],
        }
            ],
     "grounding_prompt": "pink box . brown box . yellow box . white cylinder . white cuboid . shelf . table ."
},
        "examples": [
            {
    "instruction_text": (
        "任务1：抓取粉色方块，放到原白色圆柱所在的货架层\n"
        "任务2：抓取棕色方块，放到粉色方块原来在桌子上的位置\n"
        "任务3：抓取黄色方块，放到货架中白色长方体的左边"
    ),
    "output": {
        "schema_version": "1.0",
        "original_instruction": "完整原始任务文本",
        "tasks": [
            {
                "task_id": 1,
                "original_instruction": "抓取粉色方块，放到原白色圆柱所在的货架层",
                "target_object": {
                    "category": "box",
                    "color": "pink",
                    "attributes": ["block"],
                    "query_phrases": ["pink box"]
                },
                "destination": {
                    "type": "shelf_layer",
                    "spatial_relation": "same_layer"
                },
                "reference_object": {
                    "category": "cylinder",
                    "color": "white",
                    "attributes": [],
                    "query_phrases": ["white cylinder"]
                },
                "context_objects": ["shelf"]
            },
            {
                "task_id": 2,
                "original_instruction": "抓取棕色方块，放到粉色方块原来在桌子上的位置",
                "target_object": {
                    "category": "box",
                    "color": "brown",
                    "attributes": ["block"],
                    "query_phrases": ["brown box"]
                },
                "destination": {
                    "type": "original_position",
                    "spatial_relation": "as_original_position"
                },
                "reference_object": {
                    "category": "box",
                    "color": "pink",
                    "attributes": [],
                    "query_phrases": ["pink box"]
                },
                "context_objects": ["table"]
            },
            {
                "task_id": 3,
                "original_instruction": "抓取黄色方块，放到货架中白色长方体的左边",
                "target_object": {
                    "category": "box",
                    "color": "yellow",
                    "attributes": ["block"],
                    "query_phrases": ["yellow box"]
                },
                "destination": {
                    "type": "relative_position",
                    "spatial_relation": "left_of"
                },
                "reference_object": {
                    "category": "cuboid",
                    "color": "white",
                    "attributes": [],
                    "query_phrases": ["white cuboid"]
                },
                "context_objects": ["shelf"]
            }
        ],
        "grounding_prompt": "pink box . brown box . yellow box . white cylinder . white cuboid . shelf . table ."
    }
}
        ],
        "output_rules": [
            "只输出JSON",
            "不要输出代码块",
            "不要复述解释",
            "grounding_prompt必须包含目标物和参考物",
            "target_object是要抓取/拿取/购买/递给用户的物体",
            "reference_object只是定位参照物，不是要抓取的物体",
            "优先使用 official_layout_context 中存在的物体名称、颜色、类别和位置",
            "grounding_prompt 必须尽量使用 known_visual_objects 中的英文短语",
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )

SYSTEM_PROMPT = """
你是动作目标选择模块。

你的任务：
根据 visual_query 和 detection_summary，选择当前 active_task 真正要抓取的检测目标，并输出给动作模块的最简 JSON。

硬性规则：
1. 只输出 JSON，不要 Markdown，不要解释。
2. 不要输出 scene_summary、objects、future_action、uncertainties。
3. target.object_id 必须来自 detection_summary。
4. target.pose_world 必须直接复制 detection_summary 中同一 object_id 的 pose_world，禁止编造坐标。
5. target.support_surface、target.shelf_layer 必须直接复制 detection_summary。
6. reference_object_id 只能用于放置参考，不能作为 target.object_id。
7. 如果目标没检测到，target.object_id 为 null，requires_reobserve 为 true。
8. white cube、white cuboid、white cylinder 不能互相替代。

只允许输出格式：
{
  "schema_version": "action_hint.v1",
  "active_task_id": 1,
  "target": {
    "object_id": null,
    "label": null,
    "category": null,
    "color": null,
    "pose_world": null,
    "support_surface": null,
    "shelf_layer": null
  },
  "place": {
    "type": null,
    "reference_object_id": null,
    "reference_label": null,
    "spatial_relation": null
  },
  "confidence": 0.0,
  "requires_reobserve": true
}
"""

def build_user_prompt(
    instruction_payload,
    detection_payload,
    query_payload=None,
):
    instruction_text = _instruction_to_text(
        instruction_payload
    )

    payload = {
        "instruction_text": instruction_text,
        "visual_query": query_payload,
        "official_layout_context": build_layout_context(),
        "detection_summary": _summarize_detections(
            detection_payload
        ),
        "source_stamp": detection_payload.get(
            "source_stamp"
        ),
        "task": (
            "请结合 instruction_text、visual_query 和 detection_summary "
            "输出最终场景语义JSON。"
            "visual_query.target_object 是用户真正要操作的目标物体。"
            "visual_query.reference_object 只是空间定位参考物，不能作为 target_object。"
            "必须从 detection_summary 中选择 selected_object_id。"
            "优先选择同时匹配 target_object.color、target_object.category "
            "和 query_phrases 的检测对象。"
            "如果 detection_summary 中没有匹配目标，则 requires_reobserve 必须为 true。"
            "official_layout_context 是官方场景先验，只能辅助理解物体名称、初始位置和放置区域。"
            "最终 selected_object_id 必须来自 detection_summary，不能直接使用 layout 的 world_position 代替视觉检测。"
        ),
        "output_schema": {
            "schema_version": "action_hint.v1",
            "active_task_id": "当前任务id",
            "target": {
                "object_id": "来自detection_summary，未检测到则为null",
                "label": "目标类别标签",
                "category": "box/cube/cuboid/cylinder",
                "color": "pink/brown/yellow/white/null",
                "pose_world": "直接复制目标检测结果pose_world",
                "support_surface": "table/shelf/unknown",
                "shelf_layer": "如果在货架上则为层号，否则null"
            },
            "place": {
                "type": "shelf_layer/original_position/relative_position/table_point/shelf_point",
                "reference_object_id": "参考物object_id或null",
                "reference_label": "参考物label或null",
                "spatial_relation": "same_layer/as_original_position/left_of/right_of/on/null"
            },
            "confidence": 0.0,
            "requires_reobserve": False
        },

        "output_rules": [
            "只输出JSON",
            "不要输出objects数组",
            "不要输出scene_summary",
            "不要输出future_action",
            "不要输出Markdown代码块",
            "pose_world必须复制detection_summary中的值",
            "如果目标没有pose_world，则requires_reobserve=true",
            "reference_object_id不能作为target.object_id"
        ],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
