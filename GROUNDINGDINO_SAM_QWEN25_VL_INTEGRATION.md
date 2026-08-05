# GroundingDINO + SAM + Qwen2.5-VL 融合实施教程

## 1. 范围

本教程在现有GroundingDINO+SAM基础上接入Qwen2.5-VL，完成：

1. 机器人RGB视觉输入。
2. GroundingDINO开放词汇检测。
3. SAM实例分割。
4. 比赛指令和游客自然语言理解。
5. 根据官方随机指令动态生成GroundingDINO检测提示词。
6. 图像、检测结果与语言指令融合。
7. 发布结构化场景语义。
8. 为未来动作规划预留接口。

当前不实现导航、机械臂、抓取和放置，也不允许Qwen发布机器人控制命令。

---

## 2. 总体架构

~~~text
/material/instruction
    |
    v
Qwen2.5-VL 容器或轻量指令解析器
    |
    +-- /vlm/dino_query
    |
    v
GroundingDINO + SAM 容器 <--- 机器人RGB相机
    |
    +-- /grounded_sam/detections
    +-- /grounded_sam/keyframe
    +-- /grounded_sam/annotated
                 |
/material/instruction
                 |
                 v
Qwen2.5-VL 容器
                 |
                 v
/vlm/scene_understanding
                 |
                 v
未来VLA动作规划器，当前不实现
~~~

GroundingDINO+SAM提供对象事实和几何锚点。Qwen提供语义理解、指令目标匹配、场景关系和不确定性判断。程序负责校验，防止Qwen虚构对象和坐标。

随机赛题下，Qwen不能只在检测之后工作。它还要先读取/material/instruction，把官方随机任务指令解析成短的英文检测词，再发布给GroundingDINO。GroundingDINO不直接理解完整中文任务句，只接收Qwen或规则解析器生成的开放词汇prompt。

---

## 3. 为什么拆成两个容器

GroundingDINO当前使用：

~~~text
transformers 4.30.2
tokenizers 0.13.3
~~~

Qwen2.5-VL要求较新的Transformers，建议：

~~~text
transformers >= 4.49
~~~

二者放在同一Python环境会导致GroundingDINO的BertModel接口不兼容。因此必须使用：

~~~text
challengecup-vision:latest
  GroundingDINO + SAM
  Transformers 4.30.2

challengecup-qwen-vlm:latest
  Qwen2.5-VL
  Transformers 4.49或更高
~~~

两个容器都设置：

~~~text
--network host
--ipc host
ROS_DOMAIN_ID=99
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
~~~

不要在Qwen容器安装GroundingDINO，不要在感知容器升级Transformers。

---

## 4. 创建目录

执行：

~~~bash
cd /media/jiangzhenmin/系统/JZM/Vision

mkdir -p docker vlm requirements tests

touch vlm/__init__.py
touch vlm/vlm_config.py
touch vlm/schemas.py
touch vlm/prompt_builder.py
touch vlm/instruction_parser.py
touch vlm/json_parser.py
touch vlm/qwen_engine.py
touch vlm/qwen_vl_node.py
touch vlm/offline_test.py

touch requirements/qwen-vlm.txt
touch docker/Dockerfile.qwen-vlm
touch tests/sample_instruction.json
touch tests/sample_detections.json
~~~

最终结构：

~~~text
Vision/
├── grounded_sam.py
├── docker/
│   └── Dockerfile.qwen-vlm
├── requirements/
│   └── qwen-vlm.txt
├── tests/
│   ├── sample_instruction.json
│   └── sample_detections.json
└── vlm/
    ├── __init__.py
    ├── vlm_config.py
    ├── schemas.py
    ├── prompt_builder.py
    ├── instruction_parser.py
    ├── json_parser.py
    ├── qwen_engine.py
    ├── qwen_vl_node.py
    └── offline_test.py
~~~

---

## 5. ROS话题设计

| 话题 | 类型 | 用途 |
|---|---|---|
| /material/instruction | std_msgs/msg/String | 官方任务指令 |
| /vlm/dino_query | std_msgs/msg/String | Qwen或规则解析器生成的GroundingDINO动态检测词 |
| /grounded_sam/detections | std_msgs/msg/String | 检测JSON |
| /grounded_sam/keyframe | sensor_msgs/msg/Image | 检测对应的无标注原图 |
| /grounded_sam/annotated | sensor_msgs/msg/Image | 人工调试 |
| /vlm/scene_understanding | std_msgs/msg/String | VLM语义输出 |
| /vlm/status | std_msgs/msg/String | 状态和错误 |
| /vla/action_request | 预留 | 当前禁止发布 |

Qwen使用keyframe，不使用annotated图。标注图上的文字、框和Mask颜色可能干扰语义模型。

---

## 5.1 随机赛题指令如何驱动GroundingDINO

原文档的基础链路是：GroundingDINO+SAM先检测，Qwen再结合检测结果和指令做理解。这个链路能完成“根据已有检测结果理解任务”，但对官方随机任务还不够完整。

官方任务随机发布时，必须增加一条前置链路：

~~~text
/material/instruction
    |
    v
指令解析器，优先规则解析，必要时使用Qwen
    |
    v
/vlm/dino_query
    |
    v
GroundingDINO使用动态prompt检测当前任务相关物体
~~~

也就是说，Qwen不是把完整中文任务句直接交给GroundingDINO，而是先把任务句翻译成短、明确、可检测的英文词组。

例如官方指令：

~~~text
把桌面侧边的粉色箱子搬到货架空层
~~~

应该转成：

~~~json
{
  "schema_version": "1.0",
  "original_instruction": "把桌面侧边的粉色箱子搬到货架空层",
  "target": {
    "category": "box",
    "color": "pink",
    "source_location": "table_side",
    "destination_location": "shelf_empty_layer"
  },
  "target_prompts": [
    "pink box",
    "magenta box",
    "pink cube box"
  ],
  "context_prompts": [
    "shelf",
    "table",
    "empty shelf layer",
    "white cube",
    "white cuboid"
  ],
  "grounding_prompt": "pink box . magenta box . pink cube box . shelf . table . empty shelf layer . white cube . white cuboid ."
}
~~~

GroundingDINO真正使用的是`grounding_prompt`字段，不使用整句中文任务。

### 5.1.1 为什么要优先规则解析

比赛任务不是完全开放聊天，而是有限物体、有限颜色、有限搬运关系。规则解析比纯Qwen更稳定。

建议流程：

1. 先用规则从中文指令里提取颜色、类别、起点和终点。
2. 如果规则提取失败，再调用Qwen把指令转成结构化JSON。
3. 程序校验Qwen输出，禁止输出不存在的颜色、类别和地点。
4. 用校验后的结构化结果生成GroundingDINO prompt。

推荐维护固定词表：

~~~python
COLOR_MAP = {
    "粉色": "pink",
    "粉": "pink",
    "棕色": "brown",
    "黄褐色": "brown",
    "黄色": "yellow",
    "白色": "white",
}

CATEGORY_MAP = {
    "箱子": "box",
    "彩色箱": "box",
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
~~~

### 5.1.2 新增vlm/instruction_parser.py

该文件负责把官方指令转成DINO查询JSON。它可以先用规则实现，后续再接Qwen增强。

~~~python
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


def instruction_to_dino_query_json(instruction: str) -> str:
    return json.dumps(
        normalize_instruction(instruction),
        ensure_ascii=False,
    )
~~~

### 5.1.3 Qwen节点收到/material/instruction后先发布/vlm/dino_query

在`vlm/qwen_vl_node.py`中，除了保存`latest_instruction`，还应该发布DINO查询。

新增导入：

~~~python
from std_msgs.msg import String

from .instruction_parser import (
    instruction_to_dino_query_json,
)
~~~

初始化时新增发布器：

~~~python
self.dino_query_publisher = self.create_publisher(
    String,
    "/vlm/dino_query",
    10,
)
~~~

在`instruction_callback`里加入：

~~~python
def instruction_callback(self, msg):
    self.latest_instruction = msg.data

    query_msg = String()
    query_msg.data = instruction_to_dino_query_json(
        msg.data
    )
    self.dino_query_publisher.publish(query_msg)

    self.get_logger().info(
        f"已根据比赛指令生成DINO查询: {query_msg.data}"
    )
~~~

这样每次官方发布随机任务时，GroundingDINO都会收到新的检测词。

### 5.1.4 GroundingDINO节点订阅/vlm/dino_query

在`grounded_sam.py`或`grounded_sam_camera_node.py`中订阅`/vlm/dino_query`，把原来的固定`args.text`改成动态`self.text_prompt`。

新增导入：

~~~python
import json
import threading
from std_msgs.msg import String
~~~

初始化时新增：

~~~python
self.text_prompt = args.text
self.text_lock = threading.Lock()

self.dino_query_subscriber = self.create_subscription(
    String,
    "/vlm/dino_query",
    self.dino_query_callback,
    10,
)
~~~

新增回调：

~~~python
def dino_query_callback(self, msg):
    try:
        payload = json.loads(msg.data)
        prompt = payload.get(
            "grounding_prompt",
            "",
        ).strip()

        if not prompt:
            self.get_logger().warn(
                "收到空的grounding_prompt，保持原检测词"
            )
            return

        if not prompt.endswith("."):
            prompt = prompt + " ."

        with self.text_lock:
            self.text_prompt = prompt

        self.get_logger().info(
            f"GroundingDINO检测词已更新: {prompt}"
        )

    except Exception as error:
        self.get_logger().error(
            f"解析/vlm/dino_query失败: {error}"
        )
~~~

推理时，把原来的：

~~~python
text_prompt=args.text,
~~~

替换成：

~~~python
with self.text_lock:
    text_prompt = self.text_prompt

# 后面predict调用使用text_prompt
~~~

然后在`predict(...)`里使用：

~~~python
caption=text_prompt,
~~~

如果你的函数参数名是`text_prompt`，就写：

~~~python
text_prompt=text_prompt,
~~~

具体按你当前`grounded_sam.py`里的函数定义为准。

### 5.1.5 启动顺序

推荐启动顺序：

1. 启动官方仿真和裁判系统。
2. 启动GroundingDINO+SAM视觉节点，先使用默认全量检测词。
3. 启动Qwen VLM节点。
4. Qwen节点监听`/material/instruction`。
5. 官方发布随机任务。
6. Qwen或规则解析器发布`/vlm/dino_query`。
7. GroundingDINO自动切换检测词。
8. GroundingDINO+SAM发布检测结果。
9. Qwen结合图像、检测结果和原始指令发布`/vlm/scene_understanding`。

### 5.1.6 为什么还要保留默认全量检测词

刚启动时可能还没有收到官方指令。如果GroundingDINO没有默认prompt，就无法先输出调试画面。

建议默认prompt设置为：

~~~text
pink box . brown box . yellow box . white cube . white cuboid . shelf . table .
~~~

收到`/vlm/dino_query`后，再切换成当前任务相关prompt。

---

## 6. 修改 grounded_sam.py

只增加关键帧和对象ID发布，不把Qwen模型写入grounded_sam.py。

### 6.1 增加命令行参数

在parse_args中加入：

~~~python
parser.add_argument(
    "--keyframe-topic",
    default="/grounded_sam/keyframe",
)
~~~

### 6.2 增加发布器

在GroundedSamCameraNode初始化中加入：

~~~python
self.keyframe_publisher = self.create_publisher(
    Image,
    args.keyframe_topic,
    10,
)
~~~

### 6.3 增加发布函数

~~~python
def publish_keyframe(self, image, header):
    message = self.bridge.cv2_to_imgmsg(
        image,
        encoding="bgr8",
    )

    if header is not None:
        message.header = header

    self.keyframe_publisher.publish(message)
~~~

### 6.4 推理完成后发布原图

在发布annotated图之后加入：

~~~python
self.publish_keyframe(
    image=frame,
    header=header,
)
~~~

必须传frame，不能传annotated。

### 6.5 增加object_id

生成records后、发布JSON前加入：

~~~python
if header is not None:
    frame_key = (
        f"{header.stamp.sec}_"
        f"{header.stamp.nanosec}"
    )
else:
    frame_key = str(sequence)

for index, record in enumerate(records):
    record["object_id"] = (
        f"{frame_key}_{index}"
    )
~~~

输出示例：

~~~json
{
  "source_stamp": {
    "sec": 1784705801,
    "nanosec": 774044039
  },
  "detections": [
    {
      "object_id": "1784705801_774044039_0",
      "label": "pink box",
      "dino_score": 0.86,
      "sam_score": 0.94,
      "box_xyxy": [120, 80, 340, 290],
      "mask_area": 18324,
      "centroid_uv": [228.4, 184.7]
    }
  ]
}
~~~

keyframe的Header时间戳必须与source_stamp一致，Qwen节点依靠该字段配对。

---

## 7. Qwen依赖文件

requirements/qwen-vlm.txt写入：

~~~text
transformers>=4.49,<5
accelerate>=1.2,<2
qwen-vl-utils>=0.0.8
pydantic>=2.7,<3
Pillow>=10
~~~

不要加入torch、torchvision、groundingdino和segment_anything。Qwen镜像继承官方镜像的Torch 2.7.1+cu118。

安装成功后执行pip freeze，记录最终可工作版本并固定。

---

## 8. 下载Qwen模型

推荐：

~~~text
Qwen/Qwen2.5-VL-3B-Instruct
~~~

创建目录：

~~~bash
mkdir -p /media/jiangzhenmin/系统/JZM/models/Qwen2.5-VL-3B-Instruct
~~~

使用Hugging Face：

~~~bash
huggingface-cli download \
  Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir /media/jiangzhenmin/系统/JZM/models/Qwen2.5-VL-3B-Instruct
~~~

网络不可用时使用ModelScope：

~~~bash
python3 -m pip install modelscope

modelscope download \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /media/jiangzhenmin/系统/JZM/models/Qwen2.5-VL-3B-Instruct
~~~

比赛时必须从本地加载，不允许运行时下载。

---

## 9. vlm_config.py

写入：

~~~python
MODEL_PATH = "/models/Qwen2.5-VL-3B-Instruct"

KEYFRAME_TOPIC = "/grounded_sam/keyframe"
DETECTIONS_TOPIC = "/grounded_sam/detections"
INSTRUCTION_TOPIC = "/material/instruction"
DINO_QUERY_TOPIC = "/vlm/dino_query"

OUTPUT_TOPIC = "/vlm/scene_understanding"
STATUS_TOPIC = "/vlm/status"

MAX_NEW_TOKENS = 768
IMAGE_CACHE_SIZE = 10
INFERENCE_COOLDOWN_S = 2.0

ACTION_INTERFACE_VERSION = "1.0"
ACTION_ENABLED = False
~~~

当前阶段ACTION_ENABLED必须为False。

---

## 10. schemas.py

使用Pydantic约束输出：

~~~python
from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class InstructionItem(StrictModel):
    task_id: Optional[int] = None
    target_category: str
    target_color: Optional[str] = None
    source_location: Optional[str] = None
    destination_type: Optional[str] = None
    reference_object: Optional[str] = None
    spatial_relation: Optional[str] = None
    original_instruction: str


class DinoQueryTarget(StrictModel):
    category: str
    color: Optional[str] = None
    source_location: Optional[str] = None
    destination_location: Optional[str] = None


class DinoQuery(StrictModel):
    schema_version: str = "1.0"
    original_instruction: str
    target: DinoQueryTarget
    target_prompts: List[str]
    context_prompts: List[str]
    grounding_prompt: str


class SceneObject(StrictModel):
    object_id: str
    label: str
    semantic_role: str
    location: str
    attributes: List[str] = Field(
        default_factory=list
    )
    relations: List[str] = Field(
        default_factory=list
    )
    confidence: float


class GroundingDecision(StrictModel):
    selected_object_id: Optional[str] = None
    selected_label: Optional[str] = None
    reason: str
    confidence: float
    requires_reobserve: bool


class FutureActionSlot(StrictModel):
    interface_version: str = "1.0"
    enabled: bool = False
    target_object_id: Optional[str] = None
    skills: List[str] = Field(
        default_factory=list
    )


class VLMSceneUnderstanding(StrictModel):
    schema_version: str = "1.0"
    source_stamp_sec: int
    source_stamp_nanosec: int
    scene_summary: str
    instruction_understanding: List[
        InstructionItem
    ]
    objects: List[SceneObject]
    grounding: GroundingDecision
    uncertainties: List[str] = Field(
        default_factory=list
    )
    future_action: FutureActionSlot
~~~

必须满足：

1. object_id来自检测结果。
2. Qwen不生成坐标。
3. future_action.enabled为False。
4. skills为空数组。
5. 未定义字段直接拒绝。

---

## 11. prompt_builder.py

写入：

~~~python
import json


SYSTEM_PROMPT = """
你是文旅机器人视觉语言理解模块。

结合机器人相机图像、比赛指令和经过
GroundingDINO+SAM验证的检测结果，
输出结构化场景语义。

规则：
1. 检测列表是对象ID和几何信息的唯一来源。
2. 不得创建不存在的object_id。
3. 不得编造坐标、深度和机器人动作。
4. 理解颜色、类别、来源、目的地和空间关系。
5. 信息不足时requires_reobserve必须为true。
6. future_action.enabled必须为false。
7. future_action.skills必须为空。
8. 只输出合法JSON，不输出解释文字。
"""


def build_user_prompt(
    instruction_payload,
    detection_payload,
):
    payload = {
        "competition_instruction": (
            instruction_payload
        ),
        "perception_result": detection_payload,
        "output_constraints": {
            "schema_version": "1.0",
            "future_action_enabled": False,
            "future_action_skills": [],
        },
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
~~~

Qwen只解释指令，不覆盖指令中的target_color、place_world等事实字段。

---

## 12. json_parser.py

写入：

~~~python
import json

from .schemas import (
    VLMSceneUnderstanding,
)


def extract_json_object(text):
    text = text.strip()
    start = text.find("{")

    if start < 0:
        raise ValueError(
            "Qwen输出中没有JSON"
        )

    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(
        text[start:]
    )

    return payload


def validate_vlm_output(
    raw_text,
    detection_payload,
):
    payload = extract_json_object(raw_text)

    result = (
        VLMSceneUnderstanding
        .model_validate(payload)
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
            raise ValueError(
                f"非法object_id: {item.object_id}"
            )

    selected_id = (
        result.grounding.selected_object_id
    )

    if (
        selected_id is not None
        and selected_id not in allowed_ids
    ):
        raise ValueError(
            f"目标不存在: {selected_id}"
        )

    if result.future_action.enabled:
        raise ValueError(
            "当前禁止动作输出"
        )

    if result.future_action.skills:
        raise ValueError(
            "当前skills必须为空"
        )

    return result
~~~

JSON失败后最多修正一次，不允许无限重试。

---

## 13. qwen_engine.py

写入模型封装：

~~~python
import torch

from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from .prompt_builder import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from .vlm_config import MAX_NEW_TOKENS


class QwenVLEngine:
    def __init__(self, model_path):
        self.processor = (
            AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
        )

    def infer(
        self,
        image_pil,
        instruction_payload,
        detection_payload,
    ):
        prompt = build_user_prompt(
            instruction_payload,
            detection_payload,
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_pil,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        text = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(messages)
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(
            self.model.device
        )

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        trimmed = [
            output[len(input_ids):]
            for input_ids, output
            in zip(
                inputs.input_ids,
                generated,
            )
        ]

        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
~~~

初期使用BF16和SDPA，不启用FlashAttention、量化、视频和多图。

---

## 14. qwen_vl_node.py

节点职责：

1. 订阅keyframe。
2. 订阅detections。
3. 订阅instruction。
4. 按时间戳配对图像和检测。
5. 后台执行Qwen，不能阻塞ROS回调。
6. 校验并发布结果。
7. 发布状态和错误。
8. 不发布动作话题。

核心导入：

~~~python
import argparse
import json
import threading
from collections import OrderedDict

import cv2
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .json_parser import (
    validate_vlm_output,
)
from .qwen_engine import QwenVLEngine
from .vlm_config import *
~~~

时间戳键：

~~~python
def stamp_key(sec, nanosec):
    return (
        f"{int(sec)}:{int(nanosec)}"
    )
~~~

初始化状态：

~~~python
self.bridge = CvBridge()
self.engine = QwenVLEngine(model_path)

self.image_cache = OrderedDict()
self.detection_cache = OrderedDict()
self.latest_instruction = None

self.last_processed_key = None
self.busy = False
self.lock = threading.Lock()
~~~

订阅：

~~~python
self.create_subscription(
    Image,
    KEYFRAME_TOPIC,
    self.image_callback,
    10,
)

self.create_subscription(
    String,
    DETECTIONS_TOPIC,
    self.detection_callback,
    10,
)

self.create_subscription(
    String,
    INSTRUCTION_TOPIC,
    self.instruction_callback,
    10,
)
~~~

发布：

~~~python
self.output_pub = self.create_publisher(
    String,
    OUTPUT_TOPIC,
    10,
)

self.status_pub = self.create_publisher(
    String,
    STATUS_TOPIC,
    10,
)
~~~

图像回调将BGR转换为RGB PIL，并以Header时间戳存入image_cache。检测回调解析source_stamp并存入detection_cache。指令回调保存最新JSON列表。

调度器每0.2秒检查：

~~~python
matched_keys = (
    set(self.image_cache)
    & set(self.detection_cache)
)
~~~

只处理最新匹配时间戳，且不能重复处理last_processed_key。Qwen推理放入单独Worker线程。

Worker核心：

~~~python
raw_text = self.engine.infer(
    image_pil=image,
    instruction_payload=instruction,
    detection_payload=detections,
)

result = validate_vlm_output(
    raw_text,
    detections,
)

message = String()
message.data = result.model_dump_json()

self.output_pub.publish(message)
~~~

实际实现应使用一个固定Worker线程和容量为1的任务队列，避免不断创建线程。新任务到达时替换尚未执行的旧任务，只保留最新场景。

主函数接收：

~~~text
--model-path /models/Qwen2.5-VL-3B-Instruct
~~~

---

## 15. 离线测试

tests/sample_instruction.json：

~~~json
[
  {
    "task": 1,
    "target_color": "pink",
    "target_body": "pink_box",
    "place_type": "shelf_empty_layer",
    "place_world": [-2.55, 0.78, 1.15]
  }
]
~~~

tests/sample_detections.json：

~~~json
{
  "source_stamp": {
    "sec": 1,
    "nanosec": 1
  },
  "detections": [
    {
      "object_id": "1_1_0",
      "label": "pink box",
      "dino_score": 0.86,
      "sam_score": 0.94,
      "box_xyxy": [120, 80, 340, 290],
      "mask_area": 18324,
      "centroid_uv": [228.4, 184.7]
    }
  ]
}
~~~

offline_test.py完成：

1. 读取图片。
2. 读取两个JSON。
3. 初始化QwenVLEngine。
4. 调用infer。
5. 调用validate_vlm_output。
6. 打印格式化JSON。
7. 不初始化ROS。

先通过离线测试再进行ROS联调。

---

## 16. Dockerfile.qwen-vlm

写入：

~~~dockerfile
FROM crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/material_sorting:latest

WORKDIR /opt/qwen_vlm

COPY requirements/qwen-vlm.txt /opt/qwen_vlm/requirements.txt

RUN python3 -m pip install \
    --no-cache-dir \
    -r /opt/qwen_vlm/requirements.txt

COPY vlm /opt/qwen_vlm/vlm

ENV PYTHONPATH=/opt/qwen_vlm
ENV HF_HOME=/LLM/huggingface
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
~~~

不要覆盖官方ENTRYPOINT。

构建：

~~~bash
cd /media/jiangzhenmin/系统/JZM

docker build \
  -f Vision/docker/Dockerfile.qwen-vlm \
  -t challengecup-qwen-vlm:latest \
  .
~~~

验证：

~~~bash
docker run --rm \
  --gpus all \
  challengecup-qwen-vlm:latest \
  python3 -c "
import torch
import transformers
print(torch.__version__)
print(torch.version.cuda)
print(transformers.__version__)
print(torch.cuda.is_available())
"
~~~

---

## 17. 离线运行

~~~bash
docker run --rm -it \
  --gpus all \
  --ipc host \
  -v /media/jiangzhenmin/系统/JZM/LLM/Qwen2.5-VL-3B-Instruct:/LLM/Qwen2.5-VL-3B-Instruct:ro \
  -v /media/jiangzhenmin/系统/JZM/Vision:/workspace/Vision:ro \
  challengecup-qwen-vlm:latest \
  python3 -m vlm.offline_test \
  --model-path /LLM/Qwen2.5-VL-3B-Instruct \
  --image /workspace/Vision/input/person.png \
  --instruction /workspace/Vision/tests/sample_instruction.json \
  --detections /workspace/Vision/tests/sample_detections.json
~~~

通过标准：

1. 本地模型加载成功。
2. 输出是合法JSON。
3. Schema校验通过。
4. 目标ID来自检测输入。
5. 不生成坐标。
6. 动作接口关闭。

---

## 18. ROS联调

启动官方server和GroundedSAM后检查：

~~~bash
ros2 topic hz /grounded_sam/keyframe
ros2 topic hz /grounded_sam/detections
ros2 topic echo /material/instruction --full-length --once
~~~

启动Qwen：

~~~bash
docker run --rm -it \
  --gpus all \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID=99 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -v /media/jiangzhenmin/系统/JZM/models/Qwen2.5-VL-3B-Instruct:/models/Qwen2.5-VL-3B-Instruct:ro \
  challengecup-qwen-vlm:latest \
  python3 -m vlm.qwen_vl_node \
  --model-path /models/Qwen2.5-VL-3B-Instruct
~~~

查看状态：

~~~bash
ros2 topic echo /vlm/status --full-length
~~~

查看结果：

~~~bash
ros2 topic echo \
  /vlm/scene_understanding \
  --field data \
  --full-length
~~~

---

## 19. 期望输出

~~~json
{
  "schema_version": "1.0",
  "source_stamp_sec": 1784705801,
  "source_stamp_nanosec": 774044039,
  "scene_summary": "桌面区域检测到粉色文创商品包装箱。",
  "instruction_understanding": [
    {
      "task_id": 1,
      "target_category": "box",
      "target_color": "pink",
      "source_location": "table_side",
      "destination_type": "shelf_empty_layer",
      "reference_object": null,
      "spatial_relation": null,
      "original_instruction": "..."
    }
  ],
  "objects": [
    {
      "object_id": "1784705801_774044039_0",
      "label": "pink box",
      "semantic_role": "target_product",
      "location": "table_side",
      "attributes": ["pink", "rectangular"],
      "relations": [],
      "confidence": 0.86
    }
  ],
  "grounding": {
    "selected_object_id": "1784705801_774044039_0",
    "selected_label": "pink box",
    "reason": "颜色与任务指令一致。",
    "confidence": 0.86,
    "requires_reobserve": false
  },
  "uncertainties": [],
  "future_action": {
    "interface_version": "1.0",
    "enabled": false,
    "target_object_id": "1784705801_774044039_0",
    "skills": []
  }
}
~~~

---

## 20. 动作算法预留

未来创建：

~~~text
Vision/vla/vla_planner_node.py
~~~

未来订阅：

~~~text
/vlm/scene_understanding
/joint_states
/slamware_ros_sdk_server_node/odom
/referee/taskinfo
~~~

未来发布：

~~~text
/vla/action_request
~~~

预留格式：

~~~json
{
  "interface_version": "1.0",
  "request_id": "uuid",
  "target_object_id": "object_id",
  "skills": [],
  "safety_checked": false
}
~~~

当前Qwen节点禁止发布cmd_vel、机械臂、升降、夹爪和action_request话题。

---

## 21. 性能策略

1. 使用Qwen2.5-VL-3B BF16。
2. 使用SDPA。
3. 不处理每一帧。
4. GroundedSAM约1Hz。
5. Qwen只在新指令或新匹配关键帧时触发。
6. max_new_tokens先设768。
7. 使用nvidia-smi监控显存。
8. 显存不足时优先将SAM ViT-H换成ViT-B。
9. 最后才考虑Qwen 4bit。
10. 初期不启用FlashAttention、多图和视频。

---

## 22. 测试清单

单元测试：

1. 合法JSON通过。
2. 非法object_id被拒绝。
3. 动作接口开启时被拒绝。
4. skills非空时被拒绝。
5. 缺失字段被拒绝。

场景测试：

1. 桌面侧边包装箱。
2. 正方体顶部包装箱。
3. 货架包装箱。
4. 目标与障碍物同时出现。
5. 目标遮挡。
6. 目标不存在。
7. 模糊指令。
8. 指令颜色不存在。

ROS测试：

1. 指令先到。
2. 图像先到。
3. 时间戳不匹配。
4. 推理期间新帧到达。
5. Qwen输出非法JSON。
6. 任一容器重启。
7. 显存不足。

随机seed记录：

- 指令理解正确率。
- 目标ID选择正确率。
- JSON合法率。
- 重新观察判断准确率。
- 推理时间。
- 显存峰值。

---

## 23. 完成标准

1. GroundedSAM发布detections。
2. GroundedSAM发布相同时间戳的keyframe。
3. 两个模型环境独立。
4. Qwen从本地权重加载。
5. Qwen接收图像、检测JSON和指令。
6. 输出通过Schema校验。
7. 目标ID来自检测列表。
8. Qwen不虚构坐标。
9. scene_understanding稳定输出。
10. Qwen不发布机器人控制命令。
11. future_action存在但禁用。
12. 多个随机seed下语义匹配稳定。

完成后系统可表述为：

~~~text
GroundingDINO+SAM提供开放词汇检测、实例分割与几何锚定；
Qwen2.5-VL先把官方随机任务指令解析成GroundingDINO可用的动态检测prompt，
再融合原始视觉、检测结果和自然语言指令，
输出面向文旅商品自主拾取任务的结构化场景语义。
~~~

下一阶段再接入VLA动作规划和机器人技能执行。

