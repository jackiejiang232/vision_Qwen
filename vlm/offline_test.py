#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image as PILImage

try:
    from .json_parser import validate_vlm_output
    from .qwen_engine import QwenVLEngine
    from .vlm_config import MODEL_PATH
except ImportError:
    from json_parser import validate_vlm_output
    from qwen_engine import QwenVLEngine
    from vlm_config import MODEL_PATH


def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL offline test"
    )

    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Qwen2.5-VL 本地模型路径",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="测试图片路径",
    )

    parser.add_argument(
        "--instruction",
        required=True,
        help="sample_instruction.json 路径",
    )

    parser.add_argument(
        "--detections",
        required=True,
        help="sample_detections.json 路径",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    instruction_path = Path(args.instruction)
    detections_path = Path(args.detections)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if not instruction_path.exists():
        raise FileNotFoundError(f"指令文件不存在: {instruction_path}")

    if not detections_path.exists():
        raise FileNotFoundError(f"检测文件不存在: {detections_path}")

    print(f"加载模型: {args.model_path}")
    engine = QwenVLEngine(args.model_path)

    print(f"读取图片: {image_path}")
    image_pil = PILImage.open(image_path).convert("RGB")

    print(f"读取指令: {instruction_path}")
    instruction_payload = load_json(instruction_path)

    print(f"读取检测结果: {detections_path}")
    detection_payload = load_json(detections_path)

    print("开始 Qwen 推理...")
    raw_text = engine.infer(
        image_pil=image_pil,
        instruction_payload=instruction_payload,
        detection_payload=detection_payload,
    )

    print("\n========== Qwen 原始输出 ==========")
    print(raw_text)

    print("\n========== JSON 校验结果 ==========")
    result = validate_vlm_output(
        raw_text,
        detection_payload,
    )

    print(
        json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()