#!/usr/bin/env python3

import argparse
import copy
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from .instruction_parser import instruction_to_dino_query_json
    from .json_parser import (
        build_fallback_vlm_output,
        model_to_dict,
        validate_dino_query_output,
        validate_vlm_output,
    )
    from .qwen_engine import QwenVLEngine
    from .vlm_config import (
        DETECTIONS_TOPIC,
        DINO_QUERY_TOPIC,
        IMAGE_CACHE_SIZE,
        INFERENCE_COOLDOWN_S,
        INSTRUCTION_TOPIC,
        KEYFRAME_TOPIC,
        MODEL_PATH,
        OUTPUT_TOPIC,
        STATUS_TOPIC,
    )
except ImportError:
    from instruction_parser import instruction_to_dino_query_json
    from json_parser import (
        build_fallback_vlm_output,
        model_to_dict,
        validate_dino_query_output,
        validate_vlm_output,
    )
    from qwen_engine import QwenVLEngine
    from vlm_config import (
        DETECTIONS_TOPIC,
        DINO_QUERY_TOPIC,
        IMAGE_CACHE_SIZE,
        INFERENCE_COOLDOWN_S,
        INSTRUCTION_TOPIC,
        KEYFRAME_TOPIC,
        MODEL_PATH,
        OUTPUT_TOPIC,
        STATUS_TOPIC,
    )


SCHEDULER_PERIOD_S = 0.2


def stamp_key(sec, nanosec):
    return f"{int(sec)}:{int(nanosec)}"


def parse_json_payload(raw_text):
    text = raw_text.strip()

    if not text:
        return ""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text

def scene_to_action_payload(scene_payload):
    task = (scene_payload.get("task_queue") or [None])[0]
    if not task:
        return {
            "task_id": None,
            "target_object_id": None,
            "target_pose_world": None,
            "requires_reobserve": True,
            "confidence": 0.0,
        }

    target = task.get("target") or {}
    place = task.get("place_goal") or {}

    return {
        "task_id": task.get("task_id"),
        "target_object_id": target.get("object_id"),
        "target_label": target.get("label"),
        "target_color": target.get("color"),
        "target_category": target.get("category"),
        "target_pose_world": target.get("pose_world"),
        "source_location": target.get("support_surface"),
        "on_table": target.get("on_table"),
        "on_shelf": target.get("on_shelf"),
        "shelf_layer": target.get("shelf_layer"),
        "place_type": place.get("type"),
        "reference_object_id": place.get("reference_object_id"),
        "reference_label": place.get("reference_label"),
        "spatial_relation": place.get("spatial_relation"),
        "place_pose_world": place.get("pose_world"),
        "requires_reobserve": target.get("requires_reobserve", True),
        "confidence": target.get("confidence", 0.0),
    }

def instruction_payload_to_text(payload):
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, list):
        parts = [
            instruction_payload_to_text(item)
            for item in payload
        ]
        parts = [item for item in parts if item]
        return "\n".join(parts).strip()

    if isinstance(payload, dict):
        for key in (
            "original_instruction",
            "instruction",
            "text",
            "content",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if {
            "target_color",
            "target_body",
            "place_type",
        }.issubset(payload.keys()):
            target_color = payload.get("target_color", "")
            target_body = payload.get("target_body", "")
            place_type = payload.get("place_type", "")
            return f"把{target_color}的{target_body}搬到{place_type}".strip()

        return json.dumps(payload, ensure_ascii=False)

    return str(payload).strip()


class QwenVLNode(Node):
    def __init__(self, model_path):
        super().__init__("qwen_vl_node")

        self.model_path = model_path
        self.bridge = CvBridge()
        self.engine = QwenVLEngine(model_path)

        self.image_cache = OrderedDict()
        self.detection_cache = OrderedDict()

        self.latest_instruction = None
        self.latest_instruction_text = ""
        self.latest_query_payload = None
        self.instruction_version = 0

        self.last_processed_signature = None
        self.pending_signature = None
        self.pending_job = None

        self.busy = False
        self.last_inference_started = 0.0

        self.lock = threading.Lock()
        self.condition = threading.Condition()
        self.shutdown_event = threading.Event()

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

        self.dino_query_pub = self.create_publisher(
            String,
            DINO_QUERY_TOPIC,
            10,
        )

        self.scheduler_timer = self.create_timer(
            SCHEDULER_PERIOD_S,
            self.schedule_latest_job,
        )

        self.worker_thread = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker_thread.start()

        self.get_logger().info(f"Qwen模型路径: {model_path}")
        self.get_logger().info(f"订阅关键帧: {KEYFRAME_TOPIC}")
        self.get_logger().info(f"订阅检测结果: {DETECTIONS_TOPIC}")
        self.get_logger().info(f"订阅任务指令: {INSTRUCTION_TOPIC}")
        self.get_logger().info(f"发布DINO查询: {DINO_QUERY_TOPIC}")
        self.get_logger().info(f"发布语义结果: {OUTPUT_TOPIC}")

    def destroy_node(self):
        self.shutdown_event.set()
        with self.condition:
            self.condition.notify_all()

        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        return super().destroy_node()

    def publish_status(self, level, message, extra=None):
        payload = {
            "level": level,
            "message": message,
            "busy": self.busy,
            "timestamp": time.time(),
        }

        if extra:
            payload.update(extra)

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _trim_cache(self, cache):
        while len(cache) > IMAGE_CACHE_SIZE:
            cache.popitem(last=False)

    def image_callback(self, message):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
            frame_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
            image_pil = PILImage.fromarray(frame_rgb)

            key = stamp_key(
                message.header.stamp.sec,
                message.header.stamp.nanosec,
            )

            with self.lock:
                self.image_cache[key] = {
                    "image": image_pil,
                    "header": message.header,
                }
                self.image_cache.move_to_end(key)
                self._trim_cache(self.image_cache)

        except Exception as error:
            self.publish_status(
                "error",
                f"关键帧转换失败: {error}",
            )

    def detection_callback(self, message):
        try:
            payload = json.loads(message.data)

            source_stamp = payload.get("source_stamp")
            if not isinstance(source_stamp, dict):
                self.publish_status(
                    "warning",
                    "检测结果缺少 source_stamp",
                )
                return

            key = stamp_key(
                source_stamp.get("sec", 0),
                source_stamp.get("nanosec", 0),
            )

            with self.lock:
                self.detection_cache[key] = payload
                self.detection_cache.move_to_end(key)
                self._trim_cache(self.detection_cache)

        except Exception as error:
            self.publish_status(
                "error",
                f"检测结果解析失败: {error}",
            )

    def instruction_callback(self, message):
        try:
            payload = parse_json_payload(message.data)
            instruction_text = instruction_payload_to_text(payload)

            if not instruction_text:
                self.publish_status(
                    "warning",
                    "收到空指令，忽略",
                )
                return
            with self.lock:
                if instruction_text == self.latest_instruction_text:
                    return

            try:
                raw_query = (
                    self.engine
                    .infer_instruction_to_dino_query(
                        instruction_text
                    )
                )

                query_payload = validate_dino_query_output(
                    raw_query,
                    instruction_text=instruction_text,
                )

                query_json = json.dumps(
                    query_payload,
                    ensure_ascii=False,
                )

                query_source = "qwen"

            except Exception as error:
                self.publish_status(
                    "warning",
                    f"Qwen生成DINO查询失败，改用规则解析: {error}",
                )

                query_json = instruction_to_dino_query_json(
                    payload
                )
                query_payload = json.loads(query_json)
                query_source = "rule_fallback"

            query_msg = String()
            query_msg.data = query_json
            self.dino_query_pub.publish(query_msg)

            with self.lock:
                self.latest_instruction = payload
                self.latest_instruction_text = instruction_text
                self.latest_query_payload = query_payload
                self.instruction_version += 1

            self.publish_status(
                "info",
                "已接收任务指令并发布 DINO 查询",
                {
                    "instruction_version": self.instruction_version,
                    "query_source": query_source,
                },
            )

            self.schedule_latest_job()

        except Exception as error:
            self.publish_status(
                "error",
                f"指令处理失败: {error}",
            )

    def schedule_latest_job(self):
        if self.shutdown_event.is_set():
            return

        with self.lock:
            if not self.latest_instruction_text:
                return
            if self.busy:
                return

            if (
                time.monotonic() - self.last_inference_started
            ) < INFERENCE_COOLDOWN_S:
                return

            matched_keys = sorted(
                set(self.image_cache.keys())
                & set(self.detection_cache.keys())
            )

            if not matched_keys:
                return

            key = matched_keys[-1]
            signature = f"{key}|{self.instruction_version}"

            if signature == self.last_processed_signature:
                return

            if signature == self.pending_signature:
                return

            image_entry = self.image_cache[key]
            detection_payload = copy.deepcopy(
                self.detection_cache[key]
            )
            instruction_payload = copy.deepcopy(
                self.latest_instruction
            )
            query_payload = copy.deepcopy(
                self.latest_query_payload
            )

            self.pending_job = {
                "key": key,
                "signature": signature,
                "image": image_entry["image"],
                "detection_payload": detection_payload,
                "instruction_payload": instruction_payload,
                "instruction_text": self.latest_instruction_text,
                "query_payload": query_payload,
                "header": image_entry["header"],
            }
            self.pending_signature = signature

        with self.condition:
            self.condition.notify()

    def worker_loop(self):
        while not self.shutdown_event.is_set():
            with self.condition:
                while (
                    self.pending_job is None
                    and not self.shutdown_event.is_set()
                ):
                    self.condition.wait(timeout=0.5)

                if self.shutdown_event.is_set():
                    return

                job = self.pending_job
                self.pending_job = None
                self.pending_signature = None

            if job is not None:
                self.run_job(job)

    def run_job(self, job):
        key = job["key"]
        signature = job["signature"]

        with self.lock:
            self.busy = True
            self.last_processed_signature = signature
            self.last_inference_started = time.monotonic()

        self.publish_status(
            "info",
            "开始 Qwen 推理",
            {
                "key": key,
                "signature": signature,
            },
        )

        raw_text = ""

        try:
            raw_text = self.engine.infer(
                image_pil=job["image"],
                instruction_payload=job["instruction_payload"],
                detection_payload=job["detection_payload"],
                query_payload=job.get("query_payload"),
            )

            result = validate_vlm_output(
                raw_text,
                job["detection_payload"],
                instruction_text=job["instruction_text"],
                query_payload=job.get("query_payload"),
            )

            scene_payload = model_to_dict(result)
            action_payload = scene_to_action_payload(scene_payload)

            output_msg = String()
            output_msg.data = json.dumps(action_payload, ensure_ascii=False)
            self.output_pub.publish(output_msg)

            self.publish_status(
                "info",
                "已发布 /vlm/scene_understanding",
                {
                    "key": key,
                },
            )

        except Exception as error:
            fallback_result = build_fallback_vlm_output(
                instruction_text=job["instruction_text"],
                detection_payload=job["detection_payload"],
                reason=str(error),
                query_payload=job.get("query_payload"),
            )

            scene_payload = model_to_dict(result)
            action_payload = scene_to_action_payload(scene_payload)

            output_msg = String()
            output_msg.data = json.dumps(action_payload, ensure_ascii=False)
            self.output_pub.publish(output_msg)

            self.publish_status(
                "warning",
                "Qwen语义JSON失败，已发布规则兜底 /vlm/scene_understanding",
                {
                    "key": key,
                    "error": str(error),
                    "raw_output_head": raw_text[:1200],
                    "fallback_target_id": (
                        fallback_result.grounding.selected_object_id
                    ),
                },
            )

        finally:
            with self.lock:
                self.busy = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL scene understanding node"
    )

    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Qwen2.5-VL 本地模型路径",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()

    node = QwenVLNode(args.model_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()