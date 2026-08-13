#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import time
import threading
import os
from scipy.spatial.transform import Rotation
from vlm.scene_memory import SceneMemory
from vlm.layout_context import infer_surface_location

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.ops import box_convert, nms

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry


# ROS 2 只在相机模式中需要。
# 这样宿主机 vision 环境没有 rclpy 时，仍然可以运行图片模式。
try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from discoverse.robots.mmk2.mmk2_fk import MMK2FK

    ROS_AVAILABLE = True
    ROS_IMPORT_ERROR = None
except ImportError as error:
    rclpy = None
    Node = object
    ROS_AVAILABLE = False
    ROS_IMPORT_ERROR = error
try:
    from geometry_3d import (
        build_pose_camera_from_detection,
        camera_info_to_intrinsics,
        estimate_box_size_from_mask_depth,
        transform_pose_camera_to_world,
    )
except ImportError:
    from .geometry_3d import (
        build_pose_camera_from_detection,
        camera_info_to_intrinsics,
        estimate_box_size_from_mask_depth,
        transform_pose_camera_to_world,
    )

# =========================
# 路径配置
# =========================

DINO_ROOT = Path("/media/jiangzhenmin/data/Challengecup2026/JZM/GroundingDINO")
SAM_ROOT = Path("/media/jiangzhenmin/data/Challengecup2026/JZM/segment-anything")

DINO_CONFIG = (
    DINO_ROOT
    / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

DINO_CHECKPOINT = (
    DINO_ROOT
    / "weights/groundingdino_swint_ogc.pth"
)

SAM_CHECKPOINT = (
    SAM_ROOT
    / "sam_vit_h_4b8939.pth"
)

DEFAULT_IMAGE = Path(
    "/media/jiangzhenmin/data/Challengecup2026/JZM/Vision/input/box.png"
)

DEFAULT_OUTPUT = Path(
    "/media/jiangzhenmin/data/Challengecup2026/JZM/Vision/output"
)


# =========================
# GroundingDINO 图像预处理
# =========================

DINO_TRANSFORM = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GroundingDINO + SAM image/ROS2 camera inference"
    )

    parser.add_argument(
        "--mode",
        choices=["image", "camera"],
        default="image",
        help="image 表示本地图片；camera 表示 ROS2 机器人相机",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="图片模式下的输入图片",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=(
            "pink box . brown box . yellow box . "
            "white cylinder . white cuboid . white cube . "
            "shelf . table ."
        ),
        help='检测提示词，例如 "box . bottle . cup ."',
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--target-box-threshold",
        type=float,
        default=0.30,
        help="抓取目标动态查询时使用的较低框阈值，适应近距离/裁边目标",
    )

    parser.add_argument(
        "--target-text-threshold",
        type=float,
        default=0.22,
        help="抓取目标动态查询时使用的较低文本阈值",
    )

    parser.add_argument(
        "--vertical-box-aspect-ratio",
        type=float,
        default=1.25,
        help="判定竖放箱体的最小检测框高宽比",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    parser.add_argument(
        "--image-topic",
        default="/head_camera/color/image_raw",
        help="机器人 RGB 相机话题",
    )

    parser.add_argument(
    "--color-camera-info-topic",
    default="/head_camera/color/camera_info",
    help="RGB相机内参话题",
    )

    parser.add_argument(
    "--depth-topic",
    default="/head_camera/aligned_depth_to_color/image_raw",
    help="对齐到RGB图像的深度图话题，mono16，单位毫米",
    )

    parser.add_argument(
    "--depth-camera-info-topic",
    default="/head_camera/aligned_depth_to_color/camera_info",
    help="深度相机内参话题",
    )
    
    parser.add_argument(
        "--result-topic",
        default="/grounded_sam/detections",
        help="JSON 检测结果发布话题",
    )

    parser.add_argument(
        "--annotated-topic",
        default="/grounded_sam/annotated",
        help="可视化结果图像话题",
    )

    parser.add_argument(
        "--infer-period",
        type=float,
        default=1.0,
        help="两次推理之间的时间，单位为秒",
    )

    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否使用 OpenCV 窗口显示结果",
    )

    parser.add_argument(
        "--window-name",
        default="grounded_sam_camera",
    )

    parser.add_argument(
        "--keyframe-topic",
        default="/grounded_sam/keyframe",
    )

    parser.add_argument(
        "--dino-query-topic",
        default="/vlm/dino_query",
        help="Qwen或指令解析器发布的GroundingDINO动态检测词话题",
    )
    return parser.parse_args()


def select_device(device_argument):
    if device_argument == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_argument == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但 torch.cuda.is_available() 为 False")

    return device_argument


def validate_model_paths():
    paths = [
        DINO_CONFIG,
        DINO_CHECKPOINT,
        SAM_CHECKPOINT,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")


def read_image(image_path):
    """支持包含中文的图片路径。"""
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"无法读取图片：{image_path}")

    return image_bgr


def save_image(image_path, image):
    """支持包含中文的保存路径。"""
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = image_path.suffix or ".png"
    success, encoded_image = cv2.imencode(suffix, image)

    if not success:
        raise RuntimeError(f"无法编码图片：{image_path}")

    encoded_image.tofile(str(image_path))


def preprocess_frame(frame_bgr):
    """
    将 ROS/OpenCV 的 BGR numpy 图像转换为
    GroundingDINO 所需的归一化 Tensor。
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = PILImage.fromarray(frame_rgb)
    image_tensor, _ = DINO_TRANSFORM(image_pil, None)

    return image_tensor


def load_models(device):
    print(f"使用设备：{device}")

    print("正在加载 GroundingDINO...")

    dino_model = load_model(
        str(DINO_CONFIG),
        str(DINO_CHECKPOINT),
        device=device,
    )
    dino_model = dino_model.to(device)
    dino_model.eval()

    print("正在加载 SAM...")

    sam = sam_model_registry["vit_h"](
        checkpoint=str(SAM_CHECKPOINT)
    )
    sam = sam.to(device)
    sam.eval()

    sam_predictor = SamPredictor(sam)

    print("模型加载完成")

    return dino_model, sam_predictor


def run_grounding_dino(
    model,
    frame_bgr,
    text_prompt,
    box_threshold,
    text_threshold,
    device,
):
    image_tensor = preprocess_frame(frame_bgr)

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    return boxes, logits, phrases


def convert_boxes_to_xyxy(
    boxes,
    image_width,
    image_height,
):
    """
    GroundingDINO:
        归一化 cxcywh

    SAM:
        原图像素 xyxy
    """
    boxes_xyxy = box_convert(
        boxes=boxes,
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )

    scale = torch.tensor(
        [
            image_width,
            image_height,
            image_width,
            image_height,
        ],
        dtype=boxes_xyxy.dtype,
    )

    boxes_xyxy = boxes_xyxy * scale

    if len(boxes_xyxy) > 0:
        boxes_xyxy[:, 0::2].clamp_(0, image_width - 1)
        boxes_xyxy[:, 1::2].clamp_(0, image_height - 1)

    return boxes_xyxy

def filter_boxes_by_nms(
    boxes_xyxy,
    logits,
    phrases,
    iou_threshold=0.5,
    max_detections=8,
):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, logits, phrases

    keep = nms(
        boxes_xyxy.cpu(),
        logits.cpu(),
        iou_threshold,
    )

    keep = keep[:max_detections]

    return (
        boxes_xyxy[keep],
        logits[keep],
        [phrases[index] for index in keep.tolist()],
    )

def run_sam(
    predictor,
    image_rgb,
    image_bgr,
    boxes_xyxy,
    phrases,
    device,
    target_label=None,
):
    height, width = image_rgb.shape[:2]

    if len(boxes_xyxy) == 0:
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros((0,), dtype=np.float32),
        )

    predictor.set_image(image_rgb)

    all_masks = []
    all_scores = []

    for index, box in enumerate(boxes_xyxy):
        # 动态抓取查询可能返回错误的候选短语，例如把箱体标成
        # white cuboid。此时用任务目标标签生成已有的颜色提示点，
        # 避免 SAM 按错误短语去分割货架背景。HSV 阈值本身不变。
        prompt_label = target_label or phrases[index]
        point_coords, point_labels = get_color_prompt_points(
            image_bgr=image_bgr,
            box_xyxy=box.detach().cpu().numpy(),
            label=prompt_label,
        )

        transformed_box = predictor.transform.apply_boxes_torch(
            box[None, :].to(device),
            image_rgb.shape[:2],
        )

        if point_coords is not None:
            point_coords_torch = torch.as_tensor(
                point_coords,
                dtype=torch.float32,
                device=device,
            )[None, :, :]

            point_labels_torch = torch.as_tensor(
                point_labels,
                dtype=torch.int64,
                device=device,
            )[None, :]
        else:
            point_coords_torch = None
            point_labels_torch = None

        masks, mask_scores, _ = predictor.predict_torch(
            point_coords=point_coords_torch,
            point_labels=point_labels_torch,
            boxes=transformed_box,
            multimask_output=True,
        )

        best_index = torch.argmax(mask_scores[0])

        best_mask = masks[0, best_index]
        best_mask_numpy = best_mask.detach().cpu().numpy().astype(bool)
        if target_label:
            best_mask_numpy = refine_mask_for_target_color(
                image_bgr,
                best_mask_numpy,
                target_label,
            )
        best_score = mask_scores[0, best_index]

        all_masks.append(
            best_mask_numpy
        )
        all_scores.append(
            float(best_score.detach().cpu())
        )

    return (
        np.asarray(all_masks, dtype=bool),
        np.asarray(all_scores, dtype=np.float32),
    )
    # masks, mask_scores, _ = predictor.predict_torch(
    #     point_coords=None,
    #     point_labels=None,
    #     boxes=transformed_boxes,
    #     multimask_output=False,
    # )
    #
    # return (
    #     masks[:, 0].detach().cpu().numpy(),
    #     mask_scores[:, 0].detach().cpu().numpy(),
    # )


def infer_frame(
    frame_bgr,
    dino_model,
    sam_predictor,
    text_prompt,
    box_threshold,
    text_threshold,
    device,
    target_label=None,
):
    boxes, logits, phrases = run_grounding_dino(
        model=dino_model,
        frame_bgr=frame_bgr,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    height, width = frame_bgr.shape[:2]

    boxes_xyxy = convert_boxes_to_xyxy(
        boxes=boxes,
        image_width=width,
        image_height=height,
    )

    boxes_xyxy, logits, phrases = filter_boxes_by_nms(
        boxes_xyxy=boxes_xyxy,
        logits=logits,
        phrases=phrases,
        iou_threshold=0.45,
        max_detections=8,
    )
    boxes_xyxy, logits, phrases = filter_boxes_by_area(
        boxes_xyxy=boxes_xyxy,
        logits=logits,
        phrases=phrases,
        image_width=width,
        image_height=height,
        min_area_ratio=0.002,
        max_area_ratio=1.0,
    )

    image_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    masks, mask_scores = run_sam(
    predictor=sam_predictor,
    image_rgb=image_rgb,
    image_bgr=frame_bgr,
    boxes_xyxy=boxes_xyxy,
        phrases=phrases,
        device=device,
        target_label=target_label,
    )

    return (
        boxes_xyxy,
        logits,
        phrases,
        masks,
        mask_scores,
    )


def mask_centroid(mask):
    rows, columns = np.nonzero(mask)

    if len(columns) == 0:
        return None

    return [
        float(columns.mean()),
        float(rows.mean()),
    ]
def mask_bbox(mask):
    rows, columns = np.nonzero(mask)

    if len(columns) == 0:
        return None

    return [
        float(columns.min()),
        float(rows.min()),
        float(columns.max()),
        float(rows.max()),
    ]
def expected_color_from_label(label):
    label = str(label).lower()
    for color in ("pink", "yellow", "brown", "white"):
        if color in label:
            return color
    return None


KNOWN_COLORS = ("pink", "brown", "yellow", "white")


def color_consistent(label, estimated_color):
    expected = expected_color_from_label(label)
    if expected is None:
        return True
    return estimated_color == expected


def estimate_color_scores(image_bgr, mask):
    """Return HSV evidence ratios without changing the existing thresholds."""
    if mask is None or mask.sum() == 0:
        return {}

    mask_bool = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_bool = cv2.erode(mask_bool, kernel, iterations=1).astype(bool)
    pixels = image_bgr[mask_bool]
    if len(pixels) == 0:
        return {}

    hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2HSV,
    ).reshape(-1, 3)
    hue = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]
    valid = val > 40
    hue = hue[valid]
    sat = sat[valid]
    val = val[valid]
    if len(hue) == 0:
        return {}

    return {
        "pink": float(np.mean(
            (
                ((hue >= 145) & (hue <= 179))
                | ((hue >= 0) & (hue <= 5))
            )
            & (sat > 35)
            & (val > 120)
        )),
        "yellow": float(np.mean(
            (hue >= 18)
            & (hue <= 42)
            & (sat > 40)
            & (val > 120)
        )),
        "brown": float(np.mean(
            (hue >= 5)
            & (hue < 18)
            & (sat > 25)
            & (val >= 70)
            & (val < 230)
        )),
        "white": float(np.mean(
            (sat < 35)
            & (val > 160)
        )),
    }


def estimate_mask_color(image_bgr, mask, label=None):
    scores = estimate_color_scores(image_bgr, mask)
    if not scores:
        return None

    pink_ratio = scores["pink"]
    yellow_ratio = scores["yellow"]
    brown_ratio = scores["brown"]
    if yellow_ratio > 0.10 and yellow_ratio >= brown_ratio * 0.7:
        return "yellow"

    if brown_ratio > 0.10 and brown_ratio > yellow_ratio:
        return "brown"
    detected = max(scores, key=scores.get)

    # label_lower = str(label or "").lower()
    # for color in ("pink", "yellow", "brown", "white"):
    #     if color in label_lower and scores[color] > 0.08:
    #         return color

    if scores[detected] < 0.12:
        return "unknown"

    return detected

def filter_boxes_by_area(
    boxes_xyxy,
    logits,
    phrases,
    image_width,
    image_height,
    min_area_ratio=0.002,
    max_area_ratio=0.75,
):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, logits, phrases

    image_area = image_width * image_height

    widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    areas = widths * heights
    ratios = areas / float(image_area)

    keep = torch.where(
        (ratios >= min_area_ratio)
        & (ratios <= max_area_ratio)
    )[0]

    return (
        boxes_xyxy[keep],
        logits[keep],
        [phrases[index] for index in keep.tolist()],
    )
#新增函数：根据 label 在 DINO 框内找对应颜色区域
def get_color_prompt_points(image_bgr, box_xyxy, label):
    x1, y1, x2, y2 = box_xyxy.astype(int).tolist()
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    label_lower = str(label).lower()

    if "yellow" in label_lower:
        mask = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 38) & (hsv[:, :, 1] > 50)
    elif "pink" in label_lower:
        mask = ((hsv[:, :, 0] >= 140) | (hsv[:, :, 0] <= 10)) & (hsv[:, :, 1] > 30)
    elif "white" in label_lower:
        mask = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 150)
    elif "brown" in label_lower:
        mask = (
            (hsv[:, :, 0] >= 6)
            & (hsv[:, :, 0] <= 25)
            & (hsv[:, :, 1] > 25)
            & (hsv[:, :, 2] >= 80)
            & (hsv[:, :, 2] < 230)
        )
    else:
        return None, None

    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None, None

    point_x = float(x1 + xs.mean())
    point_y = float(y1 + ys.mean())

    return np.array([[point_x, point_y]], dtype=np.float32), np.array([1], dtype=np.int32)


def target_color_mask(image_bgr, label):
    """Return the existing HSV color mask for a task target."""
    label_lower = str(label or "").lower()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    if "yellow" in label_lower:
        return (
            (hsv[:, :, 0] >= 20)
            & (hsv[:, :, 0] <= 38)
            & (hsv[:, :, 1] > 50)
        )
    if "pink" in label_lower:
        return (
            ((hsv[:, :, 0] >= 140) | (hsv[:, :, 0] <= 10))
            & (hsv[:, :, 1] > 30)
        )
    if "brown" in label_lower:
        return (
            (hsv[:, :, 0] >= 6)
            & (hsv[:, :, 0] <= 25)
            & (hsv[:, :, 1] > 25)
            & (hsv[:, :, 2] >= 80)
            & (hsv[:, :, 2] < 230)
        )
    return None


def refine_mask_for_target_color(image_bgr, mask, label):
    """Trim a merged stacked-object mask to the requested object color.

    The HSV thresholds are intentionally the same ones already used by the
    project. This only changes the SAM mask and therefore the bbox/pose made
    from it; it does not relabel detections or alter color classification.
    """
    color_mask = target_color_mask(image_bgr, label)
    if color_mask is None:
        return mask.astype(bool)

    original = mask.astype(bool)
    candidate = original & color_mask
    original_area = int(original.sum())
    candidate_area = int(candidate.sum())
    if candidate_area < 20 or candidate_area < max(20, int(original_area * 0.08)):
        return original

    candidate_u8 = candidate.astype(np.uint8)
    kernel = np.ones((9, 9), dtype=np.uint8)
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_u8,
        connectivity=8,
    )
    if component_count <= 1:
        return candidate_u8.astype(bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    refined = component_labels == largest
    if int(refined.sum()) < max(20, int(candidate_area * 0.35)):
        return candidate
    return refined

def depth_consistency(depth_image, mask):
    if depth_image is None or mask is None or mask.sum() == 0:
        return None

    values = depth_image[mask.astype(bool)].astype(np.float32)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) < 20:
        return None

    p10 = np.percentile(values, 10)
    p90 = np.percentile(values, 90)

    return float((p90 - p10) / 1000.0)


def _box_region_mask(box, height, width):
    """Return a pixel mask for one original DINO box."""
    x1, y1, x2, y2 = [float(value) for value in box]
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    region = np.zeros((height, width), dtype=bool)
    region[top:bottom, left:right] = True
    return region


def _choose_detection_geometry(mask, dino_box, image_bgr, label):
    """Keep SAM for clean masks and fall back to DINO for merged masks.

    At close range a shelf board can be connected to the requested box by
    SAM.  The resulting mask then touches an image edge and its depth median
    describes the board instead of the box.  Restricting the mask to the
    original DINO box preserves the detector's object hypothesis and keeps
    all existing color thresholds unchanged.
    """
    height, width = image_bgr.shape[:2]
    original_mask = mask.astype(bool)
    mask_box = mask_bbox(original_mask)
    dino_box = [float(value) for value in dino_box]
    dino_width = max(0.0, dino_box[2] - dino_box[0])
    dino_height = max(0.0, dino_box[3] - dino_box[1])
    dino_area = dino_width * dino_height
    mask_area = float(original_mask.sum())
    mask_box_area = 0.0
    if mask_box is not None:
        mask_box_area = max(0.0, mask_box[2] - mask_box[0]) * max(
            0.0,
            mask_box[3] - mask_box[1],
        )

    mask_touches_edge = bool(
        mask_box
        and (
            mask_box[0] <= 1.0
            or mask_box[1] <= 1.0
            or mask_box[2] >= float(width - 2)
            or mask_box[3] >= float(height - 2)
        )
    )
    dino_is_interior = (
        dino_box[0] >= 4.0
        and dino_box[1] >= 4.0
        and dino_box[2] <= float(width - 4)
        and dino_box[3] <= float(height - 4)
    )
    merged_with_background = (
        mask_box_area > max(1.8 * dino_area, 0.55 * width * height)
        or (mask_touches_edge and dino_is_interior)
    )
    if not merged_with_background:
        return original_mask, dino_box, "sam_mask"

    dino_region = _box_region_mask(dino_box, height, width)
    fallback = original_mask & dino_region
    color_mask = target_color_mask(image_bgr, label)
    if color_mask is not None:
        color_fallback = color_mask & dino_region
        if int(color_fallback.sum()) >= max(20, int(fallback.sum() * 0.08)):
            fallback = color_fallback

    if int(fallback.sum()) < 20:
        return original_mask, dino_box, "sam_mask_unusable_fallback"

    return fallback, dino_box, "dino_box_fallback"

def create_detection_records(
    boxes_xyxy,
    logits,
    phrases,
    masks,
    mask_scores,
    image_bgr=None,
    depth_image=None,
    intrinsics=None,
    vertical_aspect_ratio=1.25,
):
    boxes_numpy = boxes_xyxy.cpu().numpy()
    records = []

    for index, box in enumerate(boxes_numpy):
        mask = masks[index]
        geometry_mask, geometry_box, geometry_source = (
            _choose_detection_geometry(
                mask=mask,
                dino_box=box.tolist(),
                image_bgr=image_bgr,
                label=phrases[index],
            )
            if image_bgr is not None
            else (mask.astype(bool), box.tolist(), "sam_mask")
        )
        estimated_color = (
            estimate_mask_color(
                image_bgr,
                geometry_mask,
                label=phrases[index],
            )
            if image_bgr is not None
            else None
        )
        mask_centroid_uv = mask_centroid(geometry_mask)
        box_values = [float(value) for value in geometry_box]
        box_width = max(0.0, box_values[2] - box_values[0])
        box_height = max(0.0, box_values[3] - box_values[1])
        bbox_centroid_uv = [
            (box_values[0] + box_values[2]) * 0.5,
            (box_values[1] + box_values[3]) * 0.5,
        ]
        is_vertical_box = (
            box_width > 1.0
            and box_height >= box_width * float(vertical_aspect_ratio)
            and box_height >= 60.0
        )
        # 竖放箱体的 SAM 掩膜可能只覆盖斜侧面；用 DINO 框中心作为
        # 平面目标中心更稳定，深度仍取 SAM 掩膜的中位数。
        center_uv = (
            bbox_centroid_uv if is_vertical_box else mask_centroid_uv
        )

        pose_camera = None
        size_3d = None
        label_lower = str(phrases[index]).lower()
        corrected_label = phrases[index]
        # estimated_color 只作为一致性证据，不覆盖 GroundingDINO 的
        # 类别标签。否则“white cylinder + yellow 外观”会被伪装成
        # yellow box，导航就会朝错误物体移动。
        if depth_image is not None and intrinsics is not None:
            pose_camera = build_pose_camera_from_detection(
                depth_image=depth_image,
                mask=geometry_mask,
                centroid_uv=center_uv,
                intrinsics=intrinsics,
            )
            size_3d = estimate_box_size_from_mask_depth(
                depth_image=depth_image,
                mask=geometry_mask,
                intrinsics=intrinsics,
            )
        records.append(
            {
                "index": index,
                "label": corrected_label,
                "raw_label": phrases[index],
                "corrected_label": corrected_label,
                "estimated_color": estimated_color,
                "color_scores": estimate_color_scores(image_bgr, mask),
                "color_consistent": color_consistent(
                    phrases[index],
                    estimated_color,
                ),
                "depth_span_m": depth_consistency(depth_image, geometry_mask),
                "dino_score": float(logits[index]),
                "sam_score": float(mask_scores[index]),
                "box_xyxy": geometry_box,
                "mask_area": int(geometry_mask.sum()),
                "centroid_uv": center_uv,
                "mask_centroid_uv": mask_centroid_uv,
                "bbox_centroid_uv": bbox_centroid_uv,
                "center_source": (
                    "bbox_center_vertical_box"
                    if is_vertical_box
                    else "mask_centroid"
                ),
                "vertical_box_candidate": is_vertical_box,
                "pose_camera": pose_camera,
                "pose_world": None,
                "size_3d": size_3d,
                "geometry_source": geometry_source,
            }
        
        )

    return records


def target_candidate_indices(records, target_label):
    """Keep only candidates that agree with a task-specific pick query."""
    target = str(target_label or "").lower()
    expected_color = expected_color_from_label(target)
    target_is_box = any(
        word in target for word in ("box", "cube", "cuboid", "block")
    )
    target_shape = next(
        (
            word
            for word in ("box", "cube", "cuboid", "cylinder")
            if word in target
        ),
        None,
    )

    selected = []
    for index, record in enumerate(records):
        label = str(
            record.get("corrected_label")
            or record.get("label")
            or record.get("raw_label")
            or ""
        ).lower()
        estimated_color = str(
            record.get("estimated_color") or ""
        ).lower()

        if expected_color and expected_color not in label:
            continue
        if expected_color == "brown":
            # 棕色箱在货架灯光下常被估成 yellow/unknown。仍要求 DINO
            # 语义标签是 brown box，但允许存在明确的 brown 像素证据。
            scores = record.get("color_scores") or {}
            brown_score = float(scores.get("brown") or 0.0)
            yellow_score = float(scores.get("yellow") or 0.0)
            support_surface = str(
                record.get("support_surface") or ""
            ).lower()
            on_shelf = bool(record.get("on_shelf")) or support_surface in (
                "shelf",
                "shelf_candidate",
            )
            dino_score = float(record.get("dino_score") or 0.0)
            strong_yellow_without_brown = (
                estimated_color == "yellow"
                and yellow_score >= 0.55
                and brown_score < 0.04
            )
            brown_like = (
                estimated_color == "brown"
                or brown_score >= 0.04
                or (
                    on_shelf
                    and
                    estimated_color in ("yellow", "unknown", "")
                    and dino_score >= 0.40
                    and not strong_yellow_without_brown
                )
            )
            if not brown_like:
                continue
        elif expected_color:
            # 任务已经通过精确 DINO 查询锁定了目标词（例如 pink box）。
            # 近距离货架灯光/阴影下浅粉色常被 HSV 估成 white，
            # 所以“白/未知”属于不确定证据，不能单独否决语义命中。
            # 只有明确的其他彩色证据才拒绝，避免白色货架板进入结果。
            color_scores = record.get("color_scores") or {}
            contradictory_score = max(
                float(color_scores.get(color) or 0.0)
                for color in KNOWN_COLORS
                if color != expected_color
            )
            contradictory_color = (
                estimated_color
                and estimated_color != expected_color
                and estimated_color != "white"
                and estimated_color != "unknown"
            )
            if contradictory_color and contradictory_score >= 0.45:
                continue
        if target_is_box and not any(
            word in label for word in ("box", "cube", "cuboid", "block")
        ):
            continue
        if target_shape and target_shape not in label:
            continue
        selected.append(index)

    return selected


def create_visualization(
    image_bgr,
    boxes,
    masks,
    phrases,
    logits,
):
    result = image_bgr.copy()

    colors = [
        (0, 200, 255),
        (255, 120, 0),
        (80, 220, 80),
        (220, 80, 180),
        (255, 80, 80),
        (80, 180, 255),
    ]

    for index, mask in enumerate(masks):
        color = np.array(
            colors[index % len(colors)],
            dtype=np.float32,
        )

        mask = mask.astype(bool)

        result[mask] = (
            result[mask].astype(np.float32) * 0.55
            + color * 0.45
        ).astype(np.uint8)

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int).tolist()
        color = colors[index % len(colors)]

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = (
            f"{phrases[index]} "
            f"{float(logits[index]):.2f}"
        )

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return result


def save_results(
    output_dir,
    image_bgr,
    boxes_xyxy,
    masks,
    phrases,
    logits,
    mask_scores,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes_numpy = boxes_xyxy.cpu().numpy()

    visualization = create_visualization(
        image_bgr=image_bgr,
        boxes=boxes_numpy,
        masks=masks,
        phrases=phrases,
        logits=logits,
    )

    result_path = output_dir / "grounded_sam_result.png"
    save_image(result_path, visualization)

    detection_records = create_detection_records(
        boxes_xyxy=boxes_xyxy,
        logits=logits,
        phrases=phrases,
        masks=masks,
        mask_scores=mask_scores,
        image_bgr=image_bgr,
    )

    for index, mask in enumerate(masks):
        mask_path = output_dir / f"mask_{index:03d}.png"
        save_image(
            mask_path,
            mask.astype(np.uint8) * 255,
        )

        detection_records[index]["mask_path"] = str(mask_path)

    json_path = output_dir / "detections.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            detection_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"可视化结果：{result_path}")
    print(f"检测信息：{json_path}")


def run_image_mode(
    args,
    dino_model,
    sam_predictor,
    device,
):
    if not args.image.exists():
        raise FileNotFoundError(
            f"输入图片不存在：{args.image}"
        )

    image_bgr = read_image(args.image)

    (
        boxes_xyxy,
        logits,
        phrases,
        masks,
        mask_scores,
    ) = infer_frame(
        frame_bgr=image_bgr,
        dino_model=dino_model,
        sam_predictor=sam_predictor,
        text_prompt=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    print(
        f"GroundingDINO 检测到 "
        f"{len(boxes_xyxy)} 个目标"
    )

    save_results(
        output_dir=args.output,
        image_bgr=image_bgr,
        boxes_xyxy=boxes_xyxy,
        masks=masks,
        phrases=phrases,
        logits=logits,
        mask_scores=mask_scores,
    )


class GroundedSamCameraNode(Node):
    def __init__(
        self,
        args,
        dino_model,
        sam_predictor,
        device,
    ):
        super().__init__("grounded_sam_camera_node")

        self.args = args
        self.device = device
        self.dino_model = dino_model
        self.sam_predictor = sam_predictor

        self.bridge = CvBridge()
        self.fk = MMK2FK(render_fk_xml())
        self.base_pos = None
        self.base_quat = None
        self.slide = 0.0
        self.head = [0.0, 0.0]
        self.latest_frame = None
        self.latest_header = None
        self.latest_depth = None
        self.latest_depth_header = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.frame_sequence = 0
        self.processed_sequence = -1
        self.busy = False
        self.text_prompt = args.text
        self.query_target_label = None
        self.query_role = ""
        self.text_lock = threading.Lock()
        self.scene_memory = SceneMemory()

        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        self.subscription = self.create_subscription(
            Image,
            args.image_topic,
            self.on_image,
            camera_qos,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            args.depth_topic,
            self.on_depth,
            camera_qos,
        )
        self.color_info_subscription = self.create_subscription(
            CameraInfo,
            args.color_camera_info_topic,
            self.on_color_camera_info,
            10,
        )
        self.depth_info_subscription = self.create_subscription(
            CameraInfo,
            args.depth_camera_info_topic,
            self.on_depth_camera_info,
            10,
        )
        self.dino_query_subscription = self.create_subscription(
            String,
            args.dino_query_topic,
            self.on_dino_query,
            10,
        )

        self.result_publisher = self.create_publisher(
            String,
            args.result_topic,
            10,
        )

        self.annotated_publisher = self.create_publisher(
            Image,
            args.annotated_topic,
            10,
        )

        self.timer = self.create_timer(
            args.infer_period,
            self.on_inference,
        )

        self.keyframe_publisher = self.create_publisher(
            Image,
            args.keyframe_topic,
            10,
        )
        self.joint_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_joint_states,
            10,
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            "/slamware_ros_sdk_server_node/odom",
            self.on_odom,
            10,
        )

        if args.display:
            cv2.namedWindow(
                args.window_name,
                cv2.WINDOW_NORMAL,
            )

        self.get_logger().info(
            f"订阅相机：{args.image_topic}"
        )
        self.get_logger().info(
            f"默认检测提示词：{args.text}"
        )
        self.get_logger().info(
            f"订阅动态检测词：{args.dino_query_topic}"
        )
        self.get_logger().info(
            f"推理设备：{device}"
        )

    def camera_world_tmat(self):
        if self.base_pos is None or self.base_quat is None:
            return None

        self.fk.set_base_pose(
            self.base_pos,
            self.base_quat,
        )
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints(
            [float(self.head[0]), float(self.head[1])]
        )
        self.fk.set_left_arm_joints([0.0] * 6)
        self.fk.set_right_arm_joints([0.0] * 6)

        pos, quat = self.fk.get_head_camera_pose()

        transform = np.eye(4)
        transform[:3, 3] = pos
        transform[:3, :3] = Rotation.from_quat(
            quat[[1, 2, 3, 0]]
        ).as_matrix()

        return transform
    
    def on_dino_query(self, message):
        try:
            payload = json.loads(message.data)
            prompt = payload.get(
                "grounding_prompt",
                "",
            ).strip()

            if not prompt:
                self.get_logger().warning(
                    "收到空的 grounding_prompt，保持原检测词"
                )
                return

            if not prompt.endswith("."):
                prompt = prompt + " ."

            with self.text_lock:
                self.text_prompt = prompt
                self.query_target_label = (
                    str(payload.get("target_label") or "").strip()
                    or None
                )
                self.query_role = str(
                    payload.get("query_role") or ""
                ).strip()

            self.get_logger().info(
                f"GroundingDINO检测词已更新：{prompt}，"
                f"query_role={self.query_role or 'default'}，"
                f"target_label={self.query_target_label or 'none'}"
            )

        except Exception as error:
            self.get_logger().error(
                f"解析 /vlm/dino_query 失败：{error}"
            )

    def on_image(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )

            self.latest_frame = frame.copy()
            self.latest_header = message.header
            self.frame_sequence += 1

        except Exception as error:
            self.get_logger().warning(
                f"相机图像转换失败：{error}"
            )

    def on_depth(self, message):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )

            self.latest_depth = depth.copy()
            self.latest_depth_header = message.header

        except Exception as error:
            self.get_logger().warning(
                f"深度图转换失败：{error}"
            )

    def on_color_camera_info(self, message):
        try:
            self.color_intrinsics = camera_info_to_intrinsics(
                message
            )
        except Exception as error:
            self.get_logger().warning(
                f"RGB相机内参解析失败：{error}"
            )

    def on_depth_camera_info(self, message):
        try:
            self.depth_intrinsics = camera_info_to_intrinsics(
                message
            )
        except Exception as error:
            self.get_logger().warning(
                f"深度相机内参解析失败：{error}"
            )

    def on_joint_states(self, message):
        positions = {
            name: message.position[index]
            for index, name in enumerate(message.name)
            if index < len(message.position)
        }

        self.slide = positions.get(
            "slide_joint",
            self.slide,
        )
        self.head = [
            positions.get("head_yaw_joint", self.head[0]),
            positions.get("head_pitch_joint", self.head[1]),
        ]

    def on_odom(self, message):
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation

        self.base_pos = [
            position.x,
            position.y,
            position.z,
        ]
        self.base_quat = [
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z,
        ]

    def on_inference(self):
        if self.busy:
            return

        if self.latest_frame is None:
            return

        if self.frame_sequence == self.processed_sequence:
            return

        self.busy = True

        frame = self.latest_frame.copy()
        header = self.latest_header
        sequence = self.frame_sequence
        depth = (
            self.latest_depth.copy()
            if self.latest_depth is not None
            else None
        )

        intrinsics = self.color_intrinsics or self.depth_intrinsics
        camera_world_tmat = self.camera_world_tmat()
        started_at = time.perf_counter()

        try:
            with self.text_lock:
                text_prompt = self.text_prompt
                query_target_label = self.query_target_label
                query_role = self.query_role

            targeted_pick_query = query_role == "pick_target_only" and bool(
                query_target_label
            )
            box_threshold = (
                min(self.args.box_threshold, self.args.target_box_threshold)
                if targeted_pick_query
                else self.args.box_threshold
            )
            text_threshold = (
                min(self.args.text_threshold, self.args.target_text_threshold)
                if targeted_pick_query
                else self.args.text_threshold
            )
            if targeted_pick_query and expected_color_from_label(
                query_target_label
            ) == "brown":
                # 棕色箱在货架阴影/暖光下 DINO 分数通常低于桌面目标；
                # 只放宽棕色目标的候选门槛，后续仍经过语义和颜色证据过滤。
                box_threshold = min(box_threshold, 0.24)
                text_threshold = min(text_threshold, 0.18)
            # 目标查询优先用精确短语，避免货架/桌面等对比词把目标
            # 的 GroundingDINO 分数压低。叠放拆分由颜色掩膜完成，
            # 候选仍会经过 target_candidate_indices 严格过滤。
            inference_prompt = (
                f"{query_target_label} ."
                if targeted_pick_query
                else text_prompt
            )
            (
                boxes_xyxy,
                logits,
                phrases,
                masks,
                mask_scores,
            ) = infer_frame(
                frame_bgr=frame,
                dino_model=self.dino_model,
                sam_predictor=self.sam_predictor,
                text_prompt=inference_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device,
                target_label=(query_target_label if targeted_pick_query else None),
            )

            inference_ms = (
                time.perf_counter() - started_at
            ) * 1000.0

            records = create_detection_records(
                boxes_xyxy=boxes_xyxy,
                logits=logits,
                phrases=phrases,
                masks=masks,
                mask_scores=mask_scores,
                image_bgr=frame,
                depth_image=depth,
                intrinsics=intrinsics,
                vertical_aspect_ratio=self.args.vertical_box_aspect_ratio,
            )
            for record in records:
                record["pose_world"] = transform_pose_camera_to_world(
                    record.get("pose_camera"),
                    camera_world_tmat,
        )
                surface_info = infer_surface_location(
                    pose_world=record.get("pose_world"),
                    label=record.get("corrected_label") or record.get("label"),
                )

                record.update(surface_info)

                missing = []

                if record.get("pose_camera") is None:
                    missing.append("pose_camera")

                if record.get("pose_world") is None:
                    missing.append("pose_world")

                if record.get("size_3d") is None:
                    missing.append("size_3d")

                record["contract_ready"] = len(missing) == 0
                record["contract_missing"] = missing

            raw_record_count = len(records)
            display_boxes = np.asarray(
                [
                    record.get("box_xyxy") or boxes_xyxy[index].tolist()
                    for index, record in enumerate(records)
                ],
                dtype=np.float32,
            ).reshape((-1, 4))
            display_masks = masks
            display_logits = logits
            display_phrases = [
                record.get("corrected_label") or record.get("label")
                for record in records
            ]

            # 任务抓取查询只保留语义和实际颜色都一致的候选。
            # 未通过的候选不进入scene_memory、不发给导航，也不画框。
            if targeted_pick_query:
                selected_indices = target_candidate_indices(
                    records,
                    query_target_label,
                )
                records = [records[index] for index in selected_indices]
                display_boxes = display_boxes[selected_indices]
                display_masks = masks[selected_indices]
                display_logits = logits[selected_indices]
                display_phrases = [
                    display_phrases[index]
                    for index in selected_indices
                ]

            filtered_record_count = len(records)

            if header is not None:
                    frame_key = (
                        f"{header.stamp.sec}_"
                        f"{header.stamp.nanosec}"
                    )

                    # 当前检测帧对应的ROS时间，单位为秒。
                    observed_at = (
                        float(header.stamp.sec)
                        + float(header.stamp.nanosec) * 1e-9
                    )
            else:
                    frame_key = str(sequence)

                    # 没有ROS Header时使用当前系统时间兜底。
                    observed_at = time.time()

            for index, record in enumerate(records):
                    # 当前帧内每个目标的唯一ID。
                    record["object_id"] = (
                        f"{frame_key}_{index}"
                    )

                    # 该检测结果来自头部RGB-D相机。
                    record["source_cameras"] = ["head_rgbd"]

                    # 当前目标的实际观测时间。
                    record["observed_at"] = float(observed_at)

                    # 第一版暂时没有从点云估计箱体真实yaw。
                    # 这里的0.0只是接口兜底值，不代表真实朝向。
                    record["yaw_world_rad"] = 0.0

                    # 第一版还没有做多帧位置稳定性统计。
                    record["position_std_m"] = None

                    # 第一版还没有做多帧yaw稳定性统计。
                    record["yaw_std_rad"] = None

            self.scene_memory.update_from_detections(records)
            annotated = create_visualization(
                image_bgr=frame,
                boxes=display_boxes,
                masks=display_masks,
                phrases=display_phrases,
                logits=display_logits,
            )

            self.publish_detection_result(
                records=records,
                inference_ms=inference_ms,
                header=header,
                caption=inference_prompt,
            )

            self.publish_annotated_image(
                image=annotated,
                header=header,
            )

            self.publish_keyframe(
                image=frame,
                header=header,
            )

            if self.args.display:
                cv2.imshow(
                    self.args.window_name,
                    annotated,
                )
                cv2.waitKey(1)

            self.processed_sequence = sequence
            self.get_logger().info(
                f"检测目标={len(records)}，"
                f"原始候选={raw_record_count}，"
                f"过滤后={filtered_record_count}，"
                f"推理耗时={inference_ms:.1f} ms"
            )

        except Exception as error:
            self.get_logger().error(
                f"推理失败：{error}"
            )

        finally:
            self.busy = False

    def publish_detection_result(
        self,
        records,
        inference_ms,
        header,
        caption,
    ):
        source_stamp = None
        frame_id = ""

        if header is not None:
            source_stamp = {
                "sec": int(header.stamp.sec),
                "nanosec": int(header.stamp.nanosec),
            }
            frame_id = header.frame_id

        payload = {
            "source_topic": self.args.image_topic,
            "source_stamp": source_stamp,
            "frame_id": frame_id,
            "caption": caption,
            "inference_ms": inference_ms,
            "detections": records,
            "scene_memory": self.scene_memory.to_payload(),
        }

        message = String()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
        )

        self.result_publisher.publish(message)

    def publish_annotated_image(
        self,
        image,
        header,
    ):
        message = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        if header is not None:
            message.header = header

        self.annotated_publisher.publish(message)

    def publish_keyframe(self, image, header):
        message = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        if header is not None:
            message.header = header

        self.keyframe_publisher.publish(message)

def run_camera_mode(
    args,
    dino_model,
    sam_predictor,
    device,
):
    if not ROS_AVAILABLE:
        raise RuntimeError(
            "当前 Python 环境没有 ROS 2 Python 模块。"
            f"原始错误：{ROS_IMPORT_ERROR}"
        )

    rclpy.init()

    node = GroundedSamCameraNode(
        args=args,
        dino_model=dino_model,
        sam_predictor=sam_predictor,
        device=device,
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

TASK_DIR = "/workspace/material_sorting_task/examples/material_sorting"
SOURCE_XML = os.path.join(
    TASK_DIR,
    "mjcf/material_competition.xml",
)
FK_XML = "/tmp/material_competition_fk.xml"


def render_fk_xml():
    with open(SOURCE_XML, "r", encoding="utf-8") as file:
        text = file.read().replace("__REPO_ROOT__", TASK_DIR)

    with open(FK_XML, "w", encoding="utf-8") as file:
        file.write(text)

    return FK_XML

def main():
    args = parse_args()

    validate_model_paths()

    device = select_device(args.device)

    dino_model, sam_predictor = load_models(device)

    if args.mode == "image":
        run_image_mode(
            args=args,
            dino_model=dino_model,
            sam_predictor=sam_predictor,
            device=device,
        )
    else:
        run_camera_mode(
            args=args,
            dino_model=dino_model,
            sam_predictor=sam_predictor,
            device=device,
        )


if __name__ == "__main__":
    main()
