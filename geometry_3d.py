import math

import numpy as np


def camera_info_to_intrinsics(camera_info):
    k = camera_info.k
    return {
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
        "frame_id": camera_info.header.frame_id,
    }


def valid_depth_values(depth_image, mask=None):
    depth = depth_image.astype(np.float32)

    if mask is not None:
        depth = depth[mask.astype(bool)]
    else:
        depth = depth.reshape(-1)

    depth = depth[np.isfinite(depth)]
    depth = depth[depth > 0]

    return depth


def depth_mm_to_m(depth_mm):
    return float(depth_mm) / 1000.0


def median_depth_m(depth_image, mask):
    values = valid_depth_values(depth_image, mask)

    if len(values) == 0:
        return None

    return depth_mm_to_m(np.median(values))


def pixel_to_camera_xyz(u, v, depth_m, intrinsics):
    if depth_m is None or depth_m <= 0:
        return None

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x = (float(u) - cx) * depth_m / fx
    y = (float(v) - cy) * depth_m / fy
    z = depth_m

    if not all(math.isfinite(item) for item in (x, y, z)):
        return None

    return [x, y, z]


def mask_points_camera(depth_image, mask, intrinsics, max_points=3000):
    rows, cols = np.nonzero(mask.astype(bool))

    if len(cols) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if len(cols) > max_points:
        indices = np.linspace(0, len(cols) - 1, max_points).astype(int)
        rows = rows[indices]
        cols = cols[indices]

    depth_mm = depth_image[rows, cols].astype(np.float32)
    valid = np.isfinite(depth_mm) & (depth_mm > 0)

    rows = rows[valid]
    cols = cols[valid]
    depth_m = depth_mm[valid] / 1000.0

    if len(depth_m) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    x = (cols.astype(np.float32) - cx) * depth_m / fx
    y = (rows.astype(np.float32) - cy) * depth_m / fy
    z = depth_m

    return np.stack([x, y, z], axis=1).astype(np.float32)


def estimate_box_size_from_mask_depth(depth_image, mask, intrinsics):
    points = mask_points_camera(depth_image, mask, intrinsics)

    if len(points) < 10:
        return None

    # 使用分位数减少边缘噪声和深度飞点影响。
    q_low = np.percentile(points, 5, axis=0)
    q_high = np.percentile(points, 95, axis=0)
    extent = np.maximum(q_high - q_low, 0.0)

    return {
        "length": float(extent[0]),
        "width": float(extent[1]),
        "height": float(extent[2]),
    }


def build_pose_camera_from_detection(depth_image, mask, centroid_uv, intrinsics):
    if centroid_uv is None:
        return None

    depth_m = median_depth_m(depth_image, mask)
    if depth_m is None:
        return None

    xyz = pixel_to_camera_xyz(
        centroid_uv[0],
        centroid_uv[1],
        depth_m,
        intrinsics,
    )

    if xyz is None:
        return None

    return {
        "x": xyz[0],
        "y": xyz[1],
        "z": xyz[2],
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "frame_id": intrinsics.get("frame_id"),
    }

def transform_pose_camera_to_world(pose_camera, camera_world_tmat):
    if pose_camera is None or camera_world_tmat is None:
        return None

    point = np.array(
        [
            pose_camera["x"],
            pose_camera["y"],
            pose_camera["z"],
            1.0,
        ],
        dtype=np.float64,
    )

    world = camera_world_tmat @ point

    return {
        "x": float(world[0]),
        "y": float(world[1]),
        "z": float(world[2]),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "frame_id": "world",
    }