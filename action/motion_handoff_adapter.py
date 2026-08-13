from dataclasses import dataclass

from third_party.dg202612.contracts import (
    BoxSize,
    CameraId,
)
import copy
from third_party.dg202612.navigation import operating_stance
from third_party.dg202612.vision_bridge import (
    parse_scene_understanding,
    reobserve_request,
    to_object_states,
)
try:
    from .motion_astar_nav import make_dock_route
except ImportError:
    make_dock_route = None

@dataclass
class MotionHandoffResult:
    status: str
    reason: str
    target: object = None
    approach_pose: object = None
    dock_route: object = None
    observation_request: object = None
    blocking_reasons: tuple = ()
    source_area: str = ""
    grasp_profile: str = ""
    astar_enabled: bool = False
    astar_waypoint_count: int = 0
    astar_cost_m: float = 0.0


def _box_size_from_target(target, default_box_size):
    size = target.get("size_3d") or target.get("size_m") or {}
    length = size.get("length")
    width = size.get("width")
    height = size.get("height")

    if length and width and height:
        return BoxSize(
            float(length),
            float(width),
            float(height),
        )

    return default_box_size


def _motion_profile_from_target(target, config):
    if target.get("support_surface") == "shelf" or target.get("on_shelf"):
        return {
            "source_area": "shelf",
            "grasp_profile": "shelf_extract_hug",
            "approach_direction": (-1.0, 0.0),
            "standoff": float(config.approach_distance_shelf),
        }

    return {
        "source_area": "table_side",
        "grasp_profile": "table_side_hug",
        "approach_direction": (0.0, 1.0),
        "standoff": float(config.approach_distance_table),
    }


def build_motion_handoff(
    scene_payload,
    config,
    robot_pose=None,
    active_task=None,
):
    default_box_size = BoxSize(
        float(config.motion_default_box_length),
        float(config.motion_default_box_width),
        float(config.motion_default_box_height),
    )

    dg_payload = _scene_for_dg202612(
        scene_payload,
        active_task=active_task,
    )
    scene = parse_scene_understanding(dg_payload)
    blocking = scene.blocking_reasons()
    if blocking:
        request = reobserve_request(
            scene,
            max_age=float(config.motion_reobserve_max_age_sec),
            min_confidence=float(config.motion_min_confidence),
            cameras=(CameraId.HEAD_RGBD,),
        )
        return MotionHandoffResult(
            status="need_reobserve",
            reason="; ".join(blocking),
            observation_request=request,
            blocking_reasons=blocking,
        )

    task = scene.active_task
    if task is None or not task.target_object_id:
        return MotionHandoffResult(
            status="need_reobserve",
            reason="active_task_missing_target_id",
        )

    vision_target = scene.object_by_id(task.target_object_id)
    if vision_target is None:
        return MotionHandoffResult(
            status="need_reobserve",
            reason="target_id_not_in_objects",
        )

    target_dict = None
    for item in dg_payload.get("objects") or []:
        if item.get("object_id") == task.target_object_id:
            target_dict = item
            break

    if target_dict is None:
        return MotionHandoffResult(
            status="need_reobserve",
            reason="target_raw_object_missing",
        )

    box_size = _box_size_from_target(target_dict, default_box_size)
    object_states = to_object_states(
        scene,
        box_size,
        cameras=(CameraId.HEAD_RGBD,),
    )
    object_by_id = {
        item.object_id: item
        for item in object_states
    }
    target_state = object_by_id.get(task.target_object_id)
    if target_state is None:
        return MotionHandoffResult(
            status="need_reobserve",
            reason="target_not_graspable_by_motion_contract",
        )

    profile = _motion_profile_from_target(target_dict, config)
    approach_pose = operating_stance(
        target_state,
        profile["approach_direction"],
        profile["standoff"],
    )
    dock_route = None
    astar_enabled = False
    astar_waypoint_count = 0
    astar_cost_m = 0.0

    if (
        getattr(config, "enable_motion_astar", False)
        and robot_pose is not None
        and make_dock_route is not None
    ):
        dock_route = make_dock_route(
            robot_pose,
            approach_pose,
            config,
        )
        astar_enabled = True
        astar_waypoint_count = len(dock_route.transit.waypoints)
        astar_cost_m = float(dock_route.transit.cost)

    return MotionHandoffResult(
        status="ok",
        reason="motion_contract_ready",
        target=target_state,
        approach_pose=approach_pose,
        dock_route=dock_route,
        source_area=profile["source_area"],
        grasp_profile=profile["grasp_profile"],
        astar_enabled=astar_enabled,
        astar_waypoint_count=astar_waypoint_count,
        astar_cost_m=astar_cost_m,
    )
def _pose_world_for_dg(pose):
    if pose is None:
        return None

    if isinstance(pose, dict):
        if all(axis in pose for axis in ("x", "y", "z")):
            return [
                float(pose["x"]),
                float(pose["y"]),
                float(pose["z"]),
            ]

    if isinstance(pose, (list, tuple)) and len(pose) == 3:
        return [
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
        ]

    return pose


def _safe_float(value, default=float("-inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_complete_pose(item):
    pose = item.get("pose_world")
    if isinstance(pose, dict):
        values = [pose.get(axis) for axis in ("x", "y", "z")]
    elif isinstance(pose, (list, tuple)) and len(pose) == 3:
        values = list(pose)
    else:
        return False

    return all(_safe_float(value, None) is not None for value in values)


def _object_preference(item):
    observed_at = max(
        _safe_float(item.get("observed_at")),
        _safe_float(item.get("timestamp")),
        _safe_float(item.get("stamp")),
    )
    confidence = max(
        _safe_float(item.get("confidence")),
        _safe_float(item.get("dino_score")),
        _safe_float(item.get("score")),
    )
    populated_fields = sum(
        value not in (None, "", [], {})
        for value in item.values()
    )
    return (
        int(_has_complete_pose(item)),
        observed_at,
        confidence,
        populated_fields,
    )


def _merge_object_records(preferred, fallback):
    merged = copy.deepcopy(preferred)
    for key, value in fallback.items():
        if merged.get(key) in (None, "", [], {}) and value not in (
            None,
            "",
            [],
            {},
        ):
            merged[key] = copy.deepcopy(value)
    return merged


def _deduplicate_objects(objects):
    """Merge memory/current-frame records before the strict DG parser."""
    deduplicated = []
    index_by_id = {}

    for raw_item in objects or []:
        item = copy.deepcopy(raw_item)
        if not isinstance(item, dict):
            deduplicated.append(item)
            continue

        object_id = str(item.get("object_id") or "").strip()
        if not object_id:
            # Preserve malformed records so the DG contract still reports them.
            deduplicated.append(item)
            continue

        existing_index = index_by_id.get(object_id)
        if existing_index is None:
            index_by_id[object_id] = len(deduplicated)
            deduplicated.append(item)
            continue

        existing = deduplicated[existing_index]
        if _object_preference(item) > _object_preference(existing):
            deduplicated[existing_index] = _merge_object_records(
                item,
                existing,
            )
        else:
            deduplicated[existing_index] = _merge_object_records(
                existing,
                item,
            )

    return deduplicated


def _active_task_only_scene(scene, active_task):
    """Build the DG contract from the task executor's current pick target.

    Qwen can briefly keep publishing the preceding task while the task
    executor has already advanced.  A motion handoff must therefore use the
    task that the navigation node has just bound from ``/task/pick_goal``.
    Static A* does not need unrelated visual objects, so keeping only the
    active target also prevents scene-memory duplicates from entering the
    strict DG parser.
    """
    task = copy.deepcopy(active_task or {})
    target = copy.deepcopy(task.get("target") or {})
    object_id = str(target.get("object_id") or "").strip()
    pose_world = target.get("pose_world")
    task_id = task.get("task_id")

    if not object_id or not pose_world or task_id is None:
        return scene

    dino_score = target.get("dino_score")
    if dino_score is None:
        dino_score = target.get("confidence", 0.0)
    dino_score = max(0.0, min(1.0, _safe_float(dino_score, 0.0)))

    sam_score = target.get("sam_score")
    if sam_score is not None:
        sam_score = max(0.0, min(1.0, _safe_float(sam_score, 0.0)))

    target["requires_reobserve"] = False
    task["target"] = target
    task.setdefault("status", "active")

    motion_object = copy.deepcopy(target)
    motion_object.update(
        {
            "object_id": object_id,
            "label": target.get("label") or "unknown target",
            "semantic_role": "active_task_target",
            "pose_world": pose_world,
            "dino_score": dino_score,
            "sam_score": sam_score,
            "on_shelf": bool(target.get("on_shelf")),
        }
    )

    observed_at = _safe_float(target.get("observed_at"), None)
    if observed_at is not None:
        source_sec = int(observed_at)
        source_nanosec = int((observed_at - source_sec) * 1_000_000_000)
    else:
        source_sec = scene.get("source_stamp_sec", 0)
        source_nanosec = scene.get("source_stamp_nanosec", 0)

    return {
        "schema_version": scene.get("schema_version") or "motion_active_task_1.0",
        "source_stamp_sec": source_sec,
        "source_stamp_nanosec": source_nanosec,
        "active_task_id": task_id,
        "scene_summary": f"motion handoff task {task_id}",
        "objects": [motion_object],
        "grounding": {
            "selected_object_id": object_id,
            "selected_label": motion_object["label"],
            "confidence": dino_score,
            "requires_reobserve": False,
            "reason": "task_executor_active_pick_target",
        },
        "task_queue": [task],
    }


def _scene_for_dg202612(scene_payload, active_task=None):
    scene = copy.deepcopy(scene_payload)
    if active_task is not None:
        scene = _active_task_only_scene(scene, active_task)
    scene["objects"] = _deduplicate_objects(scene.get("objects"))

    for item in scene.get("objects") or []:
        item["pose_world"] = _pose_world_for_dg(
            item.get("pose_world")
        )

    for task in scene.get("task_queue") or []:
        target = task.get("target") or {}
        if "pose_world" in target:
            target["pose_world"] = _pose_world_for_dg(
                target.get("pose_world")
            )

        place_goal = task.get("place_goal") or {}
        if "pose_world" in place_goal:
            place_goal["pose_world"] = _pose_world_for_dg(
                place_goal.get("pose_world")
            )

    return scene
