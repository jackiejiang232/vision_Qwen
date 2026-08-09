"""场景输入边界、视觉证据检查与人工案例适配。

官方检测、头部 RGB-D、左右腕 RGB 和场景记忆在这里汇聚为 ``SceneState``。
本模块只判断数据能否支持当前动作；视觉层不能在这里发布运动控制。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .contracts import (
    BoxSize,
    CameraId,
    CameraObservation,
    GraspEvidence,
    GraspProfile,
    ObjectState,
    ObservationPurpose,
    ObservationRequest,
    PickPlaceGoal,
    Pose2D,
    Pose3D,
    RobotState,
    SceneState,
    ShelfState,
    TableState,
    TaskId,
)


class SceneInputError(ValueError):
    """人工案例或未来视觉适配器没有提供完整、可追溯的数据。"""


@dataclass(frozen=True)
class FreshnessReport:
    fresh: bool
    stale_sources: tuple[str, ...]


@dataclass(frozen=True)
class ObservationReport:
    """目标定位与相机帧是否足以支持一次规划或动作检查。"""

    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceReport:
    """抓取证据检查结果；``unsafe`` 表示不能靠等待下一帧解决。"""

    accepted: bool
    unsafe: bool
    reasons: tuple[str, ...]


def freshness(scene: SceneState, now: float, max_age: float) -> FreshnessReport:
    """统一检查规划所依赖的机器人和物体观测，而不是静默使用陈旧坐标。"""

    if max_age <= 0.0:
        raise ValueError("max_age must be positive")
    if not math.isfinite(now):
        raise ValueError("now must be finite")
    stale = []
    if now - scene.robot.observed_at > max_age:
        stale.append("robot_state")
    for item in scene.objects:
        if now - item.observed_at > max_age:
            stale.append(f"object:{item.object_id}")
    return FreshnessReport(not stale, tuple(stale))


def validate_observation(
    scene: SceneState,
    request: ObservationRequest,
    *,
    now: float,
) -> ObservationReport:
    """验证主动感知请求是否已经得到一份可执行的回答。

    腕部 RGB 没有深度，箱体三维位置仍必须来自头部 RGB-D。腕部图像只证明局部
    视角已经到位；接触与抬升结论由 ``validate_grasp_evidence`` 另行检查。
    """

    if not math.isfinite(now):
        raise ValueError("now must be finite")
    reasons: list[str] = []
    target = scene.object_by_id(request.target_id)
    if target is None:
        reasons.append(f"target {request.target_id} is absent")
    elif request.require_target_pose:
        target_age = now - target.observed_at
        if target_age < 0.0:
            reasons.append("target observation timestamp is in the future")
        elif target_age > request.max_age:
            reasons.append(f"target observation is stale ({target_age:.3f}s)")
        if target.confidence < request.min_confidence:
            reasons.append(
                f"target confidence {target.confidence:.3f} is below "
                f"{request.min_confidence:.3f}"
            )
        if CameraId.HEAD_RGBD not in target.source_cameras:
            reasons.append("target 3D pose is not sourced from synchronized head RGB-D")
        if request.max_position_std_m is not None:
            if target.position_std_m is None:
                reasons.append("target position uncertainty is missing")
            elif target.position_std_m > request.max_position_std_m:
                reasons.append(
                    f"target position uncertainty {target.position_std_m:.3f}m exceeds "
                    f"{request.max_position_std_m:.3f}m"
                )
        if request.max_yaw_std_rad is not None:
            if target.yaw_std_rad is None:
                reasons.append("target yaw uncertainty is missing")
            elif target.yaw_std_rad > request.max_yaw_std_rad:
                reasons.append(
                    f"target yaw uncertainty {target.yaw_std_rad:.3f}rad exceeds "
                    f"{request.max_yaw_std_rad:.3f}rad"
                )

    robot_age = now - scene.robot.observed_at
    if robot_age < 0.0:
        reasons.append("robot state timestamp is in the future")
    elif robot_age > request.max_age:
        reasons.append(f"robot state is stale ({robot_age:.3f}s)")
    if request.require_stationary:
        if abs(scene.robot.base_linear) > request.max_base_linear:
            reasons.append("base is moving too fast for this observation")
        if abs(scene.robot.base_angular) > request.max_base_angular:
            reasons.append("base is rotating too fast for this observation")

    for camera in request.cameras:
        observation = scene.camera_by_id(camera)
        if observation is None:
            reasons.append(f"missing camera frame: {camera.value}")
            continue
        age = now - observation.observed_at
        if age < 0.0:
            reasons.append(f"camera timestamp is in the future: {camera.value}")
        elif age > request.max_age:
            reasons.append(f"stale camera frame: {camera.value} ({age:.3f}s)")
    return ObservationReport(not reasons, tuple(reasons))


def validate_grasp_evidence(
    scene: SceneState,
    request: ObservationRequest,
    *,
    now: float,
    max_centered_error_m: float,
) -> EvidenceReport:
    """检查腕部视觉是否允许继续接近、抬升或撤离。"""

    if request.purpose not in {
        ObservationPurpose.GUARD_APPROACH,
        ObservationPurpose.VERIFY_HOLD,
        ObservationPurpose.VERIFY_LIFT,
    }:
        raise ValueError("grasp evidence is not defined for this observation purpose")
    if max_centered_error_m < 0.0:
        raise ValueError("max_centered_error_m cannot be negative")
    evidence = scene.grasp_evidence
    if evidence is None:
        return EvidenceReport(False, False, ("grasp evidence is missing",))
    reasons: list[str] = []
    unsafe = False
    if evidence.target_id != request.target_id:
        reasons.append("grasp evidence belongs to another target")
    age = now - evidence.observed_at
    if age < 0.0:
        reasons.append("grasp evidence timestamp is in the future")
    elif age > request.max_age:
        reasons.append(f"grasp evidence is stale ({age:.3f}s)")
    required_wrist_cameras = {
        camera for camera in request.cameras if camera is not CameraId.HEAD_RGBD
    }
    missing = required_wrist_cameras.difference(evidence.source_cameras)
    if missing:
        reasons.append(
            "grasp evidence is missing "
            + ", ".join(sorted(camera.value for camera in missing))
        )
    if not evidence.safe_to_continue:
        reasons.append("local visual guard marked the grasp unsafe")
        unsafe = True
    if evidence.left_contact_confirmed != evidence.right_contact_confirmed:
        reasons.append("only one arm appears to contact the box")
        unsafe = True

    if request.purpose in {
        ObservationPurpose.VERIFY_HOLD,
        ObservationPurpose.VERIFY_LIFT,
    }:
        if not (
            evidence.left_contact_confirmed
            and evidence.right_contact_confirmed
        ):
            reasons.append("bilateral contact is not confirmed")
            unsafe = True
        if evidence.centered_error_m is None:
            reasons.append("box centering error is missing")
        elif evidence.centered_error_m > max_centered_error_m:
            reasons.append(
                f"box centering error {evidence.centered_error_m:.3f}m exceeds "
                f"{max_centered_error_m:.3f}m"
            )
            unsafe = True
    if (
        request.purpose is ObservationPurpose.VERIFY_LIFT
        and not evidence.object_lifted
    ):
        reasons.append("box lift is not visually confirmed")
        unsafe = True
    return EvidenceReport(not reasons, unsafe, tuple(reasons))


def object_by_color(scene: SceneState, color: str) -> ObjectState:
    """颜色必须唯一；多目标歧义由视觉/VLA 层消解，规划层不能私自猜一个。"""

    matches = [item for item in scene.objects if item.color == color]
    if len(matches) != 1:
        raise SceneInputError(
            f"expected exactly one {color} object, found {len(matches)}"
        )
    return matches[0]


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SceneInputError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise SceneInputError(f"{field} must be finite")
    return number


def _optional_number(value: Any, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _sequence(value: Any, length: int, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SceneInputError(f"{field} must contain exactly {length} values")
    return list(value)


def _pose2(value: Any, field: str) -> Pose2D:
    x, y, yaw = _sequence(value, 3, field)
    return Pose2D(_number(x, f"{field}[0]"), _number(y, f"{field}[1]"), _number(yaw, f"{field}[2]"))


def _pose3(value: Any, field: str) -> Pose3D:
    x, y, z, roll, pitch, yaw = _sequence(value, 6, field)
    return Pose3D(*(_number(item, f"{field}[{index}]") for index, item in enumerate((x, y, z, roll, pitch, yaw))))


def _box_size(value: Any, field: str) -> BoxSize:
    length, width, height = _sequence(value, 3, field)
    return BoxSize(
        _number(length, f"{field}[0]"),
        _number(width, f"{field}[1]"),
        _number(height, f"{field}[2]"),
    )


def _shelf(value: Any) -> ShelfState | None:
    """货架场景记忆是可选的；缺省表示视觉层尚未确认，而不是造一个空货架。"""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SceneInputError("scene.shelf must be an object")
    empty = value.get("empty_levels", ())
    if not isinstance(empty, (list, tuple)):
        raise SceneInputError("shelf.empty_levels must be a list")
    obstacle = value.get("obstacle_level")
    level_poses_raw = value.get("level_poses")
    level_poses = None
    if level_poses_raw is not None:
        if not isinstance(level_poses_raw, Mapping):
            raise SceneInputError("shelf.level_poses must be an object")
        level_poses = {
            int(level): _pose3(pose, f"shelf.level_poses[{level}]")
            for level, pose in level_poses_raw.items()
        }
    return ShelfState(
        empty_levels=tuple(int(level) for level in empty),
        obstacle_level=None if obstacle is None else int(obstacle),
        level_poses=level_poses,
    )


def _table(value: Any) -> TableState | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SceneInputError("scene.table must be an object")
    original = value.get("side_original_pose")
    return TableState(
        side_original_pose=None if original is None else _pose3(original, "table.side_original_pose"),
        side=value.get("side"),
    )


def _camera_observations(value: Any) -> tuple[CameraObservation, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SceneInputError("scene.camera_observations must be a list")
    observations = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SceneInputError(
                f"scene.camera_observations[{index}] must be an object"
            )
        try:
            camera = CameraId(str(item.get("camera")))
        except ValueError as exc:
            raise SceneInputError(
                f"unsupported camera in scene.camera_observations[{index}]"
            ) from exc
        observations.append(
            CameraObservation(
                camera=camera,
                observed_at=_number(
                    item.get("observed_at"),
                    f"scene.camera_observations[{index}].observed_at",
                ),
                frame_id=str(item.get("frame_id", "")),
            )
        )
    return tuple(observations)


def _grasp_evidence(value: Any) -> GraspEvidence | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SceneInputError("scene.grasp_evidence must be an object")
    raw_cameras = value.get("source_cameras")
    if not isinstance(raw_cameras, list):
        raise SceneInputError("grasp_evidence.source_cameras must be a list")
    try:
        cameras = tuple(CameraId(str(item)) for item in raw_cameras)
    except ValueError as exc:
        raise SceneInputError("grasp_evidence has an unsupported camera") from exc
    return GraspEvidence(
        target_id=str(value.get("target_id", "")).strip(),
        observed_at=_number(
            value.get("observed_at"), "grasp_evidence.observed_at"
        ),
        source_cameras=cameras,
        safe_to_continue=bool(value.get("safe_to_continue", False)),
        left_contact_confirmed=bool(
            value.get("left_contact_confirmed", False)
        ),
        right_contact_confirmed=bool(
            value.get("right_contact_confirmed", False)
        ),
        centered_error_m=_optional_number(
            value.get("centered_error_m"), "grasp_evidence.centered_error_m"
        ),
        object_lifted=bool(value.get("object_lifted", False)),
    )


class ManualSceneAdapter:
    """读取人工审核案例；该类故意不提供任何默认坐标。"""

    @staticmethod
    def from_mapping(payload: Mapping[str, Any]) -> tuple[SceneState, PickPlaceGoal]:
        scene_data = payload.get("scene")
        goal_data = payload.get("goal")
        if not isinstance(scene_data, Mapping) or not isinstance(goal_data, Mapping):
            raise SceneInputError("manual case requires scene and goal objects")
        robot_data = scene_data.get("robot")
        if not isinstance(robot_data, Mapping):
            raise SceneInputError("scene.robot is required")

        left_arm = tuple(_number(item, "robot.left_arm") for item in _sequence(robot_data.get("left_arm"), 6, "robot.left_arm"))
        right_arm = tuple(_number(item, "robot.right_arm") for item in _sequence(robot_data.get("right_arm"), 6, "robot.right_arm"))
        robot = RobotState(
            base=_pose2(robot_data.get("base_pose"), "robot.base_pose"),
            base_linear=_number(robot_data.get("base_linear"), "robot.base_linear"),
            base_angular=_number(robot_data.get("base_angular"), "robot.base_angular"),
            slide=_number(robot_data.get("slide"), "robot.slide"),
            head_yaw=_number(robot_data.get("head_yaw"), "robot.head_yaw"),
            head_pitch=_number(robot_data.get("head_pitch"), "robot.head_pitch"),
            left_arm=left_arm,
            left_gripper=_number(robot_data.get("left_gripper"), "robot.left_gripper"),
            right_arm=right_arm,
            right_gripper=_number(robot_data.get("right_gripper"), "robot.right_gripper"),
            observed_at=_number(robot_data.get("observed_at"), "robot.observed_at"),
        )

        raw_objects = scene_data.get("objects")
        if not isinstance(raw_objects, list):
            raise SceneInputError("scene.objects must be a list")
        objects = []
        for index, item in enumerate(raw_objects):
            if not isinstance(item, Mapping):
                raise SceneInputError(f"scene.objects[{index}] must be an object")
            raw_sources = item.get("source_cameras", [])
            if not isinstance(raw_sources, list):
                raise SceneInputError(
                    f"objects[{index}].source_cameras must be a list"
                )
            try:
                source_cameras = tuple(
                    CameraId(str(camera)) for camera in raw_sources
                )
            except ValueError as exc:
                raise SceneInputError(
                    f"objects[{index}] has an unsupported source camera"
                ) from exc
            objects.append(
                ObjectState(
                    object_id=str(item.get("object_id", "")).strip(),
                    color=str(item.get("color", "")).strip(),
                    pose=_pose3(item.get("pose"), f"objects[{index}].pose"),
                    size=_box_size(item.get("size"), f"objects[{index}].size"),
                    observed_at=_number(item.get("observed_at"), f"objects[{index}].observed_at"),
                    confidence=_number(item.get("confidence", 1.0), f"objects[{index}].confidence"),
                    source_cameras=source_cameras,
                    position_std_m=_optional_number(
                        item.get("position_std_m"),
                        f"objects[{index}].position_std_m",
                    ),
                    yaw_std_rad=_optional_number(
                        item.get("yaw_std_rad"),
                        f"objects[{index}].yaw_std_rad",
                    ),
                )
            )
        scene = SceneState(
            timestamp=_number(scene_data.get("timestamp"), "scene.timestamp"),
            robot=robot,
            objects=tuple(objects),
            instruction=scene_data.get("instruction") if isinstance(scene_data.get("instruction"), Mapping) else None,
            shelf=_shelf(scene_data.get("shelf")),
            table=_table(scene_data.get("table")),
            camera_observations=_camera_observations(
                scene_data.get("camera_observations")
            ),
            grasp_evidence=_grasp_evidence(scene_data.get("grasp_evidence")),
        )
        target_id = str(goal_data.get("target_id", "")).strip()
        target = scene.object_by_id(target_id)
        if target is None:
            raise SceneInputError(f"goal target_id is not present: {target_id}")
        try:
            task_id = TaskId(str(goal_data.get("task_id")))
            grasp_profile = GraspProfile(str(goal_data.get("grasp_profile")))
        except ValueError as exc:
            raise SceneInputError("goal has an unsupported task_id or grasp_profile") from exc
        source_area = str(goal_data.get("source_area", "")).strip()
        place_type = str(goal_data.get("place_type", "")).strip() or None
        return scene, PickPlaceGoal(
            task_id=task_id,
            target_id=target.object_id,
            target_color=target.color,
            target_pose=target.pose,
            target_size=target.size,
            source_area=source_area,
            grasp_profile=grasp_profile,
            place_type=place_type,
            retry_limit=int(goal_data.get("retry_limit", 0)),
        )
