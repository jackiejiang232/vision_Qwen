import copy
import math
import time

from .scene_reader import target_is_plausible_for_search


class SearchTargetLock:
    """Require geometrically consistent detections before search replanning."""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.samples = []
        self.last_observation_id = None
        self.last_valid_time = 0.0
        self.confirmed_target = None
        self.confirmed_time = 0.0
        self.last_rejection_reason = None

    @property
    def frame_count(self):
        return len(self.samples)

    def _pose_distance(self, first, second):
        return math.sqrt(
            sum(
                (float(first[axis]) - float(second[axis])) ** 2
                for axis in ("x", "y", "z")
            )
        )

    def _build_confirmed_target(self):
        target = copy.deepcopy(self.samples[-1])
        poses = [sample["pose_world"] for sample in self.samples]
        pose = dict(target["pose_world"])
        for axis in ("x", "y", "z"):
            pose[axis] = sum(float(item[axis]) for item in poses) / len(poses)
        target["pose_world"] = pose
        target["search_confirmed_frames"] = len(self.samples)
        return target

    def update(self, task, area_name, observation_id):
        if observation_id == self.last_observation_id:
            return self.get_confirmed()
        self.last_observation_id = observation_id

        now = time.monotonic()
        valid, reason = target_is_plausible_for_search(
            task,
            area_name,
            self.config,
        )
        self.last_rejection_reason = None if valid else reason
        if not valid:
            if now - self.last_valid_time > self.config.search_target_max_gap_sec:
                self.samples = []
            return self.get_confirmed()

        target = copy.deepcopy((task or {}).get("target") or {})
        task_specific = bool(
            target.get("label")
            and target.get("color")
            and target.get("category")
            and (
                target.get("support_surface")
                or target.get("source_location")
            )
        )
        required_frames = int(self.config.search_target_confirm_frames)
        if task_specific:
            required_frames = min(
                required_frames,
                int(
                    getattr(
                        self.config,
                        "search_target_confirm_frames_task",
                        required_frames,
                    )
                ),
            )
        pose = target.get("pose_world") or {}
        if self.samples:
            previous_pose = self.samples[-1].get("pose_world") or {}
            if self._pose_distance(pose, previous_pose) > float(
                self.config.search_target_pose_tolerance_m
            ):
                self.samples = []

        self.samples.append(target)
        self.samples = self.samples[-required_frames:]
        self.last_valid_time = now

        if len(self.samples) >= required_frames:
            self.confirmed_target = self._build_confirmed_target()
            self.confirmed_time = now

        return self.get_confirmed()

    def get_confirmed(self):
        if self.confirmed_target is None:
            return None
        if time.monotonic() - self.confirmed_time > float(
            self.config.search_target_lock_ttl_sec
        ):
            self.confirmed_target = None
            return None
        return copy.deepcopy(self.confirmed_target)
