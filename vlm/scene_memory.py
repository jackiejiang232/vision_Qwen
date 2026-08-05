import time

from .layout_context import is_manipulable_object


class SceneMemory:
    def __init__(self):
        self.table_original_poses = {}
        self.shelf_object_levels = {}
        self.last_update_time = 0.0

    def update_from_detections(self, detections, layout_context=None):
        self.last_update_time = time.time()

        for item in detections:
            label = str(item.get("label") or "").lower()
            pose_world = item.get("pose_world")

            if pose_world is None or not is_manipulable_object(label):
                continue

            if item.get("on_table"):
                self.table_original_poses.setdefault(
                    label,
                    {
                        "object_id": item.get("object_id"),
                        "label": label,
                        "pose_world": pose_world,
                        "size_3d": item.get("size_3d"),
                        "confidence": item.get(
                            "table_height_confidence",
                            0.0,
                        ),
                        "last_seen": self.last_update_time,
                    },
                )

            shelf_layer = item.get("shelf_layer")

            if item.get("on_shelf") and shelf_layer is not None:
                self.shelf_object_levels[label] = {
                    "object_id": item.get("object_id"),
                    "label": label,
                    "layer": int(shelf_layer),
                    "support_surface_index": item.get(
                        "support_surface_index"
                    ),
                    "pose_world": pose_world,
                    "size_3d": item.get("size_3d"),
                    "confidence": item.get(
                        "shelf_layer_confidence",
                        0.0,
                    ),
                    "last_seen": self.last_update_time,
                }

    def get_table_original_pose(self, label):
        return self.table_original_poses.get(
            str(label or "").lower()
        )

    def to_payload(self):
        return {
            "table_original_poses": self.table_original_poses,
            "shelf_object_levels": self.shelf_object_levels,
            "last_update_time": self.last_update_time,
        }
