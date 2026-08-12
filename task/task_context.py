from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskObject:
    object_id: Optional[str] = None
    label: Optional[str] = None
    pose_world: Optional[Dict[str, float]] = None
    size_3d: Optional[Dict[str, float]] = None
    confidence: float = 0.0


@dataclass
class Destination:
    type: str = "unknown"
    place_relation: Optional[str] = None
    place_type: Optional[str] = None
    direction: Optional[str] = None
    shelf_id: Optional[str] = None
    level_id: Optional[int] = None
    reference_object_id: Optional[str] = None
    reference_pose_world: Optional[Dict[str, float]] = None
    approach_pose: Optional[Dict[str, float]] = None
    place_pose: Optional[Dict[str, float]] = None


@dataclass
class TaskContext:
    task_id: str
    raw_instruction: str = ""
    state: str = "WAIT_TASK"
    reason: str = ""
    home_pose: Optional[Dict[str, float]] = None
    pick_target: TaskObject = field(default_factory=TaskObject)
    place_reference: TaskObject = field(default_factory=TaskObject)
    destination: Destination = field(default_factory=Destination)
    held_object_id: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "raw_instruction": self.raw_instruction,
            "state": self.state,
            "reason": self.reason,
            "home_pose": self.home_pose,
            "pick_target": self.pick_target.__dict__,
            "place_reference": self.place_reference.__dict__,
            "destination": self.destination.__dict__,
            "held_object_id": self.held_object_id,
            "completed_steps": list(self.completed_steps),
        }
