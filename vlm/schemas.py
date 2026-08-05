from typing import List, Optional

from pydantic import (
    BaseModel,
    Field,
)


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"


class InstructionItem(StrictModel):
    task_id: Optional[int] = None
    target_category: str
    target_color: Optional[str] = None
    source_location: Optional[str] = None
    destination_type: Optional[str] = None
    reference_object: Optional[str] = None
    spatial_relation: Optional[str] = None
    original_instruction: str

class VisualQueryObject(StrictModel):
    category: Optional[str] = None
    color: Optional[str] = None
    attributes: List[str] = Field(
        default_factory=list
    )
    query_phrases: List[str] = Field(
        default_factory=list
    )

class VisualQueryDestination(StrictModel):
    type: Optional[str] = None
    spatial_relation: Optional[str] = None

class VisualTaskQuery(StrictModel):
    task_id: int
    original_instruction: str = ""
    target_object: VisualQueryObject
    destination: Optional[VisualQueryDestination] = None
    reference_object: Optional[VisualQueryObject] = None
    context_objects: List[str] = Field(
        default_factory=list
    )

class DinoQueryTarget(StrictModel):
    category: str
    color: Optional[str] = None
    source_location: Optional[str] = None
    destination_location: Optional[str] = None
# class DinoQuery(StrictModel):
#     schema_version: str = "1.0"
#     original_instruction: str = ""
#     target: Optional[DinoQueryTarget] = None
#     target_prompts: List[str] = Field(
#         default_factory=list
#     )
#     context_prompts: List[str] = Field(
#         default_factory=list
#     )
#     grounding_prompt: str
# class DinoQuery(StrictModel):
#     schema_version: str = "1.0"
#     original_instruction: str = ""

#     target_object: Optional[VisualQueryObject] = None
#     destination: Optional[VisualQueryDestination] = None
#     reference_object: Optional[VisualQueryObject] = None
#     context_objects: List[str] = Field(
#         default_factory=list
#     )

#     target: Optional[DinoQueryTarget] = None
#     target_prompts: List[str] = Field(
#         default_factory=list
#     )
#     context_prompts: List[str] = Field(
#         default_factory=list
#     )

#     grounding_prompt: str
class DinoQuery(StrictModel):
    schema_version: str = "1.0"
    original_instruction: str = ""

    tasks: List[VisualTaskQuery] = Field(
        default_factory=list
    )

    # 兼容旧单任务字段
    target_object: Optional[VisualQueryObject] = None
    destination: Optional[VisualQueryDestination] = None
    reference_object: Optional[VisualQueryObject] = None
    context_objects: List[str] = Field(
        default_factory=list
    )

    target: Optional[DinoQueryTarget] = None
    target_prompts: List[str] = Field(
        default_factory=list
    )
    context_prompts: List[str] = Field(
        default_factory=list
    )

    grounding_prompt: str

class SceneObject(StrictModel):
    object_id: str
    label: str
    raw_label: Optional[str] = None
    corrected_label: Optional[str] = None
    estimated_color: Optional[str] = None
    color_consistent: Optional[bool] = None
    depth_span_m: Optional[float] = None
    support_surface: Optional[str] = None
    on_table: Optional[bool] = None
    on_shelf: Optional[bool] = None
    support_surface_index: Optional[int] = None
    shelf_layer: Optional[int] = None
    shelf_surface_z: Optional[float] = None
    shelf_layer_confidence: Optional[float] = None
    table_height_confidence: Optional[float] = None
    semantic_role: str
    location: str
    attributes: List[str] = Field(
        default_factory=list
    )
    relations: List[str] = Field(
        default_factory=list
    )
    confidence: float

    box_xyxy: Optional[List[float]] = None
    centroid_uv: Optional[List[float]] = None
    mask_area: Optional[int] = None
    dino_score: Optional[float] = None
    sam_score: Optional[float] = None
    pose_world: Optional[dict] = None
    size_3d: Optional[dict] = None


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
class TaskTargetBinding(StrictModel):
    object_id: Optional[str] = None
    label: Optional[str] = None
    category: Optional[str] = None
    color: Optional[str] = None
    box_xyxy: Optional[List[float]] = None
    centroid_uv: Optional[List[float]] = None
    mask_area: Optional[int] = None
    pose_world: Optional[dict] = None
    size_3d: Optional[dict] = None
    support_surface: Optional[str] = None
    on_table: Optional[bool] = None
    on_shelf: Optional[bool] = None
    shelf_layer: Optional[int] = None
    shelf_layer_confidence: Optional[float] = None
    confidence: float = 0.0
    requires_reobserve: bool = False


class PlaceGoalBinding(StrictModel):
    type: Optional[str] = None
    reference_object_id: Optional[str] = None
    reference_label: Optional[str] = None
    spatial_relation: Optional[str] = None
    pose_world: Optional[dict] = None
    support_surface: Optional[str] = None
    on_table: Optional[bool] = None
    on_shelf: Optional[bool] = None
    shelf_layer: Optional[int] = None
    shelf_layer_confidence: Optional[float] = None
    requires_planning: bool = True
    requires_scene_memory: bool = False


class ActionTaskItem(StrictModel):
    task_id: int
    status: str = "pending"
    original_instruction: str = ""
    target: TaskTargetBinding
    place_goal: PlaceGoalBinding
    uncertainties: List[str] = Field(
        default_factory=list
    )


class ExecutionPolicy(StrictModel):
    order: str = "ascending_task_id"
    allow_parallel: bool = False
    require_action_feedback: bool = True

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

    task_queue: List[ActionTaskItem] = Field(
        default_factory=list
    )
    active_task_id: Optional[int] = None
    execution_policy: ExecutionPolicy = Field(
        default_factory=ExecutionPolicy
    )
# class VLMSceneUnderstanding(StrictModel):
#     schema_version: str = "1.0"
#     source_stamp_sec: int
#     source_stamp_nanosec: int
#     scene_summary: str
#     instruction_understanding: List[
#         InstructionItem
#     ]
#     objects: List[SceneObject]
#     grounding: GroundingDecision
#     uncertainties: List[str] = Field(
#         default_factory=list
#     )
#     future_action: FutureActionSlot