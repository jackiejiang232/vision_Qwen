"""高层候选的一致性检查。

该层只决定“候选是否可以交给执行器”，不会发布底盘、升降或关节命令。
它也是未来 VLA 输出进入系统的唯一门槛。
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ActionCandidate, ActionSkill, PickPlaceGoal, SceneState
from .scene import freshness


@dataclass(frozen=True)
class CandidateReport:
    accepted: bool
    reasons: tuple[str, ...]


def validate_candidate(
    scene: SceneState,
    goal: PickPlaceGoal,
    candidate: ActionCandidate,
    *,
    now: float,
    max_scene_age: float,
    path_found: bool,
    ik_feasible: bool,
    safety_chain_ready: bool,
) -> CandidateReport:
    """集中给出拒绝原因，避免任一层在失败时默默继续使用旧数据。"""

    reasons = []
    target = scene.object_by_id(goal.target_id)
    if target is None:
        reasons.append("target is absent from SceneState")
    elif target.color != goal.target_color:
        reasons.append("target color no longer matches PickPlaceGoal")

    report = freshness(scene, now, max_scene_age)
    if not report.fresh:
        reasons.extend(f"stale {name}" for name in report.stale_sources)
    if candidate.skill is not ActionSkill.PICK:
        reasons.append("minimal execution accepts only pick candidates")
    if candidate.target_id != goal.target_id:
        reasons.append("candidate target_id differs from PickPlaceGoal")
    if candidate.grasp_profile is not goal.grasp_profile:
        reasons.append("candidate grasp profile differs from PickPlaceGoal")
    if not path_found:
        reasons.append("static A* did not find an approach path")
    if not ik_feasible:
        reasons.append("dual-arm IK is not feasible or not verified")
    if not safety_chain_ready:
        reasons.append("official safety chain is not confirmed ready")
    return CandidateReport(not reasons, tuple(reasons))
