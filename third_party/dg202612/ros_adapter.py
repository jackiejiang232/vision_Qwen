"""新具名控制目标到旧 19 维兼容接口的纯数据适配。

本文件不导入 ROS，也不发布任何话题。真正运行时，调用方应把返回值交给已有的
``SafetyGateway``；它仍是唯一能发布赛事控制话题的组件。
"""

from __future__ import annotations

from .common import ControlCommand
from .contracts import RobotTargets


def legacy_control_vector(targets: RobotTargets) -> tuple[float, ...]:
    """保持旧接口顺序：底盘 2 + 升降/头部 3 + 左臂 7 + 右臂 7。"""

    values = (
        targets.base_linear,
        targets.base_angular,
        targets.slide,
        targets.head_yaw,
        targets.head_pitch,
        *targets.left_arm,
        targets.left_gripper,
        *targets.right_arm,
        targets.right_gripper,
    )
    if len(values) != ControlCommand.SIZE:
        raise AssertionError("RobotTargets must map to exactly 19 legacy values")
    return values


def legacy_control_command(targets: RobotTargets, created_at: float | None = None) -> ControlCommand:
    """构造旧数据对象；调用它不等于允许控制或发送 ROS 消息。"""

    return ControlCommand.from_values(legacy_control_vector(targets), created_at)
