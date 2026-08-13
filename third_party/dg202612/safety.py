"""控制限幅、数据超时、人工解锁与急停。

任务节点只发布内部控制请求；本模块检查请求后才发布赛事规定的控制话题。
开发模式由审核过的单动作计划显式解锁，比赛模式在四类输入全部新鲜后自动解锁。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Sequence

from .common import ControlCommand, RunLogger


class UnsafeCommand(ValueError):
    """控制值超出机器人允许范围。"""


@dataclass(frozen=True)
class AxisLimit:
    minimum: float
    maximum: float
    rate: float


ARM_LIMITS = (
    AxisLimit(-3.151, 2.089, 1.5),
    AxisLimit(-2.963, 0.181, 1.5),
    AxisLimit(-0.094, 3.161, 1.5),
    AxisLimit(-3.012, 3.012, 2.0),
    AxisLimit(-1.859, 1.859, 2.0),
    AxisLimit(-3.017, 3.017, 2.0),
)
CONTROL_JOINT_NAMES = (
    "slide_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    *(f"left_arm_joint{index}" for index in range(1, 7)),
    "left_arm_eef_gripper_joint",
    *(f"right_arm_joint{index}" for index in range(1, 7)),
    "right_arm_eef_gripper_joint",
)

CONTROL_LIMITS = (
    AxisLimit(-0.45, 0.45, 0.8),
    AxisLimit(-1.20, 1.20, 5.0),
    AxisLimit(-0.04, 0.87, 0.25),
    AxisLimit(-0.50, 0.50, 1.0),
    AxisLimit(-1.18, 0.16, 1.0),
    *ARM_LIMITS,
    AxisLimit(0.0, 1.0, 4.0),
    *ARM_LIMITS,
    AxisLimit(0.0, 1.0, 4.0),
)


class SafetyPolicy:
    """对 19 维目标做边界检查和每秒变化率限制。"""

    def __init__(self, limits: Sequence[AxisLimit] = CONTROL_LIMITS) -> None:
        if len(limits) != ControlCommand.SIZE:
            raise ValueError("safety policy must define exactly 19 axes")
        self.limits = tuple(limits)

    def validate(self, command: ControlCommand) -> None:
        for index, (value, limit) in enumerate(zip(command.values, self.limits)):
            if not math.isfinite(value):
                raise UnsafeCommand(f"axis {index} is not finite")
            if value < limit.minimum or value > limit.maximum:
                raise UnsafeCommand(
                    f"axis {index}={value:.6f} outside "
                    f"[{limit.minimum:.6f}, {limit.maximum:.6f}]"
                )

    def rate_limit(
        self,
        previous: ControlCommand | None,
        requested: ControlCommand,
        elapsed: float,
    ) -> ControlCommand:
        self.validate(requested)
        if previous is None:
            return requested
        dt = max(0.0, min(float(elapsed), 0.2))
        values = []
        for old, new, limit in zip(previous.values, requested.values, self.limits):
            delta = limit.rate * dt
            values.append(max(old - delta, min(old + delta, new)))
        return ControlCommand.from_values(values, requested.created_at)


def stopped(
    command: ControlCommand | None, created_at: float
) -> ControlCommand | None:
    """底盘归零，升降和机械臂保持最后安全目标。"""

    if command is None:
        return None
    values = list(command.values)
    values[0:2] = [0.0, 0.0]
    return ControlCommand.from_values(values, created_at)


# 四个位置控制器各自负责 19 维里的哪一段。底盘（第 0、1 维）不在其中，因为它是
# **速度**量，语义和位置量完全不同——见 position_group_payloads 的说明。
POSITION_GROUP_SLICES: dict[str, slice] = {
    "spine": slice(2, 3),
    "head": slice(3, 5),
    "left_arm": slice(5, 12),
    "right_arm": slice(12, 19),
}
ALL_POSITION_GROUPS = frozenset(POSITION_GROUP_SLICES)


def position_group_payloads(
    command: ControlCommand,
    owned: frozenset[str] | set[str],
) -> dict[str, list[float]]:
    """把 19 维命令切成各位置控制器的载荷，**只保留本次动作真正驱动的轴组**。

    为什么需要"轴组归属"这个概念
    ----------------------------
    位置控制器和速度控制器对"不发布"的解释是相反的：

    - ``/cmd_vel`` 是**速度**。不发布 = 上一条速度继续生效 = 车继续走。所以底盘
      必须每周期显式发布，"停"要写成显式的 0。
    - 四个 ``*_forward_position_controller`` 是**位置设定点**。不发布 = 设定点
      不变 = 关节停在原地。**"不发布"才是真正的"保持不动"。**

    早前三个单动作都走另一条路：把不归自己管的轴按**实测当前值**再发一遍，本意是
    "锁住别动"。但对带重力负载的位置控制器，实测值 ≠ 设定点——MuJoCo 的位置执行器
    稳态满足 ``qpos = setpoint + 负载/kp``，中间差的就是**重力下垂量**。把已经下垂
    过的实测值当成新设定点发回去，等于让它在原来的基础上**再下垂一次**。

    2026-07-31 实测（连做两次 LOOK，除头部外全部"锁在实测值"）：

    ==================  ========  ========  ========  ==========
    轴                  执行前    第 1 次后 第 2 次后 每次增量
    ==================  ========  ========  ========  ==========
    ``slide_joint``     0.006063  0.012126  0.018188  **+6.06 mm**
    ``left_arm_joint5`` 0.002850  0.005700  0.008555  +0.163°
    左夹爪              0.003370  0.004720  0.006072  +1.35 mm
    ==================  ========  ========  ========  ==========

    完全线性，每次一个下垂步长，不收敛。``slide`` 越大躯干越低，所以这是**躯干每
    做一个动作就沉 6 mm**：做十次前置观测就沉 6 cm，足以让标定好的抓取高度全错。
    而它不报错、不越限、不触发任何检查——``joint_states`` 一直"正常"。

    为什么不用"减去下垂量"来补偿
    ----------------------------
    下垂量 = 负载/kp，负载随手臂姿态和是否夹着箱子而变，不是常数。写死一个 6.06 mm
    只在"开局空载"这一个工况成立，换个姿态就又错了——而且那正是用户明确禁止的
    "写死躯干高度常量"。按轴组归属只发自己那一段，不需要任何标定常数，也天然跟着
    负载变化走。

    参数
    ----
    command:
        完整的 19 维命令。即使某些轴组不发布，也仍然要求命令是完整且合法的——
        限幅、速率限制和记录都基于它，这样"计划里写了什么"始终可查。
    owned:
        本次动作拥有的轴组名，取值必须是 :data:`ALL_POSITION_GROUPS` 的子集。
        空集合是合法的（例如 DOCK 只驱动底盘，一个位置控制器都不碰）。

    返回
    ----
    ``{轴组名: 该控制器的载荷}``，只含 ``owned`` 里的轴组。
    """

    unknown = set(owned) - ALL_POSITION_GROUPS
    if unknown:
        raise ValueError(
            f"unknown position group(s): {sorted(unknown)}; "
            f"expected a subset of {sorted(ALL_POSITION_GROUPS)}"
        )
    values = command.values
    return {
        name: list(values[POSITION_GROUP_SLICES[name]])
        for name in POSITION_GROUP_SLICES
        if name in owned
    }


def measured_control(
    names: Sequence[str],
    positions: Sequence[float],
    created_at: float | None = None,
) -> ControlCommand | None:
    """把完整关节反馈按 19 维控制顺序排列；缺少关节时等待下一帧。"""

    joints = {
        name: float(positions[index])
        for index, name in enumerate(names)
        if index < len(positions)
    }
    if not all(name in joints for name in CONTROL_JOINT_NAMES):
        return None
    values = [0.0, 0.0, *(joints[name] for name in CONTROL_JOINT_NAMES)]
    return ControlCommand.from_values(values, created_at)


# ROS 只存在于官方镜像。纯逻辑测试仍可导入本模块中的 SafetyPolicy。
try:
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, Header, String
    from std_srvs.srv import SetBool

    ROS_AVAILABLE = True
except ModuleNotFoundError:
    ROS_AVAILABLE = False

    class Node:  # type: ignore[no-redef]
        pass


class SafetyGateway(Node):
    """正式控制话题的唯一发布者。"""

    TIMEOUT = 0.5
    RATE_HZ = 50.0

    def __init__(
        self,
        auto_enable: bool = False,
        run_dir: str | os.PathLike[str] | None = None,
        owned_position_axes: frozenset[str] | set[str] = ALL_POSITION_GROUPS,
    ) -> None:
        if not ROS_AVAILABLE:
            raise RuntimeError("SafetyGateway must run inside the official ROS image")
        super().__init__("dg202612_safety_gateway")
        self.policy = SafetyPolicy()
        self.runlog = RunLogger(run_dir)
        self.auto_enable = bool(auto_enable)
        # 本次动作真正驱动哪些位置控制器。不在这个集合里的轴组一条消息都不发，
        # 于是它们的设定点保持不变——对位置控制器来说这才是真正的"保持不动"。
        # 理由和实测数据见 position_group_payloads 的文档。默认全拥有，保持旧行为。
        self.owned_position_axes = frozenset(owned_position_axes)
        unknown = self.owned_position_axes - ALL_POSITION_GROUPS
        if unknown:
            raise ValueError(f"unknown position group(s): {sorted(unknown)}")
        self.enabled = False
        self.emergency_stop = False
        self.last_error: str | None = None
        self.last_request: ControlCommand | None = None
        self.last_measured: ControlCommand | None = None
        self.last_output: ControlCommand | None = None
        self.last_request_rx = 0.0
        self.last_heartbeat = 0.0
        self.last_joint_state = 0.0
        self.last_odom = 0.0
        self.control_output_cycles = 0
        self.safe_output_cycles = 0

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        # 未拥有的轴组不创建 publisher。这样整合动作模式下，ROS 图上也只
        # 会留下总动作节点的头腰发布者，而不是“看似不发、实际仍可能抢占”的
        # 第二个位置控制来源。
        self.spine_pub = (
            self.create_publisher(
                Float64MultiArray,
                "/spine_forward_position_controller/commands",
                5,
            )
            if "spine" in self.owned_position_axes
            else None
        )
        self.head_pub = (
            self.create_publisher(
                Float64MultiArray,
                "/head_forward_position_controller/commands",
                5,
            )
            if "head" in self.owned_position_axes
            else None
        )
        self.left_pub = (
            self.create_publisher(
                Float64MultiArray,
                "/left_arm_forward_position_controller/commands",
                5,
            )
            if "left_arm" in self.owned_position_axes
            else None
        )
        self.right_pub = (
            self.create_publisher(
                Float64MultiArray,
                "/right_arm_forward_position_controller/commands",
                5,
            )
            if "right_arm" in self.owned_position_axes
            else None
        )
        self.runtime_pub = self.create_publisher(
            String, "/dg202612/runtime_state", 10
        )

        self.create_subscription(
            Float64MultiArray,
            "/dg202612/control_request",
            self.on_request,
            10,
        )
        self.create_subscription(
            Header,
            "/dg202612/control_heartbeat",
            self.on_heartbeat,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self.on_joint, 10)
        self.create_subscription(
            Odometry,
            "/slamware_ros_sdk_server_node/odom",
            self.on_odom,
            10,
        )
        self.create_service(
            SetBool,
            "/dg202612/set_control_enabled",
            self.on_enable,
        )
        self.create_service(
            SetBool,
            "/dg202612/set_emergency_stop",
            self.on_estop,
        )
        self.create_timer(1.0 / self.RATE_HZ, self.tick)
        self.create_timer(0.2, self.publish_runtime)
        mode = "competition" if self.auto_enable else "development"
        self.runlog.event("safety_started", mode=mode, enabled=False)
        self.get_logger().info(f"Safety gateway started in {mode} mode; control locked")

    def on_request(self, message: Float64MultiArray) -> None:
        now = time.monotonic()
        try:
            requested = ControlCommand.from_values(message.data, now)
            elapsed = (
                0.0
                if self.last_request_rx <= 0.0
                else now - self.last_request_rx
            )
            self.last_request = self.policy.rate_limit(
                self.last_request,
                requested,
                elapsed,
            )
            self.last_request_rx = now
            self.last_error = None
        except (ValueError, UnsafeCommand) as exc:
            self._trip(f"invalid control request: {exc}")

    def on_heartbeat(self, _message: Header) -> None:
        self.last_heartbeat = time.monotonic()

    def on_joint(self, message: JointState) -> None:
        try:
            measured = measured_control(message.name, message.position)
        except ValueError as exc:
            self._trip(f"invalid joint feedback: {exc}")
            return
        if measured is not None:
            self.last_measured = measured
            self.last_joint_state = time.monotonic()

    def on_odom(self, _message: Odometry) -> None:
        self.last_odom = time.monotonic()

    def on_enable(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if request.data:
            if self.emergency_stop:
                response.success = False
                response.message = "请先复位急停"
                return response
            # 允许先开启网关、后启动具体动作。导航阶段由导航节点直接
            # 控制底盘/头腰，抓取开始后才会出现 control_request。
            stale = self.robot_feedback_stale_reason(time.monotonic())
            if stale is not None:
                response.success = False
                response.message = f"数据未就绪：{stale}"
                return response
        self.enabled = bool(request.data)
        self.last_error = None if self.enabled else "control disabled locally"
        self.runlog.event("control_enabled", enabled=self.enabled)
        response.success = True
        response.message = "控制已允许" if self.enabled else "控制已停止"
        return response

    def on_estop(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self.emergency_stop = bool(request.data)
        if self.emergency_stop:
            self.enabled = False
            self.last_error = "local emergency stop"
        else:
            self.last_error = None
        self.runlog.event("emergency_stop", active=self.emergency_stop)
        response.success = True
        response.message = "急停已触发" if self.emergency_stop else "急停已复位"
        return response

    def _trip(self, reason: str) -> None:
        if self.last_error != reason:
            self.get_logger().error(reason)
            self.runlog.event("safe_stop", reason=reason)
        self.enabled = False
        self.emergency_stop = True
        self.last_error = reason

    def trip(self, reason: str) -> None:
        """供同进程执行器触发不可恢复的单动作急停。"""

        self._trip(reason)

    def stale_reason(self, now: float) -> str | None:
        for label, stamp in (
            ("control request", self.last_request_rx),
            ("control heartbeat", self.last_heartbeat),
            ("joint_states", self.last_joint_state),
            ("odom", self.last_odom),
        ):
            if stamp <= 0.0 or now - stamp > self.TIMEOUT:
                return f"{label} stale for more than {self.TIMEOUT:.1f}s"
        return None

    def robot_feedback_stale_reason(self, now: float) -> str | None:
        for label, stamp in (
            ("joint_states", self.last_joint_state),
            ("odom", self.last_odom),
        ):
            if stamp <= 0.0 or now - stamp > self.TIMEOUT:
                return f"{label} stale for more than {self.TIMEOUT:.1f}s"
        return None

    def _publish(self, command: ControlCommand) -> None:
        # 底盘是速度量：不发 = 上一条速度继续生效，所以**每周期都必须显式发**，
        # "停"写成显式的 0。四个位置控制器相反：不发 = 设定点不变 = 停在原地，
        # 因此只发本次动作拥有的轴组，别的一条都不发。详见 position_group_payloads。
        values = command.values
        twist = Twist()
        twist.linear.x = values[0]
        twist.angular.z = values[1]
        self.cmd_vel_pub.publish(twist)
        publishers = {
            "spine": self.spine_pub,
            "head": self.head_pub,
            "left_arm": self.left_pub,
            "right_arm": self.right_pub,
        }
        for name, payload in position_group_payloads(
            command, self.owned_position_axes
        ).items():
            publisher = publishers[name]
            if publisher is not None:
                publisher.publish(Float64MultiArray(data=payload))

    def _publish_safe(self, now: float) -> None:
        # 位置控制器的语义是“不发布就保持当前设定点”。安全态不能把旧的
        # last_output 再发给头部/腰部，否则会与导航节点的观察目标争抢同一
        # 控制器，表现为头腰抽搐。安全状态只需要显式停止速度控制器。
        self.cmd_vel_pub.publish(Twist())

    def tick(self) -> None:
        now = time.monotonic()
        has_control_stream = (
            self.last_request is not None
            or self.last_request_rx > 0.0
            or self.last_heartbeat > 0.0
        )
        reason = (
            self.stale_reason(now)
            if has_control_stream
            else self.robot_feedback_stale_reason(now)
        )
        if self.auto_enable and not self.enabled and not self.emergency_stop:
            if reason is None:
                self.enabled = True
                self.runlog.event("control_auto_enabled")
                self.get_logger().info("Competition inputs ready; control enabled")
        if self.enabled and reason is not None:
            self._trip(reason)
        if self.enabled and not self.emergency_stop and self.last_request is not None:
            # 只有真正获得控制权后，整形过的请求才成为“已发布命令”。
            self.last_output = self.last_request
            self._publish(self.last_output)
            self.control_output_cycles += 1
        else:
            if self.emergency_stop or self.last_output is not None:
                self._publish_safe(now)
                self.safe_output_cycles += 1

    def publish_runtime(self) -> None:
        now = time.monotonic()
        state = {
            "safety_node": self.get_name(),
            "run_mode": os.environ.get("DG_RUN_MODE", "development"),
            "control_enabled": self.enabled,
            "emergency_stop": self.emergency_stop,
            "safe": not self.enabled or self.stale_reason(now) is None,
            "safety_error": self.last_error,
            "ages": {
                "request": None
                if self.last_request_rx <= 0
                else now - self.last_request_rx,
                "heartbeat": None
                if self.last_heartbeat <= 0
                else now - self.last_heartbeat,
                "joint_states": None
                if self.last_joint_state <= 0
                else now - self.last_joint_state,
                "odom": None if self.last_odom <= 0 else now - self.last_odom,
            },
        }
        self.runtime_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    def shutdown(self) -> None:
        now = time.monotonic()
        self.enabled = False
        self._publish_safe(now)
        self.runlog.event("safety_shutdown")
