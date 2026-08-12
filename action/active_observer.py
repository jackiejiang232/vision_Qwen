import time

from std_msgs.msg import Float64MultiArray


def clamp(value, low, high):
    return max(float(low), min(float(high), float(value)))


class ActiveObserver:
    """
    到达导航点后，通过头部倾仰/腰部升降主动寻找目标。

    注意：
    1. 腰部话题已在 keyboard_teleop.py 中验证：
       /spine_forward_position_controller/commands
    2. 头部话题需要你在官方镜像里确认。
       如果头部话题不存在，可以先关闭 enable_head_observe。
    """

    def __init__(self, node, config):
        self.node = node
        self.config = config

        self.head_pub = node.create_publisher(
            Float64MultiArray,
            config.head_command_topic,
            10,
        )

        self.spine_pub = node.create_publisher(
            Float64MultiArray,
            config.spine_command_topic,
            10,
        )

        self.sequence = []
        self.index = 0
        self.pose_start_time = 0.0
        self.active = False
        self.current_view_name = None
        self.current_surface = None
        self.current_mode = None
        self.last_commanded_spine = None
        self.last_command = None
        self.last_command_time = 0.0

    def expected_table_spine(self, target):
        pose = (target or {}).get("pose_world") or {}
        if "z" not in pose:
            return 0.0

        return clamp(
            float(self.config.table_pregrasp_z0) - float(pose["z"]),
            self.config.table_pregrasp_spine_min,
            self.config.table_pregrasp_spine_max,
        )

    def _table_approach_sequence(self, target):
        sequence = []
        for name, head_yaw, head_pitch, spine, hold_sec in (
            self.config.table_observe_sequence
        ):
            sequence.append(
                (
                    name,
                    head_yaw,
                    head_pitch,
                    spine,
                    hold_sec,
                )
            )
        return sequence

    def start(self, surface, resume=False, mode="approach", target=None):
        if surface == "shelf" and mode == "search":
            sequence = list(
                self.config.shelf_search_observe_sequence
            )
        elif surface == "shelf":
            sequence = list(
                self.config.shelf_observe_sequence
            )
        elif mode == "search":
            sequence = list(
                self.config.table_search_observe_sequence
            )
        else:
            sequence = self._table_approach_sequence(target)

        same_sequence = (
            surface == self.current_surface
            and mode == self.current_mode
        )
        self.sequence = sequence
        self.current_surface = surface
        self.current_mode = mode
        self.index = 0

        if (
            resume
            and same_sequence
            and self.last_commanded_spine is not None
        ):
            self.index = min(
                range(len(sequence)),
                key=lambda index: abs(
                    float(sequence[index][3])
                    - float(self.last_commanded_spine)
                ),
            )

        self.pose_start_time = time.monotonic()
        self.active = True
        self.current_view_name = None

    def stop(self):
        self.active = False
        self.current_view_name = None

    def step(self):
        """
        返回 True 表示整套观察动作已经扫完。
        返回 False 表示还在观察。
        """
        if not self.active:
            return True

        if self.index >= len(self.sequence):
            if self.config.repeat_observe_until_timeout:
                self.index = 0
                self.pose_start_time = time.monotonic()
                return False

            self.stop()
            return True

        name, head_yaw, head_pitch, spine, hold_sec = (
            self.sequence[self.index]
        )

        self.current_view_name = name
        self.last_commanded_spine = float(spine)

        self.publish_observe_pose(
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            spine=spine,
        )

        elapsed = time.monotonic() - self.pose_start_time
        if elapsed >= hold_sec:
            self.index += 1
            self.pose_start_time = time.monotonic()

        return False

    def publish_observe_pose(
        self,
        head_yaw,
        head_pitch,
        spine,
    ):
        now = time.monotonic()
        command = (
            round(float(head_yaw), 4),
            round(float(head_pitch), 4),
            round(float(spine), 4),
        )
        if (
            command == self.last_command
            and now - self.last_command_time < 0.5
        ):
            return
        self.last_command = command
        self.last_command_time = now

        if self.config.enable_head_observe:
            head_msg = Float64MultiArray()
            head_msg.data = [
                float(head_yaw),
                float(head_pitch),
            ]
            self.head_pub.publish(head_msg)

        if self.config.enable_spine_observe:
            spine_msg = Float64MultiArray()
            spine_msg.data = [float(spine)]
            self.spine_pub.publish(spine_msg)
