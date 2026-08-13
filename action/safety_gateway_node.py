#!/usr/bin/env python3
import rclpy

from action.action_config import CONFIG
from third_party.dg202612.safety import SafetyGateway


def main():
    rclpy.init()
    # 整合动作模式下，导航总动作节点直接管理头部/腰部；安全网关
    # 只保留底盘急停能力，避免旧的控制请求重新抢占头腰设定点。
    owned_position_axes = (
        frozenset()
        if bool(getattr(CONFIG, "integrated_action_mode", True))
        else None
    )
    if owned_position_axes is None:
        node = SafetyGateway(auto_enable=False)
    else:
        node = SafetyGateway(
            auto_enable=False,
            owned_position_axes=owned_position_axes,
        )
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
