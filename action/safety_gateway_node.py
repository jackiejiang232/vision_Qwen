#!/usr/bin/env python3
import rclpy

from third_party.dg202612.safety import SafetyGateway


def main():
    rclpy.init()
    node = SafetyGateway(auto_enable=False)
    try:
        rclpy.spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()