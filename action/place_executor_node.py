#!/usr/bin/env python3
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .action_config import CONFIG


class PlaceExecutorNode(Node):
    def __init__(self):
        super().__init__("place_executor_node")
        self.config = CONFIG
        self.integrated_disabled = bool(
            getattr(self.config, "integrated_action_mode", False)
        )
        if self.integrated_disabled:
            self.get_logger().info(
                "integrated action mode: standalone place executor disabled"
            )
            return
        self.state = "WAIT_PLACE"
        self.current_command = None
        self.simulation_started_at = 0.0
        self.create_subscription(
            String,
            self.config.place_command_topic,
            self.on_place_command,
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self.config.place_status_topic,
            10,
        )
        self.create_timer(0.5, self.publish_heartbeat)
        self.get_logger().info("place executor started")

    def publish_status(self, event, **kwargs):
        payload = {
            "event": event,
            "state": self.state,
            "time": time.time(),
        }
        payload.update(kwargs)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)
        self.get_logger().info(msg.data)

    def on_place_command(self, message):
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                payload = {"command": str(payload).strip().lower()}
        except json.JSONDecodeError:
            payload = {"command": message.data.strip().lower()}

        command = str(payload.get("command") or "").lower()
        if command == "start":
            if self.state != "WAIT_PLACE":
                self.publish_status(
                    "place_start_ignored",
                    reason="place executor is already active",
                )
                return
            self.current_command = payload
            self.state = "PREPLACE"
            self.simulation_started_at = time.time()
            self.publish_status(
                "place_started",
                task_id=payload.get("task_id"),
                place_pose=payload.get("place_pose"),
            )
            if self.config.place_simulation_mode:
                self.publish_status(
                    "simulation_stage",
                    stage="preplace",
                    hint="automatic simulation; no manual place confirmation required",
                )
            return

        if command == "preplace_done" and self.state == "PREPLACE":
            self.state = "APPROACH_PLACE"
            self.publish_status("preplace_done")
            return

        if command == "release_done" and self.state in ("PREPLACE", "APPROACH_PLACE"):
            self.state = "RETREAT"
            self.publish_status("release_done")
            return

        if command == "retreat_done" and self.state == "RETREAT":
            self.state = "PLACE_DONE"
            self.publish_status("place_done")
            return

        if command == "abort":
            self.state = "SAFE_STOP"
            self.publish_status("place_aborted")
            return

        self.publish_status("place_command_ignored", command=command)

    def publish_heartbeat(self):
        if self.config.place_simulation_mode:
            self.advance_simulated_place()
        self.publish_status("heartbeat")

    def advance_simulated_place(self):
        if self.state not in ("PREPLACE", "APPROACH_PLACE", "RETREAT"):
            return
        if time.time() - self.simulation_started_at < float(
            self.config.place_simulation_stage_sec
        ):
            return

        if self.state == "PREPLACE":
            self.state = "APPROACH_PLACE"
            self.simulation_started_at = time.time()
            self.publish_status("simulation_stage", stage="release")
            return
        if self.state == "APPROACH_PLACE":
            self.state = "RETREAT"
            self.simulation_started_at = time.time()
            self.publish_status("simulation_stage", stage="retreat")
            return
        self.state = "PLACE_DONE"
        self.publish_status("place_done", simulation=True)
        # 让任务总控读取 place_done 后，下一条任务可以再次 start。
        self.current_command = None
        self.state = "WAIT_PLACE"


def main():
    rclpy.init()
    node = PlaceExecutorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
