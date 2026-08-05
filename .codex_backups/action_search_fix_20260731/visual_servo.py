from geometry_msgs.msg import Twist


def clamp(value, low, high):
    return max(low, min(high, value))


class ServoState:
    MOVING = "moving"
    STABILIZING = "stabilizing"
    ALIGNED = "aligned"
    REOBSERVE = "reobserve"


class VisualServo:
    def __init__(self, publisher, config):
        self.publisher = publisher
        self.config = config
        self.stable_frames = 0
        self.last_observation_id = None

    def reset(self):
        self.stable_frames = 0
        self.last_observation_id = None

    def stop(self, reset_stability=True):
        self.publisher.publish(Twist())
        if reset_stability:
            self.reset()

    def _box_is_usable(self, box_xyxy):
        if not box_xyxy or len(box_xyxy) < 4:
            return False

        x1, y1, x2, y2 = [float(value) for value in box_xyxy]
        margin = float(self.config.servo_bbox_margin_px)

        return (
            x1 >= margin
            and y1 >= margin
            and x2 <= self.config.image_width - margin
            and y2 <= self.config.image_height - margin
            and x2 - x1 >= self.config.servo_min_bbox_width_px
            and y2 - y1 >= self.config.servo_min_bbox_height_px
        )

    def _planar_distance(self, target, robot_pose):
        pose_world = target.get("pose_world") or {}
        if robot_pose is None:
            return None
        if "x" not in pose_world or "y" not in pose_world:
            return None

        return (
            (float(pose_world["x"]) - float(robot_pose.x)) ** 2
            + (float(pose_world["y"]) - float(robot_pose.y)) ** 2
        ) ** 0.5

    def _desired_distance(self, target):
        if target.get("support_surface") == "shelf" or target.get("on_shelf"):
            return self.config.servo_target_distance_shelf
        return self.config.servo_target_distance_table

    def ready_now(self, target, robot_pose):
        centroid_uv = target.get("centroid_uv")
        if not centroid_uv or len(centroid_uv) < 2:
            return False
        if not self._box_is_usable(target.get("box_xyxy")):
            return False

        u = float(centroid_uv[0])
        v = float(centroid_uv[1])
        if abs(u - self.config.image_center_u) > self.config.servo_u_tolerance:
            return False
        if abs(v - self.config.image_center_v) > self.config.servo_v_tolerance:
            return False

        distance = self._planar_distance(target, robot_pose)
        if distance is None:
            return False

        return abs(
            distance - self._desired_distance(target)
        ) <= self.config.servo_depth_tolerance

    def step_from_target(self, target, robot_pose, observation_id=None):
        centroid_uv = target.get("centroid_uv")
        pose_world = target.get("pose_world")

        if not centroid_uv or not pose_world:
            self.stop()
            return ServoState.REOBSERVE
        if not self._box_is_usable(target.get("box_xyxy")):
            self.stop()
            return ServoState.REOBSERVE

        u = float(centroid_uv[0])
        v = float(centroid_uv[1])
        if not (self.config.servo_min_v <= v <= self.config.servo_max_v):
            self.stop()
            return ServoState.REOBSERVE
        if abs(v - self.config.image_center_v) > self.config.servo_v_tolerance:
            self.stop()
            return ServoState.REOBSERVE

        distance = self._planar_distance(target, robot_pose)
        if distance is None:
            self.stop()
            return ServoState.REOBSERVE

        u_error = u - self.config.image_center_u
        distance_error = distance - self._desired_distance(target)
        msg = Twist()

        if abs(u_error) > self.config.servo_u_tolerance:
            self.reset()
            msg.angular.z = clamp(
                -0.002 * u_error,
                -0.12,
                0.12,
            )
            self.publisher.publish(msg)
            return ServoState.MOVING

        if abs(distance_error) > self.config.servo_depth_tolerance:
            self.reset()
            msg.linear.x = clamp(
                0.45 * distance_error,
                -0.08,
                0.08,
            )
            msg.angular.z = clamp(
                -0.0012 * u_error,
                -0.06,
                0.06,
            )
            self.publisher.publish(msg)
            return ServoState.MOVING

        self.publisher.publish(Twist())
        if observation_id != self.last_observation_id:
            self.last_observation_id = observation_id
            self.stable_frames += 1

        if self.stable_frames >= self.config.servo_stable_frames:
            return ServoState.ALIGNED

        return ServoState.STABILIZING
