import math
import time

from geometry_msgs.msg import Twist


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def clamp(value, low, high):
    return max(low, min(high, value))


class SimpleCmdVelNavigator:
    def __init__(self, publisher, config):
        self.publisher = publisher
        self.config = config
        self.active_goal_key = None
        self.position_latched = False
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0
        self.last_publish_time = time.monotonic()

    def stop(self):
        self.publisher.publish(Twist())
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0
        self.last_publish_time = time.monotonic()

    def _publish_smooth(self, msg):
        now = time.monotonic()
        nominal_dt = 1.0 / float(self.config.control_rate_hz)
        dt = max(nominal_dt, min(now - self.last_publish_time, 0.20))

        linear_delta = self.config.max_linear_accel * dt
        angular_delta = self.config.max_angular_accel * dt
        linear_x = clamp(
            float(msg.linear.x),
            self.last_linear_x - linear_delta,
            self.last_linear_x + linear_delta,
        )
        angular_z = clamp(
            float(msg.angular.z),
            self.last_angular_z - angular_delta,
            self.last_angular_z + angular_delta,
        )

        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.publisher.publish(msg)
        self.last_linear_x = linear_x
        self.last_angular_z = angular_z
        self.last_publish_time = now

    def distance_to_goal(self, robot_pose, goal_pose):
        return math.hypot(
            goal_pose.x - robot_pose.x,
            goal_pose.y - robot_pose.y,
        )

    def _prepare_goal(self, goal_pose):
        goal_key = (
            round(float(goal_pose.x), 4),
            round(float(goal_pose.y), 4),
            round(float(goal_pose.yaw), 4),
        )
        if goal_key != self.active_goal_key:
            self.active_goal_key = goal_key
            self.position_latched = False

    def _update_position_latch(self, distance):
        if distance <= self.config.goal_xy_tolerance:
            self.position_latched = True
        elif distance > 2.0 * self.config.goal_xy_tolerance:
            self.position_latched = False

    def reached(self, robot_pose, goal_pose):
        self._prepare_goal(goal_pose)
        distance = self.distance_to_goal(robot_pose, goal_pose)
        self._update_position_latch(distance)
        yaw_error = abs(
            normalize_angle(goal_pose.yaw - robot_pose.yaw)
        )

        return (
            self.position_latched
            and yaw_error < self.config.goal_yaw_tolerance
        )

    def step(self, robot_pose, goal_pose):
        self._prepare_goal(goal_pose)
        dx = goal_pose.x - robot_pose.x
        dy = goal_pose.y - robot_pose.y
        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - robot_pose.yaw)
        distance = math.hypot(dx, dy)
        self._update_position_latch(distance)

        msg = Twist()

        # Once position is reached, only align final yaw. The hysteresis
        # prevents odometry noise from switching back to translation.
        if self.position_latched:
            final_yaw_error = normalize_angle(
                goal_pose.yaw - robot_pose.yaw
            )
            if abs(final_yaw_error) > self.config.goal_yaw_tolerance:
                msg.angular.z = clamp(
                    1.8 * final_yaw_error,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed,
                )
                self._publish_smooth(msg)
                return

            self.stop()
            return

        # Turn toward the translation target before moving forward.
        if abs(yaw_error) > 0.25:
            msg.angular.z = clamp(
                2.2 * yaw_error,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
            self._publish_smooth(msg)
            return

        msg.linear.x = clamp(
            1.0 * distance,
            -self.config.max_linear_speed,
            self.config.max_linear_speed,
        )
        msg.angular.z = clamp(
            1.8 * yaw_error,
            -self.config.max_angular_speed,
            self.config.max_angular_speed,
        )
        self._publish_smooth(msg)
