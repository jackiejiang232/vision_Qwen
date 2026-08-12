from geometry_msgs.msg import Twist
import math

from .scene_reader import get_servo_vertical_target


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

    def _box_is_usable(self, box_xyxy, target=None):
        if not box_xyxy or len(box_xyxy) < 4:
            return False

        x1, y1, x2, y2 = [float(value) for value in box_xyxy]
        margin = float(self.config.servo_bbox_margin_px)
        bottom_margin = margin
        target = target or {}
        if (
            target.get("support_surface") == "table"
            or target.get("on_table")
        ):
            bottom_margin = float(
                self.config.servo_table_bottom_margin_px
            )

        return (
            x1 >= margin
            and y1 >= margin
            and x2 <= self.config.image_width - margin
            and y2 <= self.config.image_height - bottom_margin
            and x2 - x1 >= self.config.servo_min_bbox_width_px
            and y2 - y1 >= self.config.servo_min_bbox_height_px
        )

    def _box_has_signal(self, box_xyxy):
        if not box_xyxy or len(box_xyxy) < 4:
            return False

        x1, y1, x2, y2 = [float(value) for value in box_xyxy]
        width = x2 - x1
        height = y2 - y1
        touches_border = (
            x1 <= 1.0
            or y1 <= 1.0
            or x2 >= self.config.image_width - 1.0
            or y2 >= self.config.image_height - 1.0
        )

        if touches_border:
            min_width = self.config.servo_edge_min_bbox_width_px
            min_height = self.config.servo_edge_min_bbox_height_px
        else:
            min_width = self.config.servo_min_bbox_width_px
            min_height = self.config.servo_min_bbox_height_px

        return (
            x2 > 0.0
            and y2 > 0.0
            and x1 < self.config.image_width
            and y1 < self.config.image_height
            and width >= min_width
            and height >= min_height
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

    def _is_table_target(self, target):
        return not (
            target.get("support_surface") == "shelf"
            or bool(target.get("on_shelf"))
        )

    def _is_shelf_target(self, target):
        return not self._is_table_target(target)

    def _yaw_error_to_target(self, target, robot_pose):
        pose_world = target.get("pose_world") or {}
        if robot_pose is None or not hasattr(robot_pose, "yaw"):
            return None
        if "x" not in pose_world or "y" not in pose_world:
            return None

        desired_yaw = math.atan2(
            float(pose_world["y"]) - float(robot_pose.y),
            float(pose_world["x"]) - float(robot_pose.x),
        )
        error = desired_yaw - float(robot_pose.yaw)
        return math.atan2(math.sin(error), math.cos(error))

    def _target_u(self, target):
        if self._is_table_target(target):
            return float(
                target.get("servo_target_u")
                or self.config.image_center_u
            )
        return float(self.config.image_center_u)

    def _is_table_dual_target(self, target):
        return (
            self._is_table_target(target)
            and target.get("selected_arm") == "dual"
        )

    def _table_linear_speed(self, distance_error):
        tolerance = float(self.config.servo_depth_tolerance)
        close_extra = float(
            getattr(self.config, "servo_table_close_extra_tolerance", 0.03)
        )

        if distance_error > tolerance:
            return clamp(
                float(getattr(self.config, "servo_table_linear_gain", 0.22))
                * distance_error,
                0.0,
                float(getattr(self.config, "servo_table_max_linear_speed", 0.055)),
            )

        if distance_error < -(tolerance + close_extra):
            return clamp(
                float(getattr(self.config, "servo_table_linear_gain", 0.22))
                * distance_error,
                -0.025,
                0.0,
            )

        return 0.0

    def ready_now(self, target, robot_pose):
        centroid_uv = target.get("centroid_uv")
        if not centroid_uv or len(centroid_uv) < 2:
            return False
        box_xyxy = target.get("box_xyxy")
        if self._is_table_dual_target(target):
            if not self._box_has_signal(box_xyxy):
                return False
        elif not self._box_is_usable(box_xyxy, target):
            return False

        u = float(centroid_uv[0])
        v = float(centroid_uv[1])
        target_v, v_tolerance = get_servo_vertical_target(
            target,
            self.config,
        )
        if abs(u - self._target_u(target)) > self.config.servo_u_tolerance:
            return False
        if self._is_shelf_target(target):
            yaw_error = self._yaw_error_to_target(target, robot_pose)
            if yaw_error is None:
                return False
            if abs(yaw_error) > float(
                getattr(
                    self.config,
                    "servo_shelf_yaw_tolerance_rad",
                    0.16,
                )
            ):
                return False
        # 桌边双臂抓取的底盘伺服只负责横向和距离；垂直视角由
        # pregrasp_adjust 里的头部/腰部闭环处理，避免在这里卡住不降腰。
        # 货架抓取时，相机常会在货架近距离看到目标偏下/偏上；
        # 底盘视觉伺服只能调横向和距离，不能可靠地把 v 拉回中心。
        # 因此 shelf 目标不把垂直像素误差作为 ready 硬条件。
        if (
            not self._is_table_dual_target(target)
            and not self._is_shelf_target(target)
            and abs(v - target_v) > v_tolerance
        ):
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
        box_xyxy = target.get("box_xyxy")
        box_is_usable = self._box_is_usable(box_xyxy, target)
        if not box_is_usable and not self._box_has_signal(box_xyxy):
            self.stop()
            return ServoState.REOBSERVE

        u = float(centroid_uv[0])
        v = float(centroid_uv[1])
        u_error = u - self._target_u(target)
        msg = Twist()
        target_v, v_tolerance = get_servo_vertical_target(
            target,
            self.config,
        )

        table_dual = self._is_table_dual_target(target)
        distance = self._planar_distance(target, robot_pose)
        desired_distance = self._desired_distance(target)
        distance_error = (
            None
            if distance is None
            else distance - desired_distance
        )

        if self._is_shelf_target(target):
            yaw_error = self._yaw_error_to_target(target, robot_pose)
            yaw_tolerance = float(
                getattr(
                    self.config,
                    "servo_shelf_yaw_tolerance_rad",
                    0.16,
                )
            )
            if yaw_error is None:
                self.stop()
                return ServoState.REOBSERVE
            if abs(yaw_error) > yaw_tolerance:
                self.reset()
                msg.angular.z = clamp(
                    float(
                        getattr(
                            self.config,
                            "servo_shelf_yaw_gain",
                            0.55,
                        )
                    )
                    * yaw_error,
                    -float(
                        getattr(
                            self.config,
                            "servo_shelf_max_angular_speed",
                            self.config.servo_max_angular_speed,
                        )
                    ),
                    float(
                        getattr(
                            self.config,
                            "servo_shelf_max_angular_speed",
                            self.config.servo_max_angular_speed,
                        )
                    ),
                )
                self.publisher.publish(msg)
                return ServoState.MOVING

        # 目标已经出现在画面边缘时，不要退回观察状态。
        # 桌边双臂抓取时，是否后退要由世界距离决定，避免画面裁切和距离闭环打架。
        if v > self.config.servo_max_v:
            self.reset()
            if table_dual and distance_error is not None:
                msg.linear.x = min(0.0, self._table_linear_speed(distance_error))
            else:
                msg.linear.x = (
                    -0.05 if self._is_table_target(target) else 0.0
                )
            msg.angular.z = clamp(
                -0.0008 * u_error,
                -0.05,
                0.05,
            )
            self.publisher.publish(msg)
            return ServoState.MOVING

        if v < self.config.servo_min_v:
            self.reset()
            if table_dual and distance_error is not None:
                msg.linear.x = max(0.0, self._table_linear_speed(distance_error))
            else:
                msg.linear.x = 0.04
            msg.angular.z = clamp(
                -0.0008 * u_error,
                -0.05,
                0.05,
            )
            self.publisher.publish(msg)
            return ServoState.MOVING

        # 桌边双臂抓取接近后，目标常会落在画面偏下位置。
        # 这里不再用普通 v 误差阻塞 ALIGNED，后续 pregrasp_adjust 会调头和腰。
        if (
            abs(v - target_v) > v_tolerance
            and not table_dual
            and not self._is_shelf_target(target)
        ):
            if self._is_table_target(target):
                self.reset()
                v_error = v - target_v
                if table_dual and distance_error is not None:
                    msg.linear.x = self._table_linear_speed(distance_error)
                else:
                    msg.linear.x = clamp(
                        -0.0012 * v_error,
                        -0.06,
                        0.05,
                    )
                msg.angular.z = clamp(
                    -0.0010 * u_error,
                    -0.06,
                    0.06,
                )
                self.publisher.publish(msg)
                return ServoState.MOVING

            self.stop()
            return ServoState.REOBSERVE

        if distance is None or distance_error is None:
            self.stop()
            return ServoState.REOBSERVE

        if not box_is_usable:
            if table_dual:
                linear_speed = self._table_linear_speed(distance_error)
                if (
                    abs(linear_speed) <= 1e-6
                    and abs(u_error) <= self.config.servo_u_tolerance
                ):
                    box_is_usable = True
                else:
                    self.reset()
                    msg.angular.z = clamp(
                        -0.00035 * u_error,
                        -0.025,
                        0.025,
                    )
                    if abs(u_error) <= self.config.servo_u_tolerance:
                        msg.linear.x = linear_speed
                    self.publisher.publish(msg)
                    return ServoState.MOVING
            else:
                self.reset()
                msg.angular.z = clamp(
                    -0.0012 * u_error,
                    -self.config.servo_max_angular_speed,
                    self.config.servo_max_angular_speed,
                )
                if abs(u_error) <= self.config.servo_u_tolerance:
                    msg.linear.x = -0.04
                self.publisher.publish(msg)
                return ServoState.MOVING

        if abs(u_error) > self.config.servo_u_tolerance:
            self.reset()

            if table_dual:
                msg.angular.z = clamp(
                    -0.00035 * u_error,
                    -0.035,
                    0.035,
                )
            else:
                msg.angular.z = clamp(
                    -0.0010 * u_error,
                    -self.config.servo_max_angular_speed,
                    self.config.servo_max_angular_speed,
                )

            self.publisher.publish(msg)
            return ServoState.MOVING

        if table_dual:
            linear_speed = self._table_linear_speed(distance_error)
            if abs(linear_speed) > 1e-6:
                self.reset()
                msg.linear.x = linear_speed
                msg.angular.z = clamp(
                    -0.00035 * u_error,
                    -0.025,
                    0.025,
                )
                self.publisher.publish(msg)
                return ServoState.MOVING

        elif abs(distance_error) > self.config.servo_depth_tolerance:
            self.reset()
            msg.linear.x = clamp(
                0.45 * distance_error,
                -self.config.servo_max_linear_speed,
                self.config.servo_max_linear_speed,
            )

            if table_dual:
                msg.angular.z = clamp(
                    -0.00035 * u_error,
                    -0.025,
                    0.025,
                )
            else:
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
