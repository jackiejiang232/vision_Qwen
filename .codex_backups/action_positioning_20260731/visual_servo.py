from geometry_msgs.msg import Twist


def clamp(value, low, high):
    return max(low, min(high, value))


class VisualServo:
    def __init__(self, publisher, config):
        self.publisher = publisher
        self.config = config

    def stop(self):
        self.publisher.publish(Twist())

    def step_from_target(self, target):
        centroid_uv = target.get("centroid_uv")
        pose_world = target.get("pose_world")

        if not centroid_uv or not pose_world:
            self.stop()
            return False

        u = float(centroid_uv[0])
        v = float(centroid_uv[1])

        u_error = u - self.config.image_center_u

        # 垂直方向如果太偏，底盘视觉伺服解决不了。
        # 这种情况交给 ACTIVE_OBSERVE 继续低头/降腰重新观察。
        if not (
            self.config.servo_min_v
            <= v
            <= self.config.servo_max_v
        ):
            self.stop()
            return False

        msg = Twist()

        if abs(u_error) > self.config.servo_u_tolerance:
            msg.angular.z = clamp(
                -0.002 * u_error,
                -0.12,
                0.12,
            )
            self.publisher.publish(msg)
            return False

        self.stop()
        return True