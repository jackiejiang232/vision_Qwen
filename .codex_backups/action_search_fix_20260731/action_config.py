from dataclasses import dataclass


@dataclass(frozen=True)
class ActionConfig:
    scene_topic: str = "/vlm/scene_understanding"
    detections_topic: str = "/grounded_sam/detections"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/slamware_ros_sdk_server_node/odom"
    ready_topic: str = "/action/ready_for_grasp"

    control_rate_hz: float = 20.0

    max_linear_speed: float = 0.15
    max_angular_speed: float = 0.35

    goal_xy_tolerance: float = 0.08
    goal_yaw_tolerance: float = 0.12

    # 先导航到观察/预抓取位姿，不直接贴到物体坐标。
    approach_distance_table: float = 0.75
    approach_distance_shelf: float = 0.80

    search_timeout_sec: float = 8.0
    reobserve_wait_sec: float = 2.0

    image_width: float = 640.0
    image_height: float = 480.0
    image_center_u: float = 320.0
    image_center_v: float = 240.0
    servo_min_v: float = 120.0
    servo_max_v: float = 390.0
    servo_u_tolerance: float = 28.0
    servo_v_tolerance: float = 55.0
    servo_bbox_margin_px: float = 20.0
    servo_min_bbox_width_px: float = 35.0
    servo_min_bbox_height_px: float = 35.0
    servo_target_distance_table: float = 0.65
    servo_target_distance_shelf: float = 0.70
    servo_depth_tolerance: float = 0.08
    servo_stable_frames: int = 3

    servo_scene_max_age_sec: float = 2.5
    visual_servo_timeout_sec: float = 20.0
    
    head_command_topic: str = "/head_forward_position_controller/commands"
    spine_command_topic: str = "/spine_forward_position_controller/commands"

    # 当前官方镜像未验证头部控制接口，先只使用已验证的腰部接口。
    enable_head_observe: bool = False
    enable_spine_observe: bool = True

    observe_fresh_timeout_sec: float = 1.5
    active_observe_timeout_sec: float = 45.0
    repeat_observe_until_timeout: bool = True

    table_observe_sequence: tuple = (
        ("table_drop_1", 0.0, -0.25, 0.12, 2.5),
        ("table_drop_2", 0.0, -0.35, 0.22, 2.5),
        ("table_drop_3", 0.0, -0.45, 0.30, 2.0),
        ("table_lower_center", 0.0, -0.50, 0.36, 2.5),
    )

    shelf_observe_sequence: tuple = (
        ("shelf_center", 0.0, -0.10, 0.0, 1.5),
        ("shelf_left", 0.25, -0.10, 0.0, 1.5),
        ("shelf_right", -0.25, -0.10, 0.0, 1.5),
        ("shelf_lower", 0.0, -0.25, 0.08, 1.5),
    )

CONFIG = ActionConfig()