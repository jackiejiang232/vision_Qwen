from dataclasses import dataclass


@dataclass(frozen=True)
class ActionConfig:
    scene_topic: str = "/vlm/scene_understanding"
    detections_topic: str = "/grounded_sam/detections"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/slamware_ros_sdk_server_node/odom"
    ready_topic: str = "/action/ready_for_grasp"

    control_rate_hz: float = 20.0

    max_linear_speed: float = 0.7
    max_angular_speed: float = 1.5
    max_linear_accel: float = 0.80
    max_angular_accel: float = 2.20

    goal_xy_tolerance: float = 0.04
    goal_yaw_tolerance: float = 0.06

    # 先导航到观察/预抓取位姿，不直接贴到物体坐标。
    approach_distance_table: float = 0.68
    approach_distance_shelf: float = 0.85

    search_timeout_sec: float = 8.0
    reobserve_wait_sec: float = 2.0

    image_width: float = 640.0
    image_height: float = 480.0
    image_center_u: float = 320.0
    image_center_v: float = 240.0
    # table_front_yaw: float = 1.5707963267948966
    #     # 桌面目标候选接近方向：
    # # dx, dy 是“从目标指向机器人底盘预抓取点”的方向。
    # # 第三个数是惩罚项，越小越优先。
    # table_approach_candidates: tuple = (
    #     (0.0, -1.0, 0.0),   # 优先从桌子正前方接近
    #     (-1.0, 0.0, 0.35),  # 备选：从目标左侧接近
    #     (1.0, 0.0, 0.35),   # 备选：从目标右侧接近
    # )
    table_front_yaw: float = 1.5707963267948966

    # 桌面箱子默认必须从桌子正前方接近，适合双臂夹取
    table_approach_candidates: tuple = (
        (0.0, -1.0, 0.0),
    )

    # 桌面双臂抓取时，机器人身体朝向必须接近桌面正前方
    table_pregrasp_table_yaw_tolerance_rad: float = 0.15

    # 如果目标在桌面右侧，允许底盘目标 x 更靠右一些
    # 这个范围需要结合仿真安全距离调，先用这个测试
    table_pick_x_range: tuple = (-1.45, 0.45)

    # ready_for_grasp 前，目标必须在机器人身体中心线附近
    table_pregrasp_lateral_tolerance_m: float = 0.04
    table_pregrasp_heading_tolerance_rad: float = 0.12
    table_pregrasp_lateral_offsets: tuple = (0.0,)
    servo_target_u_table_offset_px: float = 0.0
    # 桌面抓取时目标不追求图像正中心，而是保持完整可见并略偏上。
    servo_target_v_table: float = 220.0
    servo_target_v_shelf: float = 240.0
    servo_min_v: float = 120.0
    servo_max_v: float = 390.0
    servo_u_tolerance: float = 25.0
    servo_v_tolerance_table: float = 55.0
    servo_v_tolerance_shelf: float = 55.0
    servo_bbox_margin_px: float = 20.0
    servo_table_bottom_margin_px: float = 42.0
    servo_min_bbox_width_px: float = 35.0
    servo_min_bbox_height_px: float = 35.0
    servo_edge_min_bbox_width_px: float = 28.0
    servo_edge_min_bbox_height_px: float = 12.0
    servo_target_distance_table: float = 0.68
    servo_target_distance_shelf: float = 0.76
    servo_depth_tolerance: float = 0.07
    servo_max_linear_speed: float = 0.16
    servo_max_angular_speed: float = 0.12
    servo_stable_frames: int = 3
    # 就绪后不再用每一帧检测撤销状态，只监控底盘是否真的移动。
    ready_latch_xy_tolerance: float = 0.08
    ready_latch_yaw_tolerance: float = 0.12

    servo_scene_max_age_sec: float = 2.5
    visual_servo_timeout_sec: float = 20.0
    servo_target_lock_ttl_sec: float = 3.5
    
    head_command_topic: str = "/head_forward_position_controller/commands"
    spine_command_topic: str = "/spine_forward_position_controller/commands"
    joint_state_topic: str = "/joint_states"
    spine_joint_name: str = "slide_joint"
    head_pitch_joint_name: str = "head_pitch_joint"
    spine_state_max_age_sec: float = 2.0
    head_state_max_age_sec: float = 2.0

    # 官方镜像中头部控制接口已验证可用。
    enable_head_observe: bool = True
    enable_spine_observe: bool = True

    table_pick_x_range: tuple = (-1.35, 0.18)
    table_pick_y_range: tuple = (1.42, 1.82)
    table_pregrasp_z0: float = 1.32163718
    table_pregrasp_spine_min: float = -0.04
    table_pregrasp_spine_max: float = 0.87
    table_pregrasp_spine_tolerance: float = 0.035
    table_pregrasp_head_pitch: float = -0.65
    table_pregrasp_head_tolerance: float = 0.08
    pregrasp_adjust_enable: bool = True
    pregrasp_adjust_timeout_sec: float = 8.0
    pregrasp_adjust_min_sec: float = 2.5
    pregrasp_adjust_stable_frames: int = 5
    pregrasp_lost_max_frames: int = 10
    pregrasp_spine_reference_z: float = 1.25 
    pregrasp_grasp_height_ratio: float = 0.5 #抓箱子高度的一半
    pregrasp_spine_lower_bias: float = 0.15 #让腰部比理论值再低一点(优先调)
    pregrasp_spine_min_delta: float = -0.02
    pregrasp_spine_max_delta: float = 0.3 #单次最多下降xx m等效量，避免一下子过低丢视野
    
    #如果腰部最后太高，减小 pregrasp_spine_reference_z,如果腰部最后太低，增大 pregrasp_spine_reference_z
    pregrasp_allow_timeout_ready: bool = False

    # 腰部每次小步下降，别一次降到底
    pregrasp_spine_step: float = 0.05

    # 从视觉伺服结束时的腰部位置继续下降多少。
    # 如果你测试发现方向反了，就把 0.12 改成 -0.12。
    pregrasp_spine_delta_after_servo: float = 0.3
    pregrasp_spine_target_tolerance: float = 0.025

    # 头部回正目标。-0.65 比较俯视，-0.30 更回正。
    pregrasp_final_head_pitch: float = -0.15
    pregrasp_head_pitch_tolerance: float = 0.05

    # 桌面箱子最终希望仍然正对图像中心，但位置略偏上
    pregrasp_target_u: float = 320.0
    pregrasp_target_v: float = 210.0
    pregrasp_u_tolerance: float = 70.0
    pregrasp_v_tolerance: float = 90.0

    # 头部回正范围：你现在 -0.65 是比较俯视，回正可先试 -0.35
    pregrasp_head_pitch_min: float = -0.70
    pregrasp_head_pitch_max: float = -0.15
    pregrasp_head_pitch_step: float = 0.1  #头部俯仰速度
    table_grasp_distance_min: float = 0.61
    table_grasp_distance_max: float = 0.76

    observe_fresh_timeout_sec: float = 1.5
    active_observe_timeout_sec: float = 14.0
    repeat_observe_until_timeout: bool = False
    search_turn_timeout_sec: float = 20.0
    search_yaw_offsets: tuple = (0.0, 0.60, -0.60, 1.15, -1.15)
    # 搜索阶段不能由单帧开放词汇误检直接驱动底盘。
    search_target_confirm_frames: int = 2
    search_target_max_gap_sec: float = 2.5
    search_target_lock_ttl_sec: float = 4.0
    search_target_pose_tolerance_m: float = 0.22
    search_target_min_dino_score: float = 0.38
    # 赛事固定工作区的宽松边界，只用于拒绝明显落在墙面/地面的伪Pose3D。
    table_target_bounds_xyz: tuple = (
        (-1.90, 0.35),
        (1.70, 2.70),
        (0.68, 1.25),
    )
    shelf_target_bounds_xyz: tuple = (
        (-3.00, -2.25),
        (0.20, 1.35),
        (0.35, 1.65),
    )
    max_approach_retries: int = 2
    approach_retry_backoff_m: float = 0.12

    # 接近桌面目标后使用动态预抓取高度：
    # ActiveObserver会用 table_pregrasp_z0 - target_z 替换这里的spine占位值。
    table_observe_sequence: tuple = (
        ("table_pregrasp_view", 0.0, -0.65, 0.18, 2.4),
    )

    # 搜索只需发现目标，不做精确抓取对齐，因此避免大幅升降腰部。
    table_search_observe_sequence: tuple = (
        ("table_search_overview", 0.0, -0.20, 0.0, 1.8),
        ("table_search_down", 0.0, -0.45, 0.0, 1.8),
    )

    shelf_observe_sequence: tuple = (
        ("shelf_center", 0.0, -0.10, 0.0, 1.5),
        ("shelf_left", 0.25, -0.10, 0.0, 1.5),
        ("shelf_right", -0.25, -0.10, 0.0, 1.5),
        ("shelf_lower", 0.0, -0.25, 0.08, 1.5),
        ("shelf_lower_2", 0.0, -0.35, 0.16, 2.0),
    )

    shelf_search_observe_sequence: tuple = (
        ("shelf_search_center", 0.0, -0.10, 0.0, 1.6),
        ("shelf_search_lower", 0.0, -0.35, 0.16, 1.6),
    )

CONFIG = ActionConfig()
