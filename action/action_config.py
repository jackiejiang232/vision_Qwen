from dataclasses import dataclass
from pathlib import Path


_VISION_ROOT = Path(__file__).resolve().parents[1]
_DG_CONFIG_ROOT = _VISION_ROOT / "third_party" / "dg202612_config"


@dataclass(frozen=True)
class ActionConfig:
    scene_topic: str = "/vlm/scene_understanding"
    detections_topic: str = "/grounded_sam/detections"
    cmd_vel_topic: str = "/cmd_vel"
    odom_topic: str = "/slamware_ros_sdk_server_node/odom"
    ready_topic: str = "/action/ready_for_grasp"

    enable_motion_handoff_stance: bool = True #是否启用 DG202612 的站位规划

    motion_default_box_length: float = 0.24  #当视觉还没稳定输出尺寸时，用官方标准箱尺寸做第一版联调兜底
    motion_default_box_width: float = 0.16
    motion_default_box_height: float = 0.19

    motion_min_confidence: float = 0.35    #低于该置信度就倾向重观察
    motion_reobserve_max_age_sec: float = 1.0  #主动观测请求要求视觉数据的新鲜度

    # A* 配置
    enable_motion_astar: bool = True

    dg_scene_geometry_path: str = str(_DG_CONFIG_ROOT / "scene_geometry.json")
    dg_robot_geometry_path: str = str(_DG_CONFIG_ROOT / "robot_geometry.json")
    dg_grasp_profiles_path: str = str(_DG_CONFIG_ROOT / "grasp_profiles.json")

    astar_position_tolerance: float = 0.05
    astar_yaw_tolerance: float = 0.07
    astar_max_linear_speed: float = 0.35
    astar_max_angular_speed: float = 0.80
    astar_final_approach_distance: float = 0.25
    astar_footprint_clearance: float = 0.02
    astar_dock_stable_sec: float = 0.6

    control_rate_hz: float = 20.0

    max_linear_speed: float = 0.7
    max_angular_speed: float = 1.5
    max_linear_accel: float = 0.80
    max_angular_accel: float = 2.20

    goal_xy_tolerance: float = 0.04
    goal_yaw_tolerance: float = 0.06

    # 比赛任务链路的官方初始点。总控回原点优先使用这个固定值，
    # 避免节点启动晚或旧 odom 把桌前/货架前误记成 home。
    task_use_fixed_home_pose: bool = True
    task_home_x: float = -0.70
    task_home_y: float = 0.55005
    task_home_yaw: float = 1.5707963267948966 + 0.08

    # 先导航到观察/预抓取位姿，不直接贴到物体坐标。
    approach_distance_table: float = 0.55
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
    table_pregrasp_table_yaw_tolerance_rad: float = 0.18

    # 如果目标在桌面右侧，允许底盘目标 x 更靠右一些
    # 这个范围需要结合仿真安全距离调，先用这个测试
    table_pick_x_range: tuple = (-1.45, 0.45)

    # ready_for_grasp 前，目标必须在机器人身体中心线附近
    table_pregrasp_lateral_tolerance_m: float = 0.07
    table_pregrasp_heading_tolerance_rad: float = 0.18
    table_pregrasp_lateral_offsets: tuple = (0.0,)
    servo_target_u_table_offset_px: float = 0.0
    # 桌面抓取时目标不追求图像正中心，而是保持完整可见并略偏上。
    servo_target_v_table: float = 220.0
    servo_target_v_shelf: float = 310.0
    servo_min_v: float = 120.0
    servo_max_v: float = 390.0
    servo_u_tolerance: float = 25.0
    servo_v_tolerance_table: float = 55.0
    servo_v_tolerance_shelf: float = 95.0
    servo_bbox_margin_px: float = 20.0
    servo_table_bottom_margin_px: float = 42.0
    servo_min_bbox_width_px: float = 35.0
    servo_min_bbox_height_px: float = 35.0
    servo_edge_min_bbox_width_px: float = 28.0
    servo_edge_min_bbox_height_px: float = 12.0
    servo_target_distance_table: float = 0.52
    servo_target_distance_shelf: float = 0.62
    servo_depth_tolerance: float = 0.055
    servo_max_linear_speed: float = 0.16
    servo_max_angular_speed: float = 0.12
    servo_shelf_yaw_tolerance_rad: float = 0.16
    servo_shelf_yaw_gain: float = 0.55
    servo_shelf_max_angular_speed: float = 0.16
    servo_table_linear_gain: float = 0.22
    servo_table_max_linear_speed: float = 0.055
    servo_table_close_extra_tolerance: float = 0.025
    servo_stable_frames: int = 3
    # 近距离时目标可能占满画面，GroundingDINO/SAM 会漏检。
    # 这时只要世界坐标、底盘距离、横向偏差和朝向都已经满足桌边抓取窗口，
    # 动作侧允许锁住最后一次有效目标继续进入预抓取。
    close_range_lock_enable: bool = True
    close_range_lock_extra_distance_m: float = 0.08
    close_range_lock_lateral_tolerance_m: float = 0.10
    close_range_lock_heading_tolerance_rad: float = 0.24
    close_range_lock_table_yaw_tolerance_rad: float = 0.24
    close_range_lock_allow_pregrasp_without_detection: bool = True
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

    table_pick_x_range: tuple = (-1.45, -0.05)
    table_pick_y_range: tuple = (1.42, 1.82)
    table_pregrasp_z0: float = 1.32163718
    table_pregrasp_spine_min: float = -0.04
    table_pregrasp_spine_max: float = 0.87
    table_pregrasp_spine_tolerance: float = 0.035
    table_pregrasp_head_pitch: float = -0.65
    table_pregrasp_head_tolerance: float = 0.08
    pregrasp_adjust_enable: bool = True
    pregrasp_adjust_timeout_sec: float = 12.0
    pregrasp_adjust_min_sec: float = 2.5
    pregrasp_adjust_stable_frames: int = 5
    pregrasp_lost_max_frames: int = 10
    pregrasp_spine_reference_z: float = 1.25 
    pregrasp_grasp_height_ratio: float = 0.5 #抓箱子高度的一半
    pregrasp_spine_lower_bias: float = 0.05 #让腰部比理论值再低一点(优先调)
    pregrasp_spine_min_delta: float = -0.02
    pregrasp_spine_max_delta: float = 0.2 #单次最多下降xx m等效量，避免一下子过低丢视野
    
    #如果腰部最后太高，减小 pregrasp_spine_reference_z,如果腰部最后太低，增大 pregrasp_spine_reference_z
    pregrasp_allow_timeout_ready: bool = True

    # 腰部每次小步下降，别一次降到底
    pregrasp_spine_step: float = 0.03

    # 从视觉伺服结束时的腰部位置继续下降多少。
    # 如果你测试发现方向反了，就把 0.12 改成 -0.12。
    pregrasp_spine_delta_after_servo: float = 0.3
    pregrasp_spine_target_tolerance: float = 0.025

    # 头部回正目标。-0.65 比较俯视，-0.30 更回正。
    pregrasp_final_head_pitch: float = -0.55
    pregrasp_head_pitch_tolerance: float = 0.05

    # 桌面箱子最终希望仍然正对图像中心，但位置略偏上
    pregrasp_target_u: float = 320.0
    pregrasp_target_v: float = 210.0
    pregrasp_u_tolerance: float = 70.0
    pregrasp_v_tolerance: float = 90.0

    # 头部回正范围：你现在 -0.65 是比较俯视，回正可先试 -0.35
    pregrasp_head_pitch_min: float = -0.70
    pregrasp_head_pitch_max: float = -0.30
    pregrasp_head_pitch_step: float = 0.03  #头部俯仰速度
    table_grasp_distance_min: float = 0.42
    table_grasp_distance_max: float = 0.60

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
        ("shelf_center", 0.0, -0.20, 0.10, 1.5),
        ("shelf_left", 0.25, -0.20, 0.10, 1.5),
        ("shelf_right", -0.25, -0.20, 0.10, 1.5),
        ("shelf_lower", 0.0, -0.35, 0.20, 1.5),
        ("shelf_lower_2", 0.0, -0.45, 0.26, 2.0),
    )

    shelf_search_observe_sequence: tuple = (
        ("shelf_search_center", 0.0, -0.20, 0.10, 1.6),
        ("shelf_search_lower", 0.0, -0.42, 0.24, 1.8),
    )

    # grasp executor（抓取字段）
    enable_grasp_executor: bool = True
    grasp_auto_start: bool = False
    grasp_dry_run: bool = False
    grasp_allow_unchecked_ik: bool = True
    grasp_require_manual_confirm: bool = True
    grasp_ready_max_age_sec: float = 3.0
    grasp_control_rate_hz: float = 20.0
    grasp_max_joint_step: float = 0.035
    grasp_max_slide_step: float = 0.004
    grasp_command_topic: str = "/action/grasp_command"
    grasp_status_topic: str = "/action/grasp_status"

    # task finish reset: 回原点后把腰部、头部、双臂和夹爪拉回安全待机姿态
    reset_slide: float = 0.00606
    reset_head_yaw: float = 0.05
    reset_head_pitch: float = 0.0
    reset_left_arm: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reset_right_arm: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    reset_left_gripper: float = 0.003
    reset_right_gripper: float = 0.003
    reset_max_joint_step: float = 0.025
    reset_max_slide_step: float = 0.004
    reset_position_tolerance: float = 0.035
    reset_slide_tolerance: float = 0.025
    reset_hold_sec: float = 0.30
    reset_timeout_sec: float = 8.0

    # topics
    ready_for_grasp_topic: str = "/action/ready_for_grasp"
    odom_topic: str = "/slamware_ros_sdk_server_node/odom"
    joint_states_topic: str = "/joint_states"
    dg_control_request_topic: str = "/dg202612/control_request"
    dg_control_heartbeat_topic: str = "/dg202612/control_heartbeat"

    # joint names: 必须和 /joint_states 完全一致
    slide_joint_name: str = "slide_joint"
    head_yaw_joint_name: str = "head_yaw_joint"
    head_pitch_joint_name: str = "head_pitch_joint"
    left_arm_joint_names: tuple = (
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
    )
    right_arm_joint_names: tuple = (
        "right_arm_joint1",
        "right_arm_joint2",
        "right_arm_joint3",
        "right_arm_joint4",
        "right_arm_joint5",
        "right_arm_joint6",
    )
    left_gripper_joint_name: str = "left_arm_eef_gripper_joint"
    right_gripper_joint_name: str = "right_arm_eef_gripper_joint"

    # DG202612 pick executor limits
    grasp_coarse_max_age: float = 3.0
    grasp_coarse_min_confidence: float = 0.70
    grasp_coarse_position_std_m: float = 0.08
    grasp_fine_max_age: float = 2.0
    grasp_fine_min_confidence: float = 0.70
    grasp_fine_position_std_m: float = 0.05
    grasp_fine_yaw_std_rad: float = 0.20
    grasp_approach_max_age: float = 1.0
    grasp_max_centered_error_m: float = 0.04
    grasp_redock_position_tolerance_m: float = 0.05
    grasp_redock_yaw_tolerance_rad: float = 0.08
    grasp_observation_extra_standoff_m: float = 0.20

    # table_side_hug profile: 第一版只用桌边双臂抱持
    table_side_contact_press: float = 0.035
    table_side_pregrasp_gap: float = 0.076
    table_side_contact_longitudinal_offset: float = -0.04
    table_side_contact_height_offset: float = 0.01
    table_side_gripper_opening: float = 1.0

    # lift / retreat: 第一版只做小幅抬升，方向要先小步验证
    grasp_lift_delta_m: float = -0.05
    grasp_lift_step_m: float = 0.003
    grasp_retreat_distance_m: float = 0.20

    # task executor（任务级连续执行）
    task_command_topic: str = "/task/command"
    official_instruction_topic: str = "/material/instruction"
    task_status_topic: str = "/task/status"
    task_pick_goal_topic: str = "/task/pick_goal"

    # generic navigation skill（通用导航接口，仍由 action_navigation_node 执行）
    navigation_goal_topic: str = "/action/navigation_goal"
    navigation_status_topic: str = "/action/navigation_status"

    # place skill（放置执行接口）
    place_command_topic: str = "/action/place_command"
    place_status_topic: str = "/action/place_status"

    # task-level navigation tolerances
    task_nav_xy_tolerance: float = 0.06
    task_nav_yaw_tolerance: float = 0.10
    task_reference_query_topic: str = "/vlm/dino_query"
    task_reference_query_period_sec: float = 1.0

    # place planning（货架层空位规划）
    place_level_edge_margin_m: float = 0.08
    place_object_clearance_m: float = 0.10
    place_default_object_length_m: float = 0.22
    place_default_object_width_m: float = 0.20
    place_default_object_height_m: float = 0.10

CONFIG = ActionConfig()
