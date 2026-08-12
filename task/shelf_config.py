import math


SHELVES = {
    "shelf_01": {
        # 这里需要按仿真地图实测微调。
        # pose 是货架中心朝向参考点，approach_pose 是机器人站在货架前的安全位姿。
        "pose": {"x": -2.55, "y": 0.80, "yaw": math.pi},
        "approach_pose": {"x": -1.75, "y": 0.78, "yaw": math.pi},
        "x_range": [-2.85, -2.25],
        "y_range": [0.25, 1.35],
        "levels": [
            {"id": 0, "z_min": 0.35, "z_max": 0.65, "z_place": 0.50},
            {"id": 1, "z_min": 0.65, "z_max": 0.95, "z_place": 0.80},
            {"id": 2, "z_min": 0.95, "z_max": 1.25, "z_place": 1.10},
        ],
    }
}


def shelf_by_id(shelf_id):
    return SHELVES.get(shelf_id)


def all_shelves():
    return SHELVES.items()


def default_shelf_id():
    return "shelf_01"


def default_shelf_approach_pose():
    shelf = shelf_by_id(default_shelf_id())
    if shelf is None:
        return None
    return dict(shelf["approach_pose"])
