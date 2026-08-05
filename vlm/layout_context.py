import json
from pathlib import Path


LAYOUT_PATH = Path("/media/jiangzhenmin/data/Challengecup2026/JZM/Vision/config/material_competition_layout.json")

def _phrase_for_movable(item):
    color = item.get("color")
    kind = item.get("kind", "")

    if "box" in kind:
        return f"{color} box"

    return color or item.get("body")


def _phrase_for_prop(item):
    obstacle_kind = item.get("obstacle_kind")

    if obstacle_kind == "cube":
        return "white cube"

    if obstacle_kind == "cuboid":
        return "white cuboid"

    return item.get("prop") or item.get("body")


def load_layout():
    if not LAYOUT_PATH.exists():
        return None

    with LAYOUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_layout_context():
    layout = load_layout()

    if layout is None:
        return {
            "available": False,
            "reason": f"layout file not found: {LAYOUT_PATH}",
        }

    movable_objects = []
    for item in layout.get("movable_boxes", []):
        movable_objects.append(
            {
                "body": item.get("body"),
                "color": item.get("color"),
                "color_zh": item.get("color_zh"),
                "kind": item.get("kind"),
                "location": item.get("location"),
                "world_position": item.get("world_position"),
                "visual_phrase": _phrase_for_movable(item),
                "half_size": item.get("half_size"),
            }
        )

    fixed_props = []
    for item in layout.get("fixed_props", []):
        fixed_props.append(
            {
                "body": item.get("body"),
                "prop": item.get("prop"),
                "prop_zh": item.get("prop_zh"),
                "obstacle_kind": item.get("obstacle_kind"),
                "obstacle_zh": item.get("obstacle_zh"),
                "location": item.get("location"),
                "world_position": item.get("world_position"),
                "visual_phrase": _phrase_for_prop(item),
                "half_size": item.get("half_size"),
            }
        )

    scene = layout.get("scene", {})

    return {
        "available": True,
        "movable_objects": movable_objects,
        "fixed_props": fixed_props,
        "scene": {
            "table_top_z": scene.get("table_top_z"),
            "shelf_board_surfaces_z": scene.get(
                "shelf_board_surfaces_z"
            ),
            "table_place_zone": scene.get("table_place_zone"),
            "picking_zone": scene.get("picking_zone"),
            "delivery_zone": scene.get("delivery_zone"),
            "end_zone": scene.get("end_zone"),
        },
    }


def build_known_visual_objects():
    context = build_layout_context()

    phrases = []

    for item in context.get("movable_objects", []):
        phrase = item.get("visual_phrase")
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    for item in context.get("fixed_props", []):
        phrase = item.get("visual_phrase")
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    for item in ("table", "shelf", "shelf layer"):
        if item not in phrases:
            phrases.append(item)

    return phrases

def find_layout_object_by_label(label):
    context = build_layout_context()
    label = str(label or "").lower()

    for item in context.get("movable_objects", []):
        phrase = str(item.get("visual_phrase") or "").lower()
        if phrase and phrase in label:
            return item

    for item in context.get("fixed_props", []):
        phrase = str(item.get("visual_phrase") or "").lower()
        if phrase and phrase in label:
            return item

    return None


def layout_size_from_item(item):
    half_size = item.get("half_size")
    if not half_size or len(half_size) != 3:
        return None

    return {
        "length": float(half_size[0]) * 2.0,
        "width": float(half_size[1]) * 2.0,
        "height": float(half_size[2]) * 2.0,
    }
# =========================
# 货架几何先验
# =========================

# 官方仿真中货架中心位置，来自 material_competition.xml / layout
SHELF_CENTER_X = -2.67
SHELF_CENTER_Y = 0.778

# 赛事说明：单层尺寸 203 x 80 x 28 cm
SHELF_LENGTH = 2.03
SHELF_DEPTH = 0.80
SHELF_LAYER_USABLE_HEIGHT = 0.28

# 给定位误差留余量
SHELF_X_MARGIN = 0.15
SHELF_Y_MARGIN = 0.15
SHELF_LAYER_Z_TOLERANCE = 0.16

# 官方 layout 中的货架板面高度，优先用这个，因为它直接对应仿真
TASK_SHELF_SURFACE_Z = {
    1: 0.403,
    2: 0.732,
    3: 1.061,
    4: 1.366,
    5: 1.695,
    6: 2.024,
}

BOX_HALF_HEIGHT = 0.095
WHITE_CUBOID_HALF_HEIGHT = 0.117
SUPPORT_CLEARANCE = 0.010


TABLE_ZONE = {
    "x": (-1.37, 0.29),
    "y": (1.92, 2.71),
    "z": (0.74, 1.05),
}

TABLE_MARGIN_XY = 0.12
TABLE_Z_TOLERANCE = 0.16


def is_manipulable_object(label):
    label = str(label or "").lower()

    if label in ("shelf", "table"):
        return False

    return any(
        word in label
        for word in ("box", "cube", "cuboid", "cylinder")
    )


def is_pose_inside_shelf_xy(pose_world):
    if pose_world is None:
        return False

    x = float(pose_world.get("x", 999.0))
    y = float(pose_world.get("y", 999.0))

    x_min = SHELF_CENTER_X - SHELF_LENGTH / 2.0 - SHELF_X_MARGIN
    x_max = SHELF_CENTER_X + SHELF_LENGTH / 2.0 + SHELF_X_MARGIN

    y_min = SHELF_CENTER_Y - SHELF_DEPTH / 2.0 - SHELF_Y_MARGIN
    y_max = SHELF_CENTER_Y + SHELF_DEPTH / 2.0 + SHELF_Y_MARGIN

    return x_min <= x <= x_max and y_min <= y <= y_max


def is_pose_inside_table_xy(pose_world):
    if pose_world is None:
        return False

    x = float(pose_world.get("x", 999.0))
    y = float(pose_world.get("y", 999.0))

    x_min, x_max = TABLE_ZONE["x"]
    y_min, y_max = TABLE_ZONE["y"]

    return (
        x_min - TABLE_MARGIN_XY <= x <= x_max + TABLE_MARGIN_XY
        and y_min - TABLE_MARGIN_XY <= y <= y_max + TABLE_MARGIN_XY
    )


def object_half_height_from_label(label):
    label = str(label or "").lower()

    if "white cuboid" in label or "cuboid" in label:
        return WHITE_CUBOID_HALF_HEIGHT

    return BOX_HALF_HEIGHT


def empty_surface_location():
    return {
        "support_surface": "unknown",
        "on_table": False,
        "on_shelf": False,
        "support_surface_index": None,
        "shelf_layer": None,
        "shelf_surface_z": None,
        "shelf_layer_confidence": 0.0,
        "table_height_confidence": 0.0,
    }


def infer_table_location(pose_world, label):
    result = empty_surface_location()

    if pose_world is None or not is_manipulable_object(label):
        return result

    if not is_pose_inside_table_xy(pose_world):
        return result

    z = float(pose_world.get("z", 0.0))
    half_height = object_half_height_from_label(label)
    expected_center_z = (
        TABLE_ZONE["z"][0] + half_height + SUPPORT_CLEARANCE
    )
    error = abs(z - expected_center_z)

    if error > TABLE_Z_TOLERANCE:
        result["support_surface"] = "table_candidate"
        return result

    result.update(
        {
            "support_surface": "table",
            "on_table": True,
            "table_height_confidence": float(
                max(0.0, 1.0 - error / TABLE_Z_TOLERANCE)
            ),
        }
    )
    return result


def infer_shelf_layer(pose_world, label):
    result = empty_surface_location()

    if not is_manipulable_object(label):
        return result

    if pose_world is None:
        return result

    if not is_pose_inside_shelf_xy(pose_world):
        return result

    z = float(pose_world.get("z", 0.0))
    half_height = object_half_height_from_label(label)

    expected_centers = {
        layer: surface_z + half_height + SUPPORT_CLEARANCE
        for layer, surface_z in TASK_SHELF_SURFACE_Z.items()
    }

    best_layer = min(
        expected_centers,
        key=lambda layer: abs(z - expected_centers[layer]),
    )

    error = abs(z - expected_centers[best_layer])

    if error > SHELF_LAYER_Z_TOLERANCE:
        result["support_surface"] = "shelf_candidate"
        result["on_shelf"] = True
        result["support_surface_index"] = int(best_layer)
        result["shelf_surface_z"] = TASK_SHELF_SURFACE_Z[best_layer]
        return result

    confidence = max(
        0.0,
        1.0 - error / SHELF_LAYER_Z_TOLERANCE,
    )

    result.update(
        {
            "support_surface": "shelf",
            "on_shelf": True,
            "support_surface_index": int(best_layer),
            "shelf_layer": int(best_layer) + 1,
            "shelf_surface_z": TASK_SHELF_SURFACE_Z[best_layer],
            "shelf_layer_confidence": float(confidence),
        }
    )
    return result


def infer_surface_location(pose_world, label):
    table_result = infer_table_location(pose_world, label)
    shelf_result = infer_shelf_layer(pose_world, label)

    if table_result["on_table"]:
        return table_result

    if shelf_result["on_shelf"]:
        return shelf_result

    if table_result["support_surface"] == "table_candidate":
        return table_result

    return shelf_result
