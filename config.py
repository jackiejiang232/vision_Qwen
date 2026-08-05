from pathlib import Path


JZM_ROOT = Path("/media/jiangzhenmin/系统/JZM")

DINO_ROOT = JZM_ROOT / "GroundingDINO"
SAM_ROOT = JZM_ROOT / "segment-anything"
VISION_ROOT = JZM_ROOT / "Vision"

DINO_CONFIG = (
    DINO_ROOT
    / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

DINO_CHECKPOINT = (
    DINO_ROOT
    / "weights/groundingdino_swint_ogc.pth"
)

DINO_BERT = DINO_ROOT / "bert-base-uncased"

SAM_CHECKPOINT = (
    SAM_ROOT
    / "sam_vit_h_4b8939.pth"
)

SAM_MODEL_TYPE = "vit_h"

DEVICE = "cuda"

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

DEFAULT_PROMPT = (
    "pink box . "
    "yellow box . "
    "brown box . "
    "white cube . "
    "white rectangular box ."
)