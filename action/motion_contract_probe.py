import json
import sys
from pathlib import Path

from .action_config import CONFIG
from .motion_handoff_adapter import build_motion_handoff


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "用法: python3 -m action.motion_contract_probe scene.json"
        )

    payload = json.loads(Path(sys.argv[1]).read_text())
    result = build_motion_handoff(payload, CONFIG)

    print("status:", result.status)
    print("reason:", result.reason)
    print("blocking_reasons:", result.blocking_reasons)

    if result.approach_pose is not None:
        print(
            "approach_pose:",
            result.approach_pose.x,
            result.approach_pose.y,
            result.approach_pose.yaw,
        )


if __name__ == "__main__":
    main()