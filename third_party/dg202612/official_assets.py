"""定位官方资产：KDL 运动学模块与竞赛 MJCF 场景。

为什么需要这个模块
==================
交给举办方的是 **Client 镜像**，而官方 Client 基座 ``material_sorting:offline-client``
是一个空壳。2026-08-04 在本机实测：

    $ docker run --rm --entrypoint bash material_sorting:offline-client \\
        -lc 'find / -name mmk2_kdl.py -o -name material_competition.xml'
    （无输出）
    $ ... -lc 'python3 -c "import discoverse"'
    ModuleNotFoundError: No module named 'discoverse'
    $ ... -lc 'ls /workspace/baseline'
    （空目录，WorkingDir 就指向它，等参赛队把代码放进去）

也就是说：开发期顺手在用的 ``/workspace/material_sorting_task/examples/material_sorting``
**只存在于 Server 镜像**，Client 侧一个字节都没有。基座只提供了通用依赖
（Python 3.10、numpy、scipy、mujoco 3.3.0、rclpy、cv2、torch）。

如果运动学求解和碰撞检查把这个路径写死成唯一来源，交付镜像一运行就是
``ModuleNotFoundError: No module named 'mmk2_kdl'`` 或「找不到官方 MJCF」，
Client 进程异常退出——按官方 Q&A（Q16、Q38），Client 异常退出直接判 0 分。
这不是开发期的不便，是交付级的致命缺陷。

因此本仓库把两份官方资产复制进了 ``vendor/``（均为官方镜像内原件，已按 md5 校验）：

===========================  ========  =================================
目录                          体积      内容
===========================  ========  =================================
``vendor/official_kdl/``      ~40 KB    ``mmk2_kdl.py`` + ``arm_kdl.py``
``vendor/official_scene/``    ~12 MB    ``mjcf/`` + ``models/{mjcf,meshes,textures}``
===========================  ========  =================================

两者都是自包含的：KDL 只依赖 numpy 和同目录的 ``arm_kdl``（PyKDL / pinocchio
相关代码在官方原件里就是注释掉的）；MJCF 的 mesh 全部落在 ``models/`` 内，
不需要 ``discoverse`` 包。渲染用的 ``models/3dgs``（256 MB）没有复制——碰撞
检查只用几何，不出图。

查找顺序，以及为什么 vendor 排在官方路径前面
============================================
1. **环境变量显式指定**——排障与实验用，优先级最高；
2. **仓库 ``vendor/``**——交付镜像里唯一存在的一份；
3. **官方 Server 镜像内路径**——仅供开发期兜底。

把 vendor 放在官方路径之前是刻意的。如果反过来，开发期永远命中官方路径，
vendor 这条分支要到交付当天才第一次被执行，风险最大。现在的顺序保证
「开发时跑的就是交付时跑的」：vendor 一旦缺失或损坏，本机立刻暴露，而不是
等到赛场上。
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "KDL_MARKER",
    "OFFICIAL_EXAMPLE",
    "OfficialAssetNotFound",
    "REPO_ROOT",
    "SCENE_MARKER",
    "VENDOR_ROOT",
    "resolve_kdl_root",
    "resolve_scene_root",
]

# 本文件位于 <仓库根>/src/dg202612/official_assets.py，向上三层就是仓库根。
REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor"

# 官方 Server 镜像里的任务目录。Client 基座里不存在，只在开发期有意义。
OFFICIAL_EXAMPLE = Path("/workspace/material_sorting_task/examples/material_sorting")

# 「存在性标志」：判断候选目录是否真的装着要找的东西，而不是只看目录在不在。
# 一个空的同名目录比找不到更难排查，所以这里认文件不认目录。
KDL_MARKER = Path("mmk2_kdl.py")
SCENE_MARKER = Path("mjcf") / "material_competition.xml"

# 环境变量：前两个分别覆盖单项，最后一个指向「官方 example 那种同时含两者」的目录。
ENV_KDL = "DG202612_OFFICIAL_KDL"
ENV_SCENE = "DG202612_OFFICIAL_SCENE"
ENV_EXAMPLE = "DG202612_OFFICIAL_EXAMPLE"


class OfficialAssetNotFound(RuntimeError):
    """所有候选位置都没有找到官方资产。

    错误信息里会列出**实际查过的每一个路径**。资产缺失在赛场上等价于丢分，
    排查时间宝贵，所以宁可把话说满，不要只丢一句「找不到」。
    """


def _env_path(name: str) -> Path | None:
    """读取环境变量形式的路径覆盖；未设置或为空串时返回 ``None``。"""

    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def _resolve(marker: Path, candidates: Iterable[Path | None], what: str) -> Path:
    """按候选顺序返回第一个含 ``marker`` 的目录。

    ``candidates`` 里允许出现 ``None``（表示「这一档没有配置」），直接跳过，
    这样调用方可以把环境变量原样塞进列表，不必先做过滤。
    """

    tried: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        root = Path(candidate)
        tried.append(str(root))
        if (root / marker).is_file():
            return root
    raise OfficialAssetNotFound(
        "找不到{what}（标志文件 {marker}）。已按顺序尝试：\n{tried}\n"
        "若在交付镜像内出现，说明 vendor/ 没有被复制进镜像——"
        "官方 Client 基座不自带任何官方代码或场景资产。".format(
            what=what,
            marker=marker,
            tried="\n".join(f"  {index}. {path}" for index, path in enumerate(tried, 1)),
        )
    )


def resolve_kdl_root(explicit: Path | str | None = None) -> Path:
    """返回含 ``mmk2_kdl.py`` 的目录，供加入 ``sys.path`` 后导入。

    ``explicit`` 由调用方显式指定时优先级最高（测试与实验用），其后依次是
    ``DG202612_OFFICIAL_KDL``、``DG202612_OFFICIAL_EXAMPLE``、仓库 vendor、
    官方 Server 路径。
    """

    return _resolve(
        KDL_MARKER,
        (
            explicit,
            _env_path(ENV_KDL),
            _env_path(ENV_EXAMPLE),
            VENDOR_ROOT / "official_kdl",
            OFFICIAL_EXAMPLE,
        ),
        "官方 KDL 运动学模块",
    )


def resolve_scene_root(explicit: Path | str | None = None) -> Path:
    """返回含 ``mjcf/material_competition.xml`` 的目录。

    该目录同时充当 MJCF 里 ``__REPO_ROOT__`` 占位符的替换值——官方服务器就是
    这么拼路径的（``material_sorting_server.py``），所以 ``models/`` 必须与
    ``mjcf/`` 平级，vendor 里的目录结构是照抄官方的，不能擅自扁平化。
    """

    return _resolve(
        SCENE_MARKER,
        (
            explicit,
            _env_path(ENV_SCENE),
            _env_path(ENV_EXAMPLE),
            VENDOR_ROOT / "official_scene",
            OFFICIAL_EXAMPLE,
        ),
        "官方竞赛 MJCF 场景",
    )
