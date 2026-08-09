"""官方裁判规则的唯一读取入口。

规则来自两份原件，都在 ``config/official/`` 下，都是从正式镜像
``material_sorting:offline-server`` 里原样取出、一个字节都没改过的：

* ``referee.py`` —— 裁判程序本身。它内部有一份 ``DEFAULTS`` 字典，装着**完整**的
  规则：阈值、区域、计分表，以及只存在于这里的**连杆名单**
  （``touch_links`` / ``left_grip_links`` / ``right_grip_links`` / ``robot_links``）。
* ``material_referee_config.json`` —— 赛事方下发的覆盖层。它只列了会被改动的那些
  字段，其余的**沿用 DEFAULTS**。

裁判的读法是 ``_merge(DEFAULTS, json)``：JSON 里有的以 JSON 为准，JSON 里没有的用
默认值。**这个模块做的是同一件事**，所以我们查到的每一条都和裁判当场用的完全一致。

这一点当初差点搞错：最早只读 JSON，结果连杆名单一条都查不到——因为它根本不在 JSON
里。只读覆盖层等于只拿到了半份规则，而缺的那半份恰好是判"有没有夹住"的依据。

**为什么不把这些数字抄进我们自己的配置文件。**

这些阈值不是我们标定出来的，是赛事方定义的评分依据。抄一份到自己家里，就等于
凭空造出"两个都自称权威的数"：官方哪天改了配置，我们的副本不会跟着变，而代码
仍然照着副本跑——分数按官方的算，动作按我们的做，两边对不上的时候没有任何东西
会报错，只会莫名其妙丢分。所以这里的做法是：**原件进仓库，代码只读原件**。

同一个理由也解释了为什么这个模块几乎不做转换：读出来是什么就是什么。凡是需要
换算的地方（比如把结束区矩形喂给 A*），换算写在调用方，原始数值留在这里，出问题
时能一眼看出是抄错了还是算错了。

**这些数字为什么对运动模块要紧**（每一条都直接对应分数或判负）：

* ``end_zone``：机器人**离开**它，当前任务的一次尝试才开始；**回到**它，这次尝试
  才结算并拿"安全返回"分。三项任务各 10 分，合计 30 分——占满分 160 的将近两成，
  而且它是唯一一个"不抓不放也能拿到"的分项。
  另有一条容易踩的坑：回到结束区是**立即结算、不可撤销**的，所以携着箱子的路径
  绝不能路过结束区。详见 ``docs/官方判分规则.md``。
* ``carry_out_dist``：目标箱相对**本局开局位置**的水平位移要够这个数，才算"夹持
  并搬离"。注意基准是箱子的开局位置，不是底盘走了多远。
* ``drop_z``：已判定搬离的箱子掉到这个高度以下，当前尝试立即结算——抱着箱子的
  全程高度都要留出余量。
* ``settle_speed``：放下之后要"运动稳定"才算数，松爪就走可能白放。
* ``place_point_radius`` / ``shelf_place_z_tol`` / ``place_side_offset``：放置判定的
  水平半径、货架层高容差和"左边"的侧向偏移量。
* ``collision_structures``：**只有货架和外围墙**算结构碰撞。碰到桌子不扣自动裁判
  的分——但裁判组另有"任务安全性"主观分，碰撞风险可能判负分，所以这不是"可以随便
  撞桌子"的许可。
* ``grip_links``：判"夹住了"用的连杆名单，**包含腕部 ``*_arm_link6``**，不是只看
  指尖。这是抱持（hug）方案成立的直接依据——用两条前臂内侧把箱子抱住，即使指垫
  没贴上，腕部接触一样算夹持。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

#: 官方原件所在目录。放在 ``config/official/`` 下而不是和我们自己的配置混在一起，
#: 是为了让"这些文件不许手改"这件事一眼可见。
_OFFICIAL_DIR = Path(__file__).resolve().parents[2] / "config" / "official"

#: 赛事方下发的覆盖层（只列被改动的字段）。
OFFICIAL_REFEREE_CONFIG = _OFFICIAL_DIR / "material_referee_config.json"

#: 裁判程序本身，内含完整规则的 ``DEFAULTS``。
OFFICIAL_REFEREE_SOURCE = _OFFICIAL_DIR / "referee.py"

#: ``referee.py`` 里那个字典的变量名。单独列出来是因为它是我们与官方源码之间唯一的
#: 约定：官方哪天改了名字，解析会当场报错，而不是悄悄退回一份空规则。
_DEFAULTS_SYMBOL = "DEFAULTS"


class RefereeRulesError(ValueError):
    """官方裁判配置读不出来，或者缺了我们依赖的字段。

    单独起一个异常类型，是为了让"官方文件对不上"和"我们自己算错了"在日志里分得开：
    前者意味着镜像换了版本、该重新取一份原件，后者才是我们的 bug。
    """


@dataclass(frozen=True)
class Rect2D:
    """轴对齐矩形区域，单位米，坐标系与 ``/slamware_ros_sdk_server_node/odom`` 一致。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        """点是否落在区域内（含边界）。

        含边界而不是开区间：裁判判定用的是同一份矩形，我们比它更严只会自找麻烦——
        明明已经到了却以为没到，白白多走一段。
        """

        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def center(self) -> tuple[float, float]:
        """区域中心。返回结束区中心是最保险的"回家"目标点。"""

        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RefereeRulesError(f"官方裁判配置的 {field} 不是对象：{value!r}")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefereeRulesError(f"官方裁判配置的 {field} 不是数值：{value!r}")
    return float(value)


def _pair(value: Any, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RefereeRulesError(f"官方裁判配置的 {field} 不是两元素区间：{value!r}")
    low = _number(value[0], f"{field}[0]")
    high = _number(value[1], f"{field}[1]")
    if low > high:
        raise RefereeRulesError(f"官方裁判配置的 {field} 区间上下颠倒：{value!r}")
    return low, high


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise RefereeRulesError(f"{field} 不是字符串列表：{value!r}")
    return tuple(value)


@lru_cache(maxsize=4)
def _load_defaults(source: Path) -> Mapping[str, Any]:
    """从 ``referee.py`` 里把 ``DEFAULTS`` 那个字典取出来。

    **为什么用 ast 解析而不是 import。**

    ``referee.py`` 顶部 import 了 mujoco、scipy 这些只有仿真环境里才有的包，直接
    ``import`` 会在开发机上炸掉。更重要的是，它是评测方的程序，我们只想读它的数据，
    不该让它在我们的进程里执行任何一行。

    **为什么用 ast 解析而不是抄一份下来。**

    和整个模块同一个理由：抄就会不一致。这里多一层——连杆名单有二十来个字符串，
    手抄漏一个不会报错，只会让我们对"夹住了没有"的判断和裁判差一点。

    只接受纯字面量（``ast.literal_eval`` 的限制）。官方若把 DEFAULTS 改成计算出来
    的，这里会抛错——那是应该抛的：那时候"读原件"这条路本身就断了，得换做法，而不是
    退回去猜。
    """

    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise RefereeRulesError(f"读不到官方裁判源码 {source}：{exc}") from exc
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise RefereeRulesError(f"官方裁判源码解析失败 {source}：{exc}") from exc

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if _DEFAULTS_SYMBOL not in names:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError as exc:
            raise RefereeRulesError(
                f"{source} 里的 {_DEFAULTS_SYMBOL} 不是纯字面量，无法安全读取：{exc}"
            ) from exc
        return _mapping(value, _DEFAULTS_SYMBOL)

    raise RefereeRulesError(f"{source} 里找不到模块级的 {_DEFAULTS_SYMBOL}")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """按官方 ``referee.py`` 的 ``_merge`` 同样的语义合并：逐层覆盖，同为字典则递归。

    重新实现而不是调用官方的那个函数，是因为我们不 import 那个模块（见
    ``_load_defaults`` 的说明）。语义必须一致——差一点就意味着我们读到的规则和裁判
    用的规则不是同一份。守门用例
    ``tests/test_referee_rules.py::MergeSemanticsTests`` 盯着这件事。
    """

    out = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            out[key] = _merge(current, value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=4)
def load_referee_rules(
    path: Path | None = None, source: Path | None = None
) -> Mapping[str, Any]:
    """读出裁判当场使用的完整规则 = ``DEFAULTS`` 被下发 JSON 覆盖之后的结果。

    结果带缓存：这份规则在一局比赛里不会变，而调用方（导航、放置判定、返回判定）
    会反复问它。缓存键是两个路径，测试里传自己的临时文件不会串味。
    """

    defaults = _load_defaults(OFFICIAL_REFEREE_SOURCE if source is None else Path(source))

    target = OFFICIAL_REFEREE_CONFIG if path is None else Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise RefereeRulesError(f"读不到官方裁判配置 {target}：{exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RefereeRulesError(f"官方裁判配置不是合法 JSON：{exc}") from exc

    return _merge(defaults, _mapping(payload, "根对象"))


def end_zone(path: Path | None = None) -> Rect2D:
    """结束区矩形。

    离开它=尝试开始，回到它=尝试结算并拿"安全返回"分。
    """

    zones = _mapping(load_referee_rules(path).get("zones"), "zones")
    zone = _mapping(zones.get("end_zone"), "zones.end_zone")
    x_min, x_max = _pair(zone.get("x"), "zones.end_zone.x")
    y_min, y_max = _pair(zone.get("y"), "zones.end_zone.y")
    return Rect2D(x_min, x_max, y_min, y_max)


def threshold(name: str, path: Path | None = None) -> float:
    """取一个判分阈值。

    不提供默认值：拿不到就抛异常，而不是悄悄用一个我们编的数继续跑。判分阈值上
    "猜一个差不多的"没有意义——差一点就是差一整个分项。
    """

    thresholds = _mapping(load_referee_rules(path).get("thresholds"), "thresholds")
    if name not in thresholds:
        raise RefereeRulesError(f"官方裁判配置里没有阈值 {name}")
    return _number(thresholds[name], f"thresholds.{name}")


def collision_structures(path: Path | None = None) -> tuple[str, ...]:
    """会被判"结构碰撞"的物体名单。

    碰到名单里的东西，本次尝试拿不到安全返回分，且局部恢复也清不掉这条记录。
    """

    return _string_list(
        load_referee_rules(path).get("collision_structures"), "collision_structures"
    )


def touch_links(path: Path | None = None) -> tuple[str, ...]:
    """判"碰到目标箱"用的连杆名单（第一个分项）。"""

    return _string_list(load_referee_rules(path).get("touch_links"), "touch_links")


def grip_links(side: str, path: Path | None = None) -> tuple[str, ...]:
    """判"夹住了"用的单侧连杆名单。``side`` 取 ``"left"`` 或 ``"right"``。

    裁判的判据是**左侧任一 且 右侧任一**同时接触目标箱。名单里除了两个指头，还有
    腕部 ``*_arm_link6``——这正是抱持方案的依据：两条前臂把箱子抱住，指垫没贴上也算。
    """

    if side not in ("left", "right"):
        raise RefereeRulesError(f"grip_links 的 side 只能是 left 或 right，收到 {side!r}")
    field = f"{side}_grip_links"
    return _string_list(load_referee_rules(path).get(field), field)


def robot_links(path: Path | None = None) -> tuple[str, ...]:
    """裁判做碰撞检测时视作"机器人"的全部连杆。"""

    return _string_list(load_referee_rules(path).get("robot_links"), "robot_links")


def time_limit_s(path: Path | None = None) -> float:
    """本局时间上限，单位秒。注意裁判用的是 MuJoCo 仿真时间，不是墙钟。"""

    return _number(load_referee_rules(path).get("time_limit_s"), "time_limit_s")


def max_attempts(path: Path | None = None) -> int:
    """每项任务的尝试次数上限。"""

    value = load_referee_rules(path).get("max_attempts")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RefereeRulesError(f"max_attempts 不是整数：{value!r}")
    return value