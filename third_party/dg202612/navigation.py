"""透明的静态地图导航基线。

第一版只使用人工确认的墙、桌、货架等固定结构。箱子的随机位置不写进地图：
每次都由实时目标和抓取方向生成操作站位，再用 A* 规划到该站位。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Any, Iterable, Mapping, Sequence

from .contracts import ObjectState, Pose2D
from .referee_rules import end_zone


class PathNotFound(ValueError):
    """静态地图中不存在安全路径或起终点本身不合法。"""


@dataclass(frozen=True)
class AxisAlignedRect:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if not self.min_x < self.max_x or not self.min_y < self.max_y:
            raise ValueError("rectangle min values must be smaller than max values")

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def inflated(self, margin: float) -> "AxisAlignedRect":
        if margin < 0.0:
            raise ValueError("inflation margin cannot be negative")
        return AxisAlignedRect(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin,
        )

    def inset(self, margin: float) -> "AxisAlignedRect":
        return AxisAlignedRect(
            self.min_x + margin,
            self.min_y + margin,
            self.max_x - margin,
            self.max_y - margin,
        )


@dataclass(frozen=True)
class StaticMap:
    bounds: AxisAlignedRect
    obstacles: tuple[AxisAlignedRect, ...]
    resolution: float
    safety_margin: float

    def __post_init__(self) -> None:
        if self.resolution <= 0.0 or self.safety_margin < 0.0:
            raise ValueError("resolution must be positive and safety_margin non-negative")
        # 机器人中心必须留在地图内侧，障碍物则按同一半径外扩。
        self.bounds.inset(self.safety_margin)

    @property
    def usable_bounds(self) -> AxisAlignedRect:
        return self.bounds.inset(self.safety_margin)

    @property
    def inflated_obstacles(self) -> tuple[AxisAlignedRect, ...]:
        return tuple(item.inflated(self.safety_margin) for item in self.obstacles)


def end_zone_keepout() -> AxisAlignedRect:
    """把官方结束区转成规划器认识的矩形。

    坐标一律来自 :mod:`dg202612.referee_rules`，也就是官方原件；这里只做**格式**
    转换，一个数都不重写。

    转换本身有个很容易翻车的地方，所以单独写成一个函数而不是在调用处随手展开：
    两个矩形类型的字段顺序**不一样**。

    * :class:`~dg202612.referee_rules.Rect2D` 是 ``(x_min, x_max, y_min, y_max)``
      ——先把 x 说完再说 y，跟官方 JSON 里 ``{"x": [...], "y": [...]}`` 的写法一致。
    * :class:`AxisAlignedRect` 是 ``(min_x, min_y, max_x, max_y)``
      ——先说两个最小值再说两个最大值，是规划器一贯的写法。

    顺序抄反了不会报错（四个都是浮点数），只会得到一个位置完全错误的矩形。
    ``tests/test_navigation_framework.py`` 里有一条用例专门盯着这件事。
    """

    zone = end_zone()
    return AxisAlignedRect(zone.x_min, zone.y_min, zone.x_max, zone.y_max)


def with_carrying_keepout(
    static_map: StaticMap, keepout: AxisAlignedRect | None = None
) -> StaticMap:
    """返回一份**携物专用**的地图：在原有障碍之外，把结束区也设成禁行。

    **为什么携着箱子就不能路过结束区。**

    官方裁判判定一次尝试结束的条件是"机器人回到结束区，且本次尝试已推进过至少一个
    里程碑"——而且是**当场结算、不可撤销**：

    .. code-block:: python

        if in_end and (f.touched or f.lifted or f.placed):
            f.returned = True
            self._settle_attempt(t)
            return

    也就是说，抱着箱子从桌子走向货架的路上只要蹭进结束区一下，裁判就认为这次尝试
    做完了，按当前进度结算——箱子还在手上，放置那 10～20 分直接没了，还白白消耗掉
    三次机会里的一次。这种失败没有任何征兆：不报错、不碰撞、动作看上去一切正常。

    **为什么回家的时候不能用这份地图。**

    很显然：结束区成了障碍，A* 就永远规划不进去，那 30 分也就永远拿不到。所以这是
    一份**阶段性**的地图，只在"手上有东西"的那几段路上用。函数名里写 carrying
    就是为了让用错的时候读起来别扭。

    **为什么让它跟着 safety_margin 一起外扩。**

    这里放进去的是结束区的**原始**矩形，``StaticMap.inflated_obstacles`` 会像对待
    其它障碍一样把它外扩一个 ``safety_margin``。看起来比裁判严——裁判判的是
    ``site("base_link")`` 那**一个点**在不在矩形里，按理不需要给车身留半径。

    但这个"过严"恰恰是我们要的：裁判读的是 MuJoCo 里的世界坐标，我们规划时手上只有
    odom，两者在一局 600 秒里会累积漂移。多留一个底盘半径，等于给这段偏差买了保险。
    代价只是路绕远一点，而赌错的代价是整整一次尝试。

    :param keepout: 允许显式传入禁行矩形，默认取官方结束区。留这个口子是给测试用的
        ——用例可以摆一个位置好构造的矩形，不必迁就官方那份坐标。
    """

    zone = end_zone_keepout() if keepout is None else keepout
    return StaticMap(
        bounds=static_map.bounds,
        obstacles=static_map.obstacles + (zone,),
        resolution=static_map.resolution,
        safety_margin=static_map.safety_margin,
    )


@dataclass(frozen=True)
class PathPlan:
    waypoints: tuple[Pose2D, ...]
    grid_cells: tuple[tuple[int, int], ...]
    cost: float
    waypoint_cells: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class BaseFootprint:
    """底盘坐标系中的矩形碰撞外形。"""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if not self.min_x < self.max_x or not self.min_y < self.max_y:
            raise ValueError("base footprint min values must be smaller than max values")

    def inflated(self, clearance: float) -> "BaseFootprint":
        if clearance < 0.0:
            raise ValueError("footprint clearance cannot be negative")
        return BaseFootprint(
            self.min_x - clearance,
            self.min_y - clearance,
            self.max_x + clearance,
            self.max_y + clearance,
        )

    def world_corners(self, pose: Pose2D) -> tuple[tuple[float, float], ...]:
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        corners = (
            (self.min_x, self.min_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y),
            (self.min_x, self.max_y),
        )
        return tuple(
            (
                pose.x + cosine * x - sine * y,
                pose.y + sine * x + cosine * y,
            )
            for x, y in corners
        )


@dataclass(frozen=True)
class DockRoute:
    """保守 A* 路段加一段用真实矩形外形验证的精确停靠。"""

    transit: PathPlan
    goal: Pose2D


@dataclass(frozen=True)
class BaseVelocity:
    linear: float
    angular: float
    finished: bool


@dataclass(frozen=True)
class DockProgress:
    """一次路径跟踪更新的可见结果。"""

    command: BaseVelocity
    position_error: float
    yaw_error: float
    stable_for: float
    completed: bool
    phase: str


@dataclass(frozen=True)
class RetreatRoute:
    """一段已经用真实外形验证过的直线倒退。

    对应官方 ``client_task_1.py`` 的 s=8：抱着箱子沿**车头的反方向**直线退开一
    段，朝向一动不动，退完再另行转向。之所以单独建一个数据类而不复用
    :class:`DockRoute`，是因为这两件事的约束完全相反——DOCK 允许绕路（A* 想怎么
    拐弯都行，只要别撞），RETREAT 不允许拐弯（一旦拐弯，怀里的箱子就会甩出去，
    而且我们扫过的那条走廊也就不再是机器人实际走的那条）。

    ``distance`` 是冗余字段（等于 start 到 goal 的距离），但保留它是有意的：执行
    端判断"退够了没有"用的是**沿倒退方向的投影**，不是到 goal 的欧氏距离，所以
    这个标量才是判据本体，写在计划里比让执行端自己再算一遍更不容易出错。
    """

    start: Pose2D
    goal: Pose2D
    distance: float


@dataclass(frozen=True)
class RetreatProgress:
    """一次倒退更新的可见结果。

    与 :class:`DockProgress` 的三点差异，都是"倒退"这件事本身带来的：

    1. 没有 ``stable_for``。DOCK 要在终点稳住 1 秒才算数，因为紧接着就要伸手；
       RETREAT 退完就完了，后面还要重新导航，没有"稳住"的意义。
    2. 多了 ``lateral_error``。走廊是沿一条直线扫出来的，机器人一旦横向漂出去，
       它实际走的就不是被检查过的那条路了——这个量是安全判据，不是装饰。
    3. ``yaw_error`` **带符号**（``DockProgress.yaw_error`` 是绝对值）。倒退是盲
       退，符号一直偏向同一侧，说明的是标定问题而不是随机噪声，丢掉符号就看不
       出来了。
    """

    command: BaseVelocity
    travelled: float
    remaining: float
    lateral_error: float
    yaw_error: float
    completed: bool
    abort_reason: str | None = None


class StaticAStarPlanner:
    """小地图的八邻域 A*；保留原始网格路径，同时输出可跟踪的简化折线。"""

    _NEIGHBORS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    # 起点陷在膨胀区时，允许向外逃逸的最大距离。这个上限不是性能考虑，是判据：
    # 走这么远还出不去，说明机器人不是"贴着桌子停好了"，而是真的被围住了——
    # 那种情况应该停下来报错，让人来看，而不是自动生成一条长长的脱困路径。
    _MAX_ESCAPE_M = 0.5

    def __init__(self, static_map: StaticMap) -> None:
        self.map = static_map
        self._width = math.ceil((static_map.bounds.max_x - static_map.bounds.min_x) / static_map.resolution)
        self._height = math.ceil((static_map.bounds.max_y - static_map.bounds.min_y) / static_map.resolution)

    def _cell(self, x: float, y: float, *, allow_inflated: bool = False) -> tuple[int, int]:
        if not self.map.usable_bounds.contains(x, y):
            raise PathNotFound("start or goal is outside the usable map bounds")
        ix = min(self._width - 1, int((x - self.map.bounds.min_x) / self.map.resolution))
        iy = min(self._height - 1, int((y - self.map.bounds.min_y) / self.map.resolution))
        cell = (ix, iy)
        if allow_inflated:
            # 起点专用：机器人可能就停在膨胀区里（比如贴着桌子完成上一次停靠），
            # 那是它现在的物理位置，不是一个可以拒绝的规划请求。但真实障碍内部
            # 永远不接受——那意味着实测位姿本身就穿模了，该报错而不是往下算。
            if self._inside_real_obstacle(cell):
                raise PathNotFound("start lies inside a real obstacle")
            return cell
        if self._blocked(cell):
            raise PathNotFound("start or goal lies inside an inflated obstacle")
        return cell

    def _inside_real_obstacle(self, cell: tuple[int, int]) -> bool:
        """格心是否落在**未膨胀**的真实障碍里。

        与 :meth:`_blocked` 的区别就是安全余量：``_blocked`` 问的是"规划能不能
        用这一格"，这里问的是"这一格是不是实心的"。逃逸段允许穿过前者，绝不
        允许穿过后者。
        """

        if not self._in_grid(cell):
            return True
        x, y = self._center(cell)
        if not self.map.usable_bounds.contains(x, y):
            return True
        return any(rect.contains(x, y) for rect in self.map.obstacles)

    def _escape_path(self, start_cell: tuple[int, int]) -> list[tuple[int, int]]:
        """从膨胀区内的起点走到最近一个可规划格，返回含首尾的格子序列。

        **为什么需要它。** 膨胀区是安全余量，不是墙。机器人贴着桌子停靠完毕时，
        它的中心就落在桌子的膨胀区内——这是任务成功的状态，不是错误状态。可上一
        版把起点和终点一视同仁地拒掉了，后果是**一旦停靠到位就再也无法重新规划**：
        抓完箱子要去货架、放完要回来抓下一个，每一次都从"贴着某个东西"出发，
        整条任务链走不通。2026-08-01 想原地重跑 DOCK 校正偏航时撞上了这一点。

        **为什么这样做是安全的。** 逃逸段只允许穿过"膨胀但非实心"的格子，真实
        障碍一格都不碰（见 :meth:`_inside_real_obstacle`）。机器人此刻**物理上就
        在膨胀区里且实测零碰撞**，从这里沿最短路走出去，是在离开障碍而不是靠近
        它。再加两道闸：逃逸距离封顶，且出口必须是正常可规划格。

        **为什么用 BFS 而不是 A\\*。** 这里没有目标点，要找的是"最近的自由格"，
        八邻域 BFS 的首次命中就是答案，不需要启发函数。
        """

        if not self._blocked(start_cell):
            return [start_cell]
        limit_cells = math.ceil(self._MAX_ESCAPE_M / self.map.resolution)
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque(
            [(start_cell, [start_cell])]
        )
        seen = {start_cell}
        while queue:
            cell, trail = queue.popleft()
            if len(trail) > limit_cells + 1:
                break
            for dx, dy in self._NEIGHBORS:
                next_cell = (cell[0] + dx, cell[1] + dy)
                if next_cell in seen or not self._in_grid(next_cell):
                    continue
                # 实心障碍绝不穿越；膨胀区内可以走，因为机器人已经在里面了。
                if self._inside_real_obstacle(next_cell):
                    continue
                seen.add(next_cell)
                next_trail = trail + [next_cell]
                if not self._blocked(next_cell):
                    return next_trail
                queue.append((next_cell, next_trail))
        raise PathNotFound(
            "start lies inside an inflated obstacle and no free cell is reachable "
            f"within {self._MAX_ESCAPE_M:.2f} m"
        )

    def _center(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.map.bounds.min_x + (cell[0] + 0.5) * self.map.resolution,
            self.map.bounds.min_y + (cell[1] + 0.5) * self.map.resolution,
        )

    def _in_grid(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self._width and 0 <= cell[1] < self._height

    def _blocked(self, cell: tuple[int, int]) -> bool:
        if not self._in_grid(cell):
            return True
        x, y = self._center(cell)
        if not self.map.usable_bounds.contains(x, y):
            return True
        return any(rect.contains(x, y) for rect in self.map.inflated_obstacles)

    @staticmethod
    def _heuristic(first: tuple[int, int], second: tuple[int, int]) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _neighbors(self, cell: tuple[int, int]) -> Iterable[tuple[tuple[int, int], float]]:
        for dx, dy in self._NEIGHBORS:
            next_cell = (cell[0] + dx, cell[1] + dy)
            if self._blocked(next_cell):
                continue
            # 禁止从两个障碍角之间斜穿，路径形状才与真实膨胀障碍一致。
            if dx and dy and (self._blocked((cell[0] + dx, cell[1])) or self._blocked((cell[0], cell[1] + dy))):
                continue
            yield next_cell, math.hypot(dx, dy)

    def _segment_clear(self, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        """检查两个网格中心之间的超覆盖直线，障碍角之间不能斜穿。"""

        x, y = start
        end_x, end_y = goal
        dx = end_x - x
        dy = end_y - y
        count_x = abs(dx)
        count_y = abs(dy)
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        moved_x = 0
        moved_y = 0
        if self._blocked((x, y)):
            return False

        while moved_x < count_x or moved_y < count_y:
            decision = (1 + 2 * moved_x) * count_y - (1 + 2 * moved_y) * count_x
            previous = (x, y)
            if decision == 0:
                x += step_x
                y += step_y
                moved_x += 1
                moved_y += 1
                if self._blocked((previous[0] + step_x, previous[1])):
                    return False
                if self._blocked((previous[0], previous[1] + step_y)):
                    return False
            elif decision < 0:
                x += step_x
                moved_x += 1
            else:
                y += step_y
                moved_y += 1
            if self._blocked((x, y)):
                return False
        return True

    def _simplify(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """贪心保留最远可直达点，删除 A* 的锯齿状中间网格。"""

        if len(cells) <= 2:
            return list(cells)
        simplified = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            next_index = len(cells) - 1
            while next_index > anchor + 1 and not self._segment_clear(cells[anchor], cells[next_index]):
                next_index -= 1
            simplified.append(cells[next_index])
            anchor = next_index
        return simplified

    def plan(self, start: Pose2D, goal: Pose2D) -> PathPlan:
        # 起点按「机器人现在就在那儿」处理，终点按「我们打算去那儿」处理——
        # 前者是既成事实，后者是选择，所以只有后者才该被膨胀区否决。
        entry_cell = self._cell(start.x, start.y, allow_inflated=True)
        goal_cell = self._cell(goal.x, goal.y)
        escape = self._escape_path(entry_cell)
        start_cell = escape[-1]
        queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_cell)]
        cost = {start_cell: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}

        while queue:
            _priority, current_cost, current = heapq.heappop(queue)
            if current == goal_cell:
                break
            if current_cost != cost.get(current):
                continue
            for next_cell, step_cost in self._neighbors(current):
                next_cost = current_cost + step_cost
                if next_cost >= cost.get(next_cell, math.inf):
                    continue
                cost[next_cell] = next_cost
                came_from[next_cell] = current
                priority = next_cost + self._heuristic(next_cell, goal_cell)
                heapq.heappush(queue, (priority, next_cost, next_cell))
        if goal_cell not in came_from:
            raise PathNotFound("A* exhausted the static map")

        cells = []
        current: tuple[int, int] | None = goal_cell
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        # 逃逸段不参与 _simplify：简化靠 _segment_clear 验证直达，而它按膨胀区
        # 判断可通行，必然否掉这一段。逃逸段本来就是「一格一格挪出去」，原样保留。
        waypoint_cells = escape[:-1] + self._simplify(cells)
        waypoints = [Pose2D(start.x, start.y, start.yaw)]
        for cell in waypoint_cells[1:-1]:
            x, y = self._center(cell)
            waypoints.append(Pose2D(x, y, 0.0))
        waypoints.append(goal)
        escape_cost = self._trail_cost(escape)
        return PathPlan(
            tuple(waypoints),
            tuple(escape[:-1] + cells),
            (cost[goal_cell] + escape_cost) * self.map.resolution,
            tuple(waypoint_cells),
        )

    @staticmethod
    def _trail_cost(trail: list[tuple[int, int]]) -> float:
        """格子序列的欧氏长度，单位是格。斜走算 √2，与 A* 的步长口径一致。"""

        return sum(
            math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            for cur, nxt in zip(trail, trail[1:])
        )


def path_plan_svg(
    static_map: StaticMap,
    plan: PathPlan,
    *,
    demo: bool = False,
    dock_goal: Pose2D | None = None,
    status_label: str | None = None,
) -> str:
    """生成可直接在 VS Code 打开的路径图，不依赖绘图库。"""

    canvas = 720.0
    padding = 52.0
    bounds = static_map.bounds
    world_width = bounds.max_x - bounds.min_x
    world_height = bounds.max_y - bounds.min_y
    scale = min((canvas - 2 * padding) / world_width, (canvas - 2 * padding) / world_height)

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            padding + (x - bounds.min_x) * scale,
            canvas - padding - (y - bounds.min_y) * scale,
        )

    def rectangle(rect: AxisAlignedRect, color: str, opacity: float) -> str:
        left, top = point(rect.min_x, rect.max_y)
        width = (rect.max_x - rect.min_x) * scale
        height = (rect.max_y - rect.min_y) * scale
        return (
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" '
            f'height="{height:.1f}" fill="{color}" fill-opacity="{opacity}" '
            'stroke="#374151" stroke-width="1"/>'
        )

    raw_points = " ".join(
        f"{point(bounds.min_x + (cell[0] + 0.5) * static_map.resolution, bounds.min_y + (cell[1] + 0.5) * static_map.resolution)[0]:.1f},"
        f"{point(bounds.min_x + (cell[0] + 0.5) * static_map.resolution, bounds.min_y + (cell[1] + 0.5) * static_map.resolution)[1]:.1f}"
        for cell in plan.grid_cells
    )
    route_points = " ".join(f"{point(item.x, item.y)[0]:.1f},{point(item.x, item.y)[1]:.1f}" for item in plan.waypoints)
    start = plan.waypoints[0]
    transit_goal = plan.waypoints[-1]
    goal = transit_goal if dock_goal is None else dock_goal
    start_x, start_y = point(start.x, start.y)
    goal_x, goal_y = point(goal.x, goal.y)
    arrow_x = goal_x + math.cos(goal.yaw) * 34.0
    arrow_y = goal_y - math.sin(goal.yaw) * 34.0
    banner = status_label or (
        "DEMO：非标定数据" if demo else "人工确认配置"
    )

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="720" viewBox="0 0 720 720">',
        '<rect width="720" height="720" fill="#f8fafc"/>',
        f'<text x="52" y="30" font-family="sans-serif" font-size="18" font-weight="700">DG-202612 A* 路径（{banner}）</text>',
        rectangle(bounds, "#ffffff", 1.0),
    ]
    elements.extend(rectangle(item, "#64748b", 0.35) for item in static_map.obstacles)
    elements.extend(rectangle(item, "#ef4444", 0.18) for item in static_map.inflated_obstacles)
    if dock_goal is not None:
        transit_x, transit_y = point(transit_goal.x, transit_goal.y)
        elements.append(
            f'<line x1="{transit_x:.1f}" y1="{transit_y:.1f}" '
            f'x2="{goal_x:.1f}" y2="{goal_y:.1f}" '
            'stroke="#f59e0b" stroke-width="6"/>'
        )
    elements.extend(
        (
            f'<polyline points="{raw_points}" fill="none" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4 5"/>',
            f'<polyline points="{route_points}" fill="none" stroke="#2563eb" stroke-width="5" stroke-linejoin="round"/>',
            f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="8" fill="#16a34a"/>',
            f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="8" fill="#dc2626"/>',
            f'<line x1="{goal_x:.1f}" y1="{goal_y:.1f}" x2="{arrow_x:.1f}" y2="{arrow_y:.1f}" stroke="#dc2626" stroke-width="4"/>',
            '<text x="52" y="700" font-family="sans-serif" font-size="14" fill="#334155">灰虚线：原始网格　蓝线：A*　橙线：矩形底盘精确停靠　红色：膨胀障碍</text>',
            '</svg>',
        )
    )
    return "\n".join(elements)


def operating_stance(
    target: ObjectState,
    approach_direction: tuple[float, float],
    standoff: float,
) -> Pose2D:
    """从实时箱体位姿生成站位；方向表示“底盘到箱体”的世界系单位方向。"""

    dx, dy = approach_direction
    length = math.hypot(dx, dy)
    if length == 0.0 or standoff <= 0.0:
        raise ValueError("approach direction and standoff must be non-zero")
    dx /= length
    dy /= length
    return Pose2D(
        target.pose.x - dx * standoff,
        target.pose.y - dy * standoff,
        math.atan2(dy, dx),
    )


class PathFollower:
    """可解释的路径跟踪器；真正发布前仍须经过现有 SafetyGateway。"""

    def __init__(
        self,
        position_tolerance: float,
        yaw_tolerance: float,
        max_linear: float,
        max_angular: float,
        hold_final_yaw: bool = False,
    ) -> None:
        if min(position_tolerance, yaw_tolerance, max_linear, max_angular) <= 0.0:
            raise ValueError("follower limits and tolerances must be positive")
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.hold_final_yaw = hold_final_yaw
        self._index = 0

    @staticmethod
    def _wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def update(self, current: Pose2D, plan: PathPlan) -> BaseVelocity:
        while self._index < len(plan.waypoints) - 1:
            point = plan.waypoints[self._index]
            if math.hypot(point.x - current.x, point.y - current.y) > self.position_tolerance:
                break
            self._index += 1
        target = plan.waypoints[self._index]
        distance = math.hypot(target.x - current.x, target.y - current.y)
        final_leg = self._index == len(plan.waypoints) - 1
        if final_leg and self.hold_final_yaw:
            return self._hold_yaw_command(current, target, distance)
        if final_leg and distance <= self.position_tolerance:
            yaw_error = self._wrap(target.yaw - current.yaw)
            if abs(yaw_error) <= self.yaw_tolerance:
                return BaseVelocity(0.0, 0.0, True)
            return BaseVelocity(0.0, max(-self.max_angular, min(self.max_angular, yaw_error)), False)
        heading = math.atan2(target.y - current.y, target.x - current.x)
        yaw_error = self._wrap(heading - current.yaw)
        angular = max(-self.max_angular, min(self.max_angular, yaw_error))
        linear = 0.0 if abs(yaw_error) > 0.35 else min(self.max_linear, distance)
        return BaseVelocity(linear, angular, False)

    def _hold_yaw_command(
        self,
        current: Pose2D,
        target: Pose2D,
        distance: float,
    ) -> BaseVelocity:
        """终段控制律：**始终对齐目标朝向**，只沿这个朝向进退。

        与默认的"朝向目标点"制导（``heading = atan2(Δy, Δx)``）区别在参考角的
        取法，解决的是后者在近距离处的一个固有毛病——**航向参考对横向偏差的
        敏感度随距离趋零而发散**::

            航向偏差 ≈ 横向偏差 / 剩余距离

        DG202612 的 DOCK 精调段实测正是撞在这上面：底盘停在目标正后方 130 mm、
        横偏只有 4 mm，``atan2(0.025, −0.004) = 1.73`` rad，比目标朝向 1.571
        整整偏出 0.15 rad。跟随器于是先把车头拧过去 0.15 rad 去纠那 4 mm，等
        走进位置容差再拧回来。横偏本身远小于 30 mm 的位置容差、根本不需要纠，
        这一来一回却要多花约 3 秒（P 增益 1.0 时 ln(0.15/0.008) ≈ 2.9 个时间
        常数），把 30 s 的执行预算耗掉十分之一，还让最终朝向精度全看这一拧
        能不能收干净。

        **为什么可以直接不纠横偏。** 差速底盘没有横移自由度，纠 4 mm 横偏的
        唯一办法就是"转过去-开一段-转回来"，代价是整个朝向精度。而横偏的验收
        本来就归位置容差管：留着它不动，位置误差仍在容差内；硬去纠，反而把
        更贵的朝向精度赔进去。**便宜的自由度不该拿贵的去换。**

        于是本模式把两件事解耦：
          * 角速度只负责朝向——参考角恒为 ``target.yaw``，不受剩余距离影响；
          * 线速度只负责纵向——把位置误差投影到目标朝向上，投影为负说明开过头，
            允许倒车退回来。

        横偏由此完全不参与控制，但**仍然参与验收**：``finished`` 用的是
        ``distance``（含横向分量），横偏大到超出位置容差时这一段就永远不会
        判成到位，最终以超时失败暴露出来——而不是悄悄拿朝向去换位置。
        真要压横偏，该在长距离的 transit 段解决，那里距离大、航向制导好用。
        """

        yaw_error = self._wrap(target.yaw - current.yaw)
        if distance <= self.position_tolerance and abs(yaw_error) <= self.yaw_tolerance:
            return BaseVelocity(0.0, 0.0, True)
        angular = max(-self.max_angular, min(self.max_angular, yaw_error))
        if distance <= self.position_tolerance:
            # 位置已达标，只剩朝向要收：停下来原地转，避免边走边转把位置带出容差。
            return BaseVelocity(0.0, angular, False)
        forward = (target.x - current.x) * math.cos(target.yaw) + (
            target.y - current.y
        ) * math.sin(target.yaw)
        linear = max(-self.max_linear, min(self.max_linear, forward))
        return BaseVelocity(linear, angular, False)


class DockController:
    """在最终位姿持续稳定一段时间后才确认 DOCK 完成。"""

    def __init__(
        self,
        route: DockRoute,
        follower: PathFollower,
        stable_duration: float = 1.0,
    ) -> None:
        if stable_duration <= 0.0:
            raise ValueError("stable_duration must be positive")
        self.route = route
        self.transit_follower = follower
        # 精调段单独构造一个跟随器，并且**只有它**启用终段姿态锁定。
        # transit 段不能启用：那一段要跨越一米以上，靠的正是"朝向目标点"的航向
        # 制导把底盘开过去；姿态锁定模式只沿目标朝向进退，走不了折线。
        # 两段的分工因此很清楚——transit 负责把位置和横偏收进容差，precision
        # 负责在不牺牲位置的前提下把朝向收干净。
        self.precision_follower = PathFollower(
            follower.position_tolerance,
            follower.yaw_tolerance,
            follower.max_linear,
            follower.max_angular,
            hold_final_yaw=True,
        )
        self.precision_plan = PathPlan(
            (route.transit.waypoints[-1], route.goal),
            (),
            math.hypot(
                route.goal.x - route.transit.waypoints[-1].x,
                route.goal.y - route.transit.waypoints[-1].y,
            ),
            (),
        )
        self.stable_duration = stable_duration
        self._stable_since: float | None = None
        self._phase = "transit"

    def update(self, current: Pose2D, now: float) -> DockProgress:
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if self._phase == "transit":
            command = self.transit_follower.update(
                current,
                self.route.transit,
            )
            if command.finished:
                self._phase = "precision"
                command = BaseVelocity(0.0, 0.0, False)
        else:
            command = self.precision_follower.update(
                current,
                self.precision_plan,
            )
        goal = self.route.goal
        position_error = math.hypot(goal.x - current.x, goal.y - current.y)
        yaw_error = abs(
            math.atan2(
                math.sin(goal.yaw - current.yaw),
                math.cos(goal.yaw - current.yaw),
            )
        )
        if command.finished:
            if self._stable_since is None:
                self._stable_since = now
            stable_for = max(0.0, now - self._stable_since)
        else:
            self._stable_since = None
            stable_for = 0.0
        completed = stable_for >= self.stable_duration
        return DockProgress(
            BaseVelocity(
                0.0 if completed else command.linear,
                0.0 if completed else command.angular,
                completed,
            ),
            position_error,
            yaw_error,
            stable_for,
            completed,
            self._phase,
        )


class StraightRetreatController:
    """抱着箱子沿车头反方向直线倒退，朝向锁死不变。

    对应官方 ``client_task_1.py`` 的 ``do_reverse()``，那一段的全部逻辑是：

        yaw_err = wrap_to_pi(yaw_ref - self.base_yaw)
        cur = self.base_xy[axis]
        if (sign > 0 and cur < limit) or (sign < 0 and cur > limit):
            self.set_twist(-0.35, 1.0 * yaw_err)
        else:
            self.set_twist(0.0, 0.0)

    本类与官方的两处**有意**不同，都写在这里，方便对照：

    **一、进度用投影量，不按坐标轴取分量。** 官方的 ``reverse_target_for_yaw()``
    只认两个朝向（正北、正西），别的朝向直接 ``raise``，因为它是按"哪个坐标轴、
    往哪个方向"来记终点的。随机场景里站位朝向不一定卡在这两个值上，所以这里改
    成沿倒退方向做投影：

        travelled = (当前位置 − 起点) · (车头方向的反方向)

    朝向正好是正北/正西时，这个投影退化成官方那两条分支的坐标差，数值完全一
    致；朝向是别的角度时，官方会拒绝，这里照样算得出来。

    **二、横向漂移会中止倒退。** 官方不查这一项。我们查，是因为计划阶段的走廊
    校验（:func:`plan_straight_retreat`）是沿**一条直线**扫机器人外形扫出来的：
    机器人横着漂出去以后，它实际经过的地方根本没被检查过，那份"不会撞"的结论
    就作废了。这时候正确做法是停车报错让人来看，而不是硬着头皮退完。

    **偏航反馈的符号为什么不用翻。** 这是倒退控制里最容易写错的一处。翻符号的
    直觉来自"倒车时方向盘要反打"——但那说的是**用朝向去纠正横向位置**：车尾往
    左偏了，前进时该往右打，倒退时就得往左打。这里的反馈回路不是那个：
    ``angular`` 直接改变的就是 yaw 本身，而 yaw 的运动方程 ``yaw' = ω`` 里根本
    没有线速度，前进后退一个样。所以"yaw 比目标小就给正角速度"在倒退时依然成
    立，符号照抄官方的 ``+1.0 * yaw_err``，不翻。
    """

    def __init__(
        self,
        route: RetreatRoute,
        *,
        speed: float,
        yaw_gain: float,
        max_angular: float,
        lateral_tolerance: float,
    ) -> None:
        if min(speed, yaw_gain, max_angular, lateral_tolerance) <= 0.0:
            raise ValueError("retreat speed, gain, angular limit and tolerance must be positive")
        self.route = route
        self.speed = speed
        self.yaw_gain = yaw_gain
        self.max_angular = max_angular
        self.lateral_tolerance = lateral_tolerance
        # 车头方向的反方向（倒退方向），以及它左手边的法向。两者都是单位向量，
        # 构成一组正交基：任何位移都能唯一分解成"退了多远"加"横着漂了多远"。
        yaw = route.goal.yaw
        self._backward = (-math.cos(yaw), -math.sin(yaw))
        self._lateral = (-math.sin(yaw), math.cos(yaw))

    @staticmethod
    def _wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def update(self, current: Pose2D) -> RetreatProgress:
        offset = (current.x - self.route.start.x, current.y - self.route.start.y)
        travelled = offset[0] * self._backward[0] + offset[1] * self._backward[1]
        lateral = offset[0] * self._lateral[0] + offset[1] * self._lateral[1]
        # remaining 允许为负：退过头了就是负数。判完成用的是 travelled 与目标距离
        # 的比较，负的 remaining 只会出现在里程计跳变时，此时如实报出来比夹到 0
        # 更有诊断价值。
        remaining = self.route.distance - travelled
        yaw_error = self._wrap(self.route.goal.yaw - current.yaw)
        if abs(lateral) > self.lateral_tolerance:
            return RetreatProgress(
                BaseVelocity(0.0, 0.0, False),
                travelled,
                remaining,
                lateral,
                yaw_error,
                False,
                (
                    f"lateral drift {abs(lateral):.3f} m exceeds the "
                    f"{self.lateral_tolerance:.3f} m corridor that was checked"
                ),
            )
        if travelled >= self.route.distance:
            return RetreatProgress(
                BaseVelocity(0.0, 0.0, True),
                travelled,
                remaining,
                lateral,
                yaw_error,
                True,
            )
        angular = max(
            -self.max_angular,
            min(self.max_angular, self.yaw_gain * yaw_error),
        )
        return RetreatProgress(
            BaseVelocity(-self.speed, angular, False),
            travelled,
            remaining,
            lateral,
            yaw_error,
            False,
        )


def path_plan_dict(plan: PathPlan) -> dict[str, Any]:
    """把路径完整写入计划；执行端不重新规划或猜测路点。"""

    return {
        "waypoints": [
            {"x": item.x, "y": item.y, "yaw": item.yaw}
            for item in plan.waypoints
        ],
        "grid_cells": [list(item) for item in plan.grid_cells],
        "waypoint_cells": [list(item) for item in plan.waypoint_cells],
        "cost_m": plan.cost,
    }


def path_plan_from_dict(value: Mapping[str, Any]) -> PathPlan:
    """严格读取已审核路径，不接受缺字段或非有限坐标。"""

    def finite(item: Any, field: str) -> float:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field} must be finite")
        return number

    waypoints_raw = value.get("waypoints")
    if not isinstance(waypoints_raw, list) or len(waypoints_raw) < 2:
        raise ValueError("path waypoints must contain at least start and goal")
    waypoints = []
    for index, item in enumerate(waypoints_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"path waypoint {index} must be an object")
        waypoints.append(
            Pose2D(
                finite(item.get("x"), f"waypoints[{index}].x"),
                finite(item.get("y"), f"waypoints[{index}].y"),
                finite(item.get("yaw"), f"waypoints[{index}].yaw"),
            )
        )

    def cells(field: str) -> tuple[tuple[int, int], ...]:
        raw = value.get(field)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"path {field} must be a non-empty list")
        result = []
        for index, item in enumerate(raw):
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or isinstance(item[0], bool)
                or isinstance(item[1], bool)
            ):
                raise ValueError(f"{field}[{index}] must contain two integers")
            first, second = int(item[0]), int(item[1])
            if first != item[0] or second != item[1]:
                raise ValueError(f"{field}[{index}] must contain two integers")
            result.append((first, second))
        return tuple(result)

    return PathPlan(
        tuple(waypoints),
        cells("grid_cells"),
        finite(value.get("cost_m"), "cost_m"),
        cells("waypoint_cells"),
    )


def dock_route_dict(route: DockRoute) -> dict[str, Any]:
    return {
        "transit": path_plan_dict(route.transit),
        "goal": {
            "x": route.goal.x,
            "y": route.goal.y,
            "yaw": route.goal.yaw,
        },
    }


def dock_route_from_dict(value: Mapping[str, Any]) -> DockRoute:
    transit = value.get("transit")
    goal = value.get("goal")
    if not isinstance(transit, Mapping) or not isinstance(goal, Mapping):
        raise ValueError("dock route requires transit and goal objects")
    try:
        goal_pose = Pose2D(
            float(goal.get("x")),
            float(goal.get("y")),
            float(goal.get("yaw")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("dock route goal must contain finite x, y and yaw") from exc
    return DockRoute(path_plan_from_dict(transit), goal_pose)


def _polygons_overlap(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> bool:
    """用分离轴定理检查两个凸四边形。"""

    for polygon in (first, second):
        for start, end in zip(polygon, (*polygon[1:], polygon[0])):
            axis = (-(end[1] - start[1]), end[0] - start[0])
            first_projection = tuple(x * axis[0] + y * axis[1] for x, y in first)
            second_projection = tuple(x * axis[0] + y * axis[1] for x, y in second)
            if max(first_projection) <= min(second_projection):
                return False
            if max(second_projection) <= min(first_projection):
                return False
    return True


def _rect_corners(rectangle: AxisAlignedRect) -> tuple[tuple[float, float], ...]:
    return (
        (rectangle.min_x, rectangle.min_y),
        (rectangle.max_x, rectangle.min_y),
        (rectangle.max_x, rectangle.max_y),
        (rectangle.min_x, rectangle.max_y),
    )


def plan_dock_route(
    planner: StaticAStarPlanner,
    start: Pose2D,
    goal: Pose2D,
    footprint: BaseFootprint,
    *,
    final_approach_distance: float,
    footprint_clearance: float,
    sample_step: float = 0.01,
) -> DockRoute:
    """A* 到预停靠点，再验证保持最终朝向的短直线停靠段。"""

    if min(final_approach_distance, sample_step) <= 0.0:
        raise ValueError("docking distances must be positive")
    direction = (math.cos(goal.yaw), math.sin(goal.yaw))
    pre_dock = Pose2D(
        goal.x - direction[0] * final_approach_distance,
        goal.y - direction[1] * final_approach_distance,
        goal.yaw,
    )
    transit = planner.plan(start, pre_dock)
    checked_footprint = footprint.inflated(footprint_clearance)
    samples = max(1, math.ceil(final_approach_distance / sample_step))
    physical_bounds = planner.map.bounds
    for index in range(samples + 1):
        ratio = index / samples
        pose = Pose2D(
            pre_dock.x + (goal.x - pre_dock.x) * ratio,
            pre_dock.y + (goal.y - pre_dock.y) * ratio,
            goal.yaw,
        )
        robot_polygon = checked_footprint.world_corners(pose)
        if any(
            not physical_bounds.contains(x, y)
            for x, y in robot_polygon
        ):
            raise PathNotFound("precision docking footprint leaves the field")
        if any(
            _polygons_overlap(robot_polygon, _rect_corners(obstacle))
            for obstacle in planner.map.obstacles
        ):
            raise PathNotFound(
                "precision docking footprint overlaps a fixed obstacle"
            )
    return DockRoute(transit, goal)


def plan_straight_retreat(
    planner: StaticAStarPlanner,
    start: Pose2D,
    distance: float,
    footprint: BaseFootprint,
    *,
    footprint_clearance: float,
    sample_step: float = 0.01,
) -> RetreatRoute:
    """沿车头反方向退 ``distance`` 米，逐点验证整段走廊装得下机器人。

    **为什么不走 A*。** A* 会为了绕开障碍而拐弯，而这一步机器人怀里抱着箱子，
    只靠两侧指垫的摩擦力挂着——任何转向都是在甩它。所以这里的可行性问题不是
    "能不能找到一条路"，而是"这条唯一允许走的直线走廊，够不够宽"。找不到路时
    正确的反应是拒绝规划（抛 :class:`PathNotFound`），让上层换一个站位重来，
    而不是自作主张绕一下。

    **为什么用真实矩形外形而不是膨胀栅格。** DOCK 完成时机器人是贴着桌子停的，
    它此刻**就在**膨胀区里（膨胀半径 0.35 m，比桌子到底盘的实际间隙大得多）。
    拿膨胀栅格去查，第一格就判死。这和 :func:`plan_dock_route` 的精调段是同一
    个道理，所以复用同一套做法：按 ``sample_step`` 把走廊切碎，每一点都拿加了
    ``footprint_clearance`` 余量的真实矩形去和障碍物做分离轴检测。

    **起点单独检查，且报不同的错。** 起点重叠和沿途重叠是两种故障，上层该做的
    事不一样：

    * 起点就压着障碍 ⇒ 机器人现在待的地方按地图算是实心的，也就是**实测位姿和
      地图对不上**。这时整段走廊的结论全都不可信，该停机让人来看。
    * 沿途撞上障碍 ⇒ 位姿没问题，是这个站位退不出去。换个站位重来就行。

    合成一条消息的话，上层只能一律停机，白白丢掉"换站位重试"这条出路。DOCK 的
    精调段不需要区分（它的起点刚被 A* 判过），倒退需要——倒退的起点是上一步执行
    完留下的实测位姿，在此之前没有任何规划器检查过它。

    **为什么沿途要一点一点扫，而不是只查两端。** 当前参数下（底盘外形约 0.5 m
    长，退 0.35 m）两端的外形本来就重叠，中间点被它们完全盖住，只查两端恰好等
    价于查全程。但这是**当前尺寸的巧合**，不是几何规律：随机场景里一旦退得比车
    身长，两端之间就会露出一段没被检查过的缝隙。逐点扫描守的是这种情况。
    """

    if min(distance, sample_step) <= 0.0:
        raise ValueError("retreat distance and sample step must be positive")
    backward = (-math.cos(start.yaw), -math.sin(start.yaw))
    goal = Pose2D(
        start.x + backward[0] * distance,
        start.y + backward[1] * distance,
        start.yaw,
    )
    checked_footprint = footprint.inflated(footprint_clearance)
    physical_bounds = planner.map.bounds

    def obstructed(pose: Pose2D) -> str | None:
        """这一点放不放得下机器人；放不下就说明是哪一种放不下。"""

        robot_polygon = checked_footprint.world_corners(pose)
        if any(not physical_bounds.contains(x, y) for x, y in robot_polygon):
            return "leaves the field"
        if any(
            _polygons_overlap(robot_polygon, _rect_corners(obstacle))
            for obstacle in planner.map.obstacles
        ):
            return "overlaps a fixed obstacle"
        return None

    blocked = obstructed(start)
    if blocked is not None:
        raise PathNotFound(f"retreat start pose {blocked}")
    samples = max(1, math.ceil(distance / sample_step))
    for index in range(1, samples + 1):
        ratio = index / samples
        pose = Pose2D(
            start.x + backward[0] * distance * ratio,
            start.y + backward[1] * distance * ratio,
            start.yaw,
        )
        blocked = obstructed(pose)
        if blocked is not None:
            raise PathNotFound(f"retreat footprint {blocked}")
    return RetreatRoute(start, goal, distance)


def retreat_route_dict(route: RetreatRoute) -> dict[str, Any]:
    return {
        "start": {"x": route.start.x, "y": route.start.y, "yaw": route.start.yaw},
        "goal": {"x": route.goal.x, "y": route.goal.y, "yaw": route.goal.yaw},
        "distance_m": route.distance,
    }


def retreat_route_from_dict(value: Mapping[str, Any]) -> RetreatRoute:
    """严格读取已审核的倒退段；缺字段、非有限值、非正距离一律拒绝。

    执行端只从计划里读这段路，绝不自己重算——重算就等于绕过了审核。
    """

    def pose(field: str) -> Pose2D:
        item = value.get(field)
        if not isinstance(item, Mapping):
            raise ValueError(f"retreat route requires a {field} object")
        try:
            numbers = tuple(float(item.get(key)) for key in ("x", "y", "yaw"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"retreat route {field} must contain finite x, y and yaw") from exc
        if not all(math.isfinite(number) for number in numbers):
            raise ValueError(f"retreat route {field} must contain finite x, y and yaw")
        return Pose2D(*numbers)

    try:
        distance = float(value.get("distance_m"))
    except (TypeError, ValueError) as exc:
        raise ValueError("retreat route requires a finite distance_m") from exc
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("retreat route distance_m must be positive and finite")
    return RetreatRoute(pose("start"), pose("goal"), distance)
