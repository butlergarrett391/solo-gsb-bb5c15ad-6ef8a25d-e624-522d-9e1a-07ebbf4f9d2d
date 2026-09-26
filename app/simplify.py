"""轨迹精简：Douglas-Peucker 算法按原始 trkseg 独立抽稀。

- 每个原始 trkseg 独立处理，精简结果不跨分段边界连线；
- 将该段经纬度投影到以段首点为原点的局部等距圆柱平面（米），
  以点到**有限线段**的最短距离选取最大偏差点，仅当最大偏差**超过**阈值时才拆分；
- 分段首尾点重合（弦长为零）时，偏差退化为点到端点的距离；
- 保留点直接引用原始轨迹点对象，经纬度、海拔、时间原样保留，不做插值；
- 单点段、两点段原样保留；重复点（零长度弦）按点到端点距离处理；
- 迭代实现（显式栈代替递归），十万点输入不会触发递归栈溢出。
"""
from __future__ import annotations

from dataclasses import dataclass

from .geo import to_local_xy_m

MIN_TOLERANCE_M = 1.0
MAX_TOLERANCE_M = 100.0
DEFAULT_TOLERANCE_M = 10.0


@dataclass
class SimplifyResult:
    """轨迹精简结果；simplified_segments 与输入 segments 结构一致（逐段对应）。"""

    simplified_segments: list  # list[list[TrackPoint]]
    points_before: int = 0
    points_after: int = 0


def simplify_track(segments: list, tolerance_m: float = DEFAULT_TOLERANCE_M) -> SimplifyResult:
    """对每个原始 trkseg 分别做 Douglas-Peucker 精简。

    返回精简后的分段（每段保留首尾点，分段间互不连接）与精简前后点数；
    tolerance_m 超出 1～100 米范围时抛出 ValueError，由接口层转为提示。
    """
    if not (MIN_TOLERANCE_M <= tolerance_m <= MAX_TOLERANCE_M):
        raise ValueError(
            f"误差阈值须在 {MIN_TOLERANCE_M:.0f}～{MAX_TOLERANCE_M:.0f} 米之间"
        )
    simplified = [_simplify_segment(seg, tolerance_m) for seg in segments]
    return SimplifyResult(
        simplified_segments=simplified,
        points_before=sum(len(s) for s in segments),
        points_after=sum(len(s) for s in simplified),
    )


def _simplify_segment(seg: list, tolerance_m: float) -> list:
    n = len(seg)
    if n <= 2:  # 空段 / 单点段 / 两点段：没有可精简的中间点，原样保留
        return list(seg)

    # 以段首点为原点投影到局部米制平面，后续所有距离计算都在该平面内进行
    origin = seg[0]
    xs: list = []
    ys: list = []
    for p in seg:
        x, y = to_local_xy_m(p, origin)
        xs.append(x)
        ys.append(y)

    keep = [False] * n
    keep[0] = keep[n - 1] = True
    tol2 = tolerance_m * tolerance_m  # 全程比较平方距离，避免逐点开方
    stack = [(0, n - 1)]  # 待处理的索引区间（显式栈，避免递归）
    while stack:
        lo, hi = stack.pop()
        ax, ay = xs[lo], ys[lo]
        dx, dy = xs[hi] - ax, ys[hi] - ay
        seg_len2 = dx * dx + dy * dy
        max_d2 = -1.0
        max_i = -1
        for i in range(lo + 1, hi):
            px, py = xs[i] - ax, ys[i] - ay
            if seg_len2 == 0.0:
                # 首尾重合（零长度弦）：退化为点到端点的距离
                d2 = px * px + py * py
            else:
                # 点到有限线段：投影参数截断到 [0, 1]
                t = (px * dx + py * dy) / seg_len2
                t = max(0.0, min(1.0, t))
                ex = px - t * dx
                ey = py - t * dy
                d2 = ex * ex + ey * ey
            if d2 > max_d2:
                max_d2 = d2
                max_i = i
        if max_i >= 0 and max_d2 > tol2:  # 超过阈值才继续拆分
            keep[max_i] = True
            stack.append((lo, max_i))
            stack.append((max_i, hi))
    return [p for p, k in zip(seg, keep) if k]
