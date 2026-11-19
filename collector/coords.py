"""物理坐标选择与分页规划。

目录轴是物理坐标（time/sample 为实验开始后的秒，station 为测点坐标），
可能升序也可能降序，sample 轴不保证等间隔。本模块只做“坐标 -> 该版目录内的索引”
的确定性翻译，绝不按数组形状推定对齐。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AxisSelection:
    """单个轴上的物理坐标闭区间选择。

    lo/hi 为请求的物理坐标；indices 是该版目录轴上命中的连续索引（左闭右开区间）；
    selected 是对应的实际坐标；gaps 是请求范围内未提供观测的物理坐标区间。
    """

    name: str
    lo: float
    hi: float
    indices: tuple[int, int]
    selected: list[float]
    gaps: list[tuple[float, float]]

    @property
    def empty(self) -> bool:
        return self.indices[0] == self.indices[1]


def select_indices(name: str, axis: list[float], lo: float, hi: float) -> AxisSelection:
    """把物理坐标闭区间 [lo, hi] 翻译为目录轴上的连续索引区间。

    轴允许降序；非等间隔轴照常处理。返回结构中同时给出实际命中坐标与缺口：
    缺口区分“超出轴覆盖范围”和“轴覆盖范围内但没有采样点的空隙”。
    """
    if lo > hi:
        raise ValueError(f"轴 {name} 的选择下界 {lo} 大于上界 {hi}")
    values = [float(v) for v in axis]
    ordered = sorted(values)

    hits = [v for v in ordered if lo <= v <= hi]
    gaps: list[tuple[float, float]] = []

    # 缺口只描述“请求范围内来源未提供观测”的坐标：轴覆盖范围之外的请求区间。
    # 相邻采样点之间的采样间隔不是缺口——非等间隔是轴本身的属性，
    # 由 describe_axis 单独标注，避免把合法采样轴误报成缺失。
    if not hits:
        gaps.append((lo, hi))
    else:
        first, last = hits[0], hits[-1]
        # 缺口端点沿用请求边界：例如请求 [1.5, 9] 而轴上只有坐标 2，
        # 缺口为 [1.5, 2) 与 (2, 9]，明确指出哪些请求坐标没有观测。
        if lo < first:
            gaps.append((lo, first))
        if hi > last:
            gaps.append((last, hi))

    if hits:
        positions = sorted(values.index(v) for v in hits)
        start, stop = positions[0], positions[-1] + 1
        # 命中值在轴上必须连续（选择的是一个物理坐标区间）。
        if positions != list(range(start, stop)):
            raise RuntimeError(f"轴 {name} 非单调，无法按坐标区间选择")
        selected = values[start:stop]
    else:
        start = stop = 0
        selected = []

    return AxisSelection(
        name=name,
        lo=float(lo),
        hi=float(hi),
        indices=(start, stop),
        selected=selected,
        gaps=gaps,
    )


def parse_selection(text: str) -> dict[str, tuple[float, float]]:
    """解析 ``axis=lo:hi`` 形式的选择，多轴以逗号分隔。"""
    result: dict[str, tuple[float, float]] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part or ":" not in part:
            raise ValueError(f"选择片段 {part!r} 应为 axis=lo:hi 形式")
        axis, span = part.split("=", 1)
        lo_text, hi_text = span.split(":", 1)
        lo, hi = float(lo_text), float(hi_text)
        result[axis.strip()] = (lo, hi)
    if not result:
        raise ValueError("未给出任何坐标选择")
    return result


def plan_pages(
    dimensions: list[str],
    ranges: dict[str, tuple[int, int]],
    max_cells: int,
) -> list[dict[str, tuple[int, int]]]:
    """把索引空间切成单次单元数不超过 max_cells 的连续页面。

    服务不会替调用方分页，超限直接 413。切分沿各维递归进行，
    页面边界完全由目录与 max_cells 决定，因此重跑得到同一组页面，
    不会在边界处重复拼入单元。空选择返回空列表（合法情形，不发请求）。
    """
    lengths = [ranges[d][1] - ranges[d][0] for d in dimensions]
    if any(n < 0 for n in lengths):
        raise ValueError("索引区间非法")
    if any(n == 0 for n in lengths):
        return []

    pages: list[dict[str, tuple[int, int]]] = []

    def split(dim_pos: int, prefix: dict[str, tuple[int, int]]) -> None:
        if dim_pos == len(dimensions):
            pages.append(dict(prefix))
            return
        dim = dimensions[dim_pos]
        start, stop = ranges[dim]
        remaining = max_cells
        for d in dimensions[:dim_pos]:
            remaining //= prefix[d][1] - prefix[d][0]
        block = max(1, remaining)
        cur = start
        while cur < stop:
            nxt = min(cur + block, stop)
            prefix[dim] = (cur, nxt)
            split(dim_pos + 1, prefix)
            cur = nxt

    split(0, {})
    return pages


def describe_axis(axis: list[float]) -> dict[str, object]:
    """生成轴的可记录描述，显式标注方向与非等间隔事实。"""
    values = [float(v) for v in axis]
    diffs = [b - a for a, b in zip(values, values[1:])]
    descending = bool(diffs) and all(d < 0 for d in diffs)
    uniform = bool(diffs) and all(abs(d - diffs[0]) < 1e-12 for d in diffs)
    return {
        "coordinates": values,
        "descending": descending,
        "uniform_spacing": uniform,
    }


def guard_distinct_grids(
    name_a: str,
    dims_a: tuple[str, ...],
    shape_a: tuple[int, ...],
    name_b: str,
    dims_b: tuple[str, ...],
    shape_b: tuple[int, ...],
) -> None:
    """形状相同不代表网格对齐：维度名不同即拒绝按位置互相对齐。

    采集结果各自携带自己的维度与坐标，调用方若想把两个变量放一起处理，
    必须先显式按坐标连接，而不是按形状广播。
    """
    if shape_a == shape_b and dims_a != dims_b:
        raise ValueError(
            f"{name_a} 维度 {list(dims_a)} 与 {name_b} 维度 {list(dims_b)} "
            f"形状同为 {shape_a}，但采样轴不同，禁止按形状推定对齐"
        )
