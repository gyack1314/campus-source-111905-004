"""可续跑的采集作业：目录快照、切片清单、原子落盘与版次重建。

作业目录中的每份切片都独立记录版次、目录索引区间和物理坐标，
下载中途遇到版次退休时已落盘切片仍然可解释；重跑按物理坐标去重，
不会在分页边界重复拼入单元。
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coords import (
    AxisSelection,
    describe_axis,
    parse_selection,
    plan_pages,
    select_indices,
)
from .protocol import Slice, SourceClient, VersionRetired

JOB_FILE = "job.json"
CATALOG_DIR = "catalogs"
SLICE_DIR = "slices"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode())


def _page_id(variable: str, page: dict[str, tuple[int, int]]) -> str:
    span = "_".join(f"{d}-{s}-{e}" for d, (s, e) in sorted(page.items()))
    return f"{variable}__{span}"


@dataclass
class VariablePlan:
    variable: str
    dimensions: list[str]
    axes: dict[str, AxisSelection]

    @property
    def index_ranges(self) -> dict[str, tuple[int, int]]:
        return {d: self.axes[d].indices for d in self.dimensions}

    @property
    def requested_cells(self) -> int:
        n = 1
        for d in self.dimensions:
            n *= self.axes[d].indices[1] - self.axes[d].indices[0]
        return n


def _axis_coordinates(catalog: dict, dim: str) -> list[float]:
    """在线目录的轴是坐标列表；本地快照的轴是含 coordinates 的描述字典。"""
    axis = catalog["axes"][dim]
    return [float(v) for v in (axis["coordinates"] if isinstance(axis, dict) else axis)]


class JobStore:
    def __init__(self, workdir: str | os.PathLike[str]):
        self.root = Path(workdir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / CATALOG_DIR).mkdir(exist_ok=True)
        (self.root / SLICE_DIR).mkdir(exist_ok=True)
        self.job_path = self.root / JOB_FILE
        if self.job_path.exists():
            self.job = json.loads(self.job_path.read_text(encoding="utf-8"))
        else:
            self.job = None

    # ---------- 作业与目录快照 ----------

    def create(self, selection_text: str, client: SourceClient) -> dict:
        if self.job is not None:
            raise FileExistsError(f"作业已存在：{self.job_path}")
        selection = parse_selection(selection_text)
        catalog = client.get_catalog()  # 不带 version：锚定当前发布版
        version = catalog["version"]
        self._save_catalog(version, catalog)
        self.job = {
            "selection_text": selection_text,
            "selection": {k: list(v) for k, v in selection.items()},
            "status": "planned",
            "variables": {},
            "events": [
                {"event": "job_created", "version": version}
            ],
        }
        for variable, meta in catalog["variables"].items():
            dimensions = list(meta["dimensions"])
            axes = self._build_axes(variable, dimensions, selection, catalog)
            pages = plan_pages(dimensions, axes.index_ranges, int(catalog["max_cells"]))
            self.job["variables"][variable] = {
                "dimensions": dimensions,
                "catalog_version": version,
                "status": "planned",
                "plans": [
                    {
                        "version": version,
                        "index_ranges": {
                            d: list(rng) for d, rng in axes.index_ranges.items()
                        },
                        "axis_selections": {
                            d: {
                                "lo": ax.lo,
                                "hi": ax.hi,
                                "indices": list(ax.indices),
                                "selected": ax.selected,
                                "gaps": [list(g) for g in ax.gaps],
                            }
                            for d, ax in axes.axes.items()
                        },
                        "pages": [
                            {"id": _page_id(variable, page), "index_ranges": page}
                            for page in pages
                        ],
                    }
                ],
                "slices": [],
                "retired_versions": [],
            }
        self._flush()
        return self.job

    def _build_axes(
        self,
        variable: str,
        dimensions: list[str],
        selection: dict[str, tuple[float, float]],
        catalog: dict,
    ) -> VariablePlan:
        axes: dict[str, AxisSelection] = {}
        for dim in dimensions:
            if dim not in selection:
                raise ValueError(
                    f"变量 {variable} 需要轴 {dim} 的物理坐标选择，请求中未提供"
                )
            lo, hi = selection[dim]
            axes[dim] = select_indices(
                dim, _axis_coordinates(catalog, dim), lo, hi
            )
        return VariablePlan(variable, dimensions, axes)

    def _save_catalog(self, version: str, catalog: dict) -> Path:
        path = self.root / CATALOG_DIR / f"{version}.json"
        # 保留完整坐标列表（协议层要按索引切片核对），方向/等间隔作为附加元数据。
        axes = {}
        for name, axis in catalog["axes"].items():
            desc = describe_axis(axis)
            axes[name] = {
                "coordinates": desc["coordinates"],
                "descending": desc["descending"],
                "uniform_spacing": desc["uniform_spacing"],
            }
        snapshot = {
            "version": version,
            "max_cells": catalog["max_cells"],
            "axes": axes,
            "variables": catalog["variables"],
        }
        # 同版目录不可变：若已存在则必须逐字段一致。
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != snapshot:
                raise RuntimeError(f"版次 {version} 的目录快照与既有快照不一致")
        else:
            _atomic_write_json(path, snapshot)
        return path

    def _flush(self) -> None:
        _atomic_write_json(self.job_path, self.job)

    def catalog_snapshot(self, version: str) -> dict:
        return json.loads(
            (self.root / CATALOG_DIR / f"{version}.json").read_text(encoding="utf-8")
        )

    # ---------- 切片落盘与去重 ----------

    @staticmethod
    def _cell_keys(slice_record: dict) -> set[tuple]:
        """按物理坐标标识单元；版次变化导致的索引漂移不会误判为同一单元。"""
        dims = slice_record["dimensions"]
        coords = slice_record["coordinates"]
        grids = np.meshgrid(
            *[np.asarray(coords[d]) for d in dims], indexing="ij"
        )
        flat = [g.reshape(-1) for g in grids]
        return {tuple(float(x[i]) for x in flat) for i in range(flat[0].size)}

    def _covered_cells(self, variable: str) -> set[tuple]:
        covered: set[tuple] = set()
        for record in self.job["variables"][variable]["slices"]:
            covered |= self._cell_keys(record)
        return covered

    def _slice_paths(self, page_id: str) -> tuple[Path, Path]:
        stem = self.root / SLICE_DIR / page_id
        return stem.with_suffix(".npz"), stem.with_suffix(".json")

    def _slice_exists(self, variable: str, page_id: str) -> bool:
        data_path, meta_path = self._slice_paths(page_id)
        if not (data_path.exists() and meta_path.exists()):
            return False
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        return record["variable"] == variable

    def _save_slice(self, variable: str, page: dict, slc: Slice) -> str:
        page_id = page["id"]
        data_path, meta_path = self._slice_paths(page_id)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, raw=slc.raw, quality=slc.quality)
        _atomic_write(data_path, buffer.getvalue())
        record = {
            "page_id": page_id,
            "variable": variable,
            "version": slc.version,
            "dimensions": list(slc.dimensions),
            "coordinates": slc.coordinates,
            "index_ranges": {d: list(rng) for d, rng in slc.index_ranges.items()},
            "scale_factor": slc.scale_factor,
            "add_offset": slc.add_offset,
            "fill_value": slc.fill_value,
            "shape": list(slc.raw.shape),
            "data_file": data_path.name,
        }
        _atomic_write_json(meta_path, record)
        self.job["variables"][variable]["slices"].append(record)
        return page_id

    # ---------- 采集 ----------

    def collect(self, client: SourceClient) -> dict:
        """执行（或续跑）作业。返回作业状态；退休异常会被记录后继续抛出。"""
        if self.job is None:
            raise RuntimeError("作业不存在，请先 create")
        retirement: VersionRetired | None = None
        for variable, state in self.job["variables"].items():
            if state["status"] in ("complete", "mixed_version"):
                continue
            state["status"] = "in_progress"
            self._flush()
            covered = self._covered_cells(variable)
            try:
                for plan in state["plans"]:
                    version = plan["version"]
                    if version in state["retired_versions"]:
                        # 退休版次的未取页面只能由新版计划按物理坐标重建，
                        # 绝不重新解释旧版索引。
                        continue
                    catalog = self.catalog_snapshot(version)
                    for page in plan["pages"]:
                        page_id = page["id"]
                        index_ranges = {
                            d: tuple(rng) for d, rng in page["index_ranges"].items()
                        }
                        if self._slice_exists(variable, page_id):
                            continue  # 重跑：整页已落盘，不重复请求与拼入
                        if page_id in state.get("empty_pages", []):
                            continue  # 重跑：已知合法空页面，不重复请求与登记
                        slc = client.get_slice(
                            version, variable, index_ranges, catalog
                        )
                        if slc is None:
                            # 合法空选择：登记为零单元页面，不产生数据文件。
                            state.setdefault("empty_pages", []).append(page_id)
                            continue
                        keys = self._cell_keys(
                            {
                                "dimensions": slc.dimensions,
                                "coordinates": slc.coordinates,
                            }
                        )
                        if keys <= covered:
                            # 新版在轴中间补点后，旧页面的坐标可能已被其他切片完整覆盖。
                            continue
                        overlap = keys & covered
                        if overlap:
                            raise RuntimeError(
                                f"页面 {page_id} 与既有切片在 {len(overlap)} 个物理坐标上重叠，"
                                "拒绝重复拼入边界单元"
                            )
                        self._save_slice(variable, page, slc)
                        covered |= keys
                        self._flush()
            except VersionRetired as exc:
                retirement = exc
                state["status"] = "retired"
                if version not in state["retired_versions"]:
                    state["retired_versions"].append(version)
                self.job["events"].append(
                    {"event": "version_retired", "variable": variable, "version": version}
                )
                self._flush()
                continue
            self._refresh_variable_status(variable)
            self._flush()
        self._refresh_job_status()
        self._flush()
        if retirement is not None:
            raise retirement
        return self.job

    def rebuild_on_retirement(self, client: SourceClient) -> dict:
        """旧版退休后的重建：读取当前目录，按相同物理坐标重新规划并补齐缺口。

        已落盘的旧版切片原样保留（仍可解释），新计划只取尚未被任何切片
        按物理坐标覆盖的单元；作业最终若同时含两个版次则标记 mixed_version，
        不会被标为 complete。
        """
        if self.job is None:
            raise RuntimeError("作业不存在")
        catalog = client.get_catalog()
        version = catalog["version"]
        self._save_catalog(version, catalog)
        selection = {k: tuple(v) for k, v in self.job["selection"].items()}

        for variable, state in self.job["variables"].items():
            if version in state["retired_versions"]:
                raise VersionRetired(f"当前目录版次 {version} 已在退休清单中")
            dimensions = state["dimensions"]
            axes = self._build_axes(variable, dimensions, selection, catalog)
            pages = plan_pages(dimensions, axes.index_ranges, int(catalog["max_cells"]))
            covered = self._covered_cells(variable)

            pending: list[dict] = []
            for index_ranges in pages:
                # 用目录坐标计算该页覆盖的物理坐标，决定仍需请求的部分。
                for sub_range in self._split_uncovered(
                    dimensions, index_ranges, covered, catalog
                ):
                    page = {
                        "id": _page_id(variable, sub_range),
                        "index_ranges": sub_range,
                    }
                    pending.append(page)

            plan = {
                "version": version,
                "index_ranges": {d: list(rng) for d, rng in axes.index_ranges.items()},
                "axis_selections": {
                    d: {
                        "lo": ax.lo,
                        "hi": ax.hi,
                        "indices": list(ax.indices),
                        "selected": ax.selected,
                        "gaps": [list(g) for g in ax.gaps],
                    }
                    for d, ax in axes.axes.items()
                },
                "pages": pending,
            }
            state["plans"] = [p for p in state["plans"] if p["version"] != version]
            state["plans"].append(plan)
            state["catalog_version"] = version
            state["status"] = "in_progress"
            self._flush()
        return self.collect(client)

    @staticmethod
    def _split_uncovered(dimensions, index_ranges, covered, catalog):
        """把一个规划页面切成若干与已覆盖集合互不重叠的连续子区间。

        旧版切片可能只覆盖新页面中的部分物理坐标（例如只取了某个 station 列）。
        沿维度递归切分：只要当前矩形还与已覆盖集合相交，就沿第一个相关轴二分，
        直到子矩形完全不相交或缩为空。这样未覆盖的单元一个不漏，
        已覆盖单元一个不取，边界不会重复拼入。
        """
        def keys_of(ranges):
            coords = {
                d: _axis_coordinates(catalog, d)[s:e]
                for d, (s, e) in ranges.items()
            }
            return JobStore._cell_keys({"dimensions": dimensions, "coordinates": coords})

        def split(ranges, dim_pos):
            keys = keys_of(ranges)
            if not keys:
                return []
            if not (keys & covered):
                return [dict(ranges)]
            if dim_pos >= len(dimensions):
                return []  # 单个单元已覆盖
            dim = dimensions[dim_pos]
            start, stop = ranges[dim]
            if stop - start <= 1:
                return split(ranges, dim_pos + 1)
            mid = (start + stop) // 2
            left = dict(ranges)
            left[dim] = (start, mid)
            right = dict(ranges)
            right[dim] = (mid, stop)
            return split(left, dim_pos) + split(right, dim_pos)

        return split(dict(index_ranges), 0)

    # ---------- 状态汇总 ----------

    def _refresh_variable_status(self, variable: str) -> None:
        state = self.job["variables"][variable]
        versions = {r["version"] for r in state["slices"]}
        planned_pages = sum(len(p["pages"]) for p in state["plans"])
        empty_pages = len(state.get("empty_pages", []))
        saved_pages = len(state["slices"])
        if state["status"] == "retired":
            return
        if versions and len(versions) > 1:
            state["status"] = "mixed_version"
        elif saved_pages + empty_pages >= planned_pages:
            state["status"] = "complete"
        else:
            state["status"] = "in_progress"

    def _refresh_job_status(self) -> None:
        statuses = {v["status"] for v in self.job["variables"].values()}
        if statuses == {"complete"}:
            self.job["status"] = "complete"
        elif "retired" in statuses:
            self.job["status"] = "retired"
        elif "mixed_version" in statuses:
            self.job["status"] = "mixed_version"
        else:
            self.job["status"] = "in_progress"


def open_or_create_job(
    workdir: str | os.PathLike[str],
    client: SourceClient,
    selection_text: str | None = None,
) -> JobStore:
    store = JobStore(workdir)
    if store.job is None:
        if selection_text is None:
            raise ValueError("新作业必须提供坐标选择，例如 time=0:3,station=10:30")
        store.create(selection_text, client)
    return store
