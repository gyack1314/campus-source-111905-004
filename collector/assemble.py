"""把已落盘切片拼装为可用子集，并生成与实际覆盖、缺口、版次对应的完成清单。

单元被区分为三类：
* observed_usable：已取到且 quality=1；
* observed_unusable：已取到但 quality=0（资料存在、质控不可用，区别于缺测）；
* missing_fill：已取到但原始字节为 _FillValue（缺测，不参与缩放）。
未提供观测的坐标来自规划阶段记录的 gaps；未完成页面单独列为未取到单元。
"""
from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from .job import JobStore

# 单元状态编码（同时写入输出 NetCDF 的 cell_status 变量）
STATUS_USABLE = 1
STATUS_UNUSABLE = 0
STATUS_FILL = -1
STATUS_NOT_FETCHED = -2


def _load_slice(store: JobStore, record: dict) -> dict:
    data_path = store.root / SLICE_DIRNAME / record["data_file"]
    with np.load(data_path) as bundle:
        raw = np.array(bundle["raw"], dtype=np.int16)
        quality = np.array(bundle["quality"], dtype=np.int8)
    return {"record": record, "raw": raw, "quality": quality}


SLICE_DIRNAME = "slices"


def assemble_variable(store: JobStore, variable: str) -> dict:
    """拼装单个变量；混版或未完成时不生成合并阵列，只返回分版局部结果与清单。"""
    state = store.job["variables"][variable]
    dimensions = list(state["dimensions"])
    loaded = [_load_slice(store, record) for record in state["slices"]]
    versions = sorted({item["record"]["version"] for item in loaded})

    plan_axis_info = {}
    for plan in state["plans"]:
        for dim, info in plan["axis_selections"].items():
            plan_axis_info.setdefault(dim, info)

    # 防错：任何两个切片之间不得出现物理坐标重叠（重跑去重的最后一道防线）。
    seen: set[tuple] = set()
    for item in loaded:
        keys = store._cell_keys(item["record"])
        overlap = keys & seen
        if overlap:
            raise RuntimeError(
                f"变量 {variable} 切片在 {len(overlap)} 个坐标上重复，拒绝拼装"
            )
        seen |= keys

    counter: Counter[str] = Counter()
    version_cell_counts: Counter[str] = Counter()
    quality_changes: list[dict] = []
    outputs: list[Path] = []

    coherent_single_version = len(versions) == 1 and state["status"] != "retired"
    planned_pages = sum(len(p["pages"]) for p in state["plans"])
    empty_pages = len(state.get("empty_pages", []))
    fetched_pages = len(state["slices"])
    fully_fetched = fetched_pages + empty_pages >= planned_pages

    # 分版局部阵列始终可产出，保证退休中断后已落盘数据保持可解释。
    for version in versions:
        out_path = _write_partial(store, variable, version, dimensions,
                                  [i for i in loaded if i["record"]["version"] == version],
                                  counter, version_cell_counts)
        outputs.append(out_path)

    # 质量掩码在版次间可能变化：若同一物理坐标跨版重复出现（重建保留旧切片时），
    # 记录差异供清单核对。
    quality_changes = _diff_quality_across_versions(loaded, dimensions)

    gaps = {
        dim: _collect_gaps(state, dim)
        for dim in dimensions
    }
    unfetched_pages = [
        page["id"]
        for plan in state["plans"]
        for page in plan["pages"]
        if not store._slice_exists(variable, page["id"])
        and page["id"] not in state.get("empty_pages", [])
    ]

    merged_path: Path | None = None
    if coherent_single_version and fully_fetched and not unfetched_pages:
        merged_path = _write_merged(
            store, variable, dimensions, loaded, state, plan_axis_info
        )
        outputs.append(merged_path)

    # 跨版覆盖核对：目标网格取最终版计划命中的全部坐标，逐坐标确认至少被一版取到。
    coverage = _cross_version_coverage(store, variable, dimensions, loaded)

    checklist = {
        "variable": variable,
        "dimensions": dimensions,
        "status": state["status"],
        "catalog_versions_touched": versions,
        "planned_pages": planned_pages,
        "fetched_pages": fetched_pages,
        "empty_pages": empty_pages,
        "unfetched_pages": unfetched_pages,
        "cells_observed_usable": counter["usable"],
        "cells_observed_unusable_quality": counter["unusable"],
        "cells_missing_fill": counter["fill"],
        "cells_by_version": dict(version_cell_counts),
        "cross_version_coverage": coverage,
        "coordinates_not_observed": gaps,
        "quality_mask_changes": quality_changes,
        "outputs": [str(p.relative_to(store.root)) for p in outputs],
        "merged_complete_array": str(merged_path.relative_to(store.root))
        if merged_path is not None
        else None,
    }
    return checklist


def _cross_version_coverage(store: JobStore, variable: str,
                            dimensions: list[str], loaded: list[dict]) -> dict:
    """核对目标网格的每个坐标是否被某一版切片实际取到。"""
    state = store.job["variables"][variable]
    latest_plan = state["plans"][-1]
    target_coords = {
        dim: [float(v) for v in latest_plan["axis_selections"][dim]["selected"]]
        for dim in dimensions
    }
    grids = np.meshgrid(
        *[np.asarray(target_coords[d]) for d in dimensions], indexing="ij"
    )
    flat = [g.reshape(-1) for g in grids]
    target_cells = {
        tuple(float(x[i]) for x in flat) for i in range(flat[0].size)
    }
    covered = set()
    per_version: dict[str, set[tuple]] = {}
    for item in loaded:
        keys = store._cell_keys(item["record"])
        per_version.setdefault(item["record"]["version"], set()).update(keys)
        covered.update(keys)
    truly_uncovered = sorted(target_cells - covered)
    return {
        "target_cells": len(target_cells),
        "covered_cells": len(target_cells & covered),
        "truly_uncovered_coordinates": [
            dict(zip(dimensions, cell)) for cell in truly_uncovered
        ],
        "all_targets_covered_by_some_version": not truly_uncovered,
        "cells_per_version": {v: len(k) for v, k in sorted(per_version.items())},
    }


def _collect_gaps(state: dict, dim: str) -> list[list[float]]:
    gaps: list[list[float]] = []
    for plan in state["plans"]:
        info = plan["axis_selections"].get(dim)
        if info:
            for gap in info["gaps"]:
                if gap not in gaps:
                    gaps.append(list(gap))
    return gaps


def _classify(raw: np.ndarray, quality: np.ndarray, fill_value: int,
              counter: Counter[str]) -> np.ndarray:
    """三类互斥分类：fill 缺测优先判定；其余 quality=0 为质控不可用。

    契约中缺测单元的 quality 同样为 0，因此必须先排除 fill，
    不能把缺测单元重复计成“取到但质量不可用”。
    """
    fill_mask = raw == np.int16(fill_value)
    bad_mask = (~fill_mask) & (quality == 0)
    usable_mask = (~fill_mask) & (quality == 1)
    status = np.full(raw.shape, STATUS_USABLE, dtype=np.int8)
    status[bad_mask] = STATUS_UNUSABLE
    status[fill_mask] = STATUS_FILL
    counter["usable"] += int(np.count_nonzero(usable_mask))
    counter["unusable"] += int(np.count_nonzero(bad_mask))
    counter["fill"] += int(np.count_nonzero(fill_mask))
    return status


def _write_partial(store: JobStore, variable: str, version: str,
                   dimensions: list[str], items: list[dict],
                   counter: Counter, version_cell_counts: Counter) -> Path:
    """写某一版次实际取到单元的紧凑自洽阵列。

    坐标只来自该版次自己的切片，因此网格中每个单元都有来源，
    不会出现跨版次造成的 not_fetched 空洞；退休后这份文件单独打开即可解释。
    """
    arrays: list[dict] = []
    for item in items:
        record = item["record"]
        status = _classify(
            item["raw"], item["quality"], record["fill_value"], counter
        )
        values = _apply_scale(item["raw"], record)
        arrays.append({**item, "status": status, "values": values})
        version_cell_counts[version] += item["raw"].size

    out_dir = store.root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{variable}__{version}__partial.nc"
    _write_netcdf(out_path, variable, dimensions, arrays,
                  source_versions=[version], complete=False)
    return out_path


def _write_merged(store: JobStore, variable: str, dimensions: list[str],
                  items: list[dict], state: dict, plan_axis_info: dict) -> Path:
    counter: Counter = Counter()
    arrays = []
    for item in items:
        record = item["record"]
        status = _classify(item["raw"], item["quality"], record["fill_value"], counter)
        values = _apply_scale(item["raw"], record)
        arrays.append({**item, "status": status, "values": values})

    # 合并阵列使用该变量最终规划版目录的完整命中坐标网格，
    # 并校验所有切片坐标都落在该网格上。
    out_dir = store.root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{variable}__complete.nc"
    _write_netcdf(out_path, variable, dimensions, arrays,
                  source_versions=sorted({a["record"]["version"] for a in arrays}),
                  complete=True)
    return out_path


def _apply_scale(raw: np.ndarray, record: dict) -> np.ndarray:
    scaled = raw.astype(np.float64) * record["scale_factor"] + record["add_offset"]
    scaled[raw == np.int16(record["fill_value"])] = np.nan
    return scaled


def _write_netcdf(path: Path, variable: str, dimensions: list[str],
                  arrays: list[dict], source_versions: list[str], complete: bool) -> None:
    # 合并坐标：沿各维收集所有切片坐标，按目录轴方向去重排序。
    axes_order: dict[str, list[float]] = {}
    for dim in dimensions:
        coords: list[float] = []
        for item in arrays:
            for value in item["record"]["coordinates"][dim]:
                if value not in coords:
                    coords.append(value)
        descending = bool(len(coords) > 1 and coords[0] > coords[-1])
        coords.sort(reverse=descending)
        axes_order[dim] = coords

    shape = tuple(len(axes_order[d]) for d in dimensions)
    values = np.full(shape, np.nan, dtype=np.float64)
    status = np.full(shape, STATUS_NOT_FETCHED, dtype=np.int8)
    provenance = np.full(shape, -1, dtype=np.int8)
    version_codes = {v: i for i, v in enumerate(source_versions)}

    for item in arrays:
        record = item["record"]
        pos = [
            np.array([axes_order[d].index(v) for v in record["coordinates"][d]])
            for d in dimensions
        ]
        idx = tuple(np.ix_(*pos)) if len(pos) > 1 else (pos[0],)
        if np.any(~np.isnan(values[idx])):
            raise RuntimeError(f"写入 {path.name} 时发现坐标重复覆盖")
        values[idx] = item["values"]
        status[idx] = item["status"]
        provenance[idx] = version_codes[record["version"]]

    buffer = io.BytesIO()
    nc = netcdf_file(buffer, "w")
    nc.source_versions = ",".join(source_versions)
    nc.collection_complete = "true" if complete else "false"
    nc.coverage_note = (
        "cell_status=-2 marks coordinates this file's version did not fetch; "
        "they may be held by another version's partial file. A coordinate is only "
        "truly uncovered when every version marks it not_fetched."
    )
    for dim in dimensions:
        nc.createDimension(dim, len(axes_order[dim]))
        coord_var = nc.createVariable(dim, "f8", (dim,))
        coord_var[:] = np.array(axes_order[dim], dtype=np.float64)
    data_var = nc.createVariable(variable, "f8", tuple(dimensions))
    data_var.scale_convention = "float NaN marks fill; raw int16 retained in slice archive"
    data_var[:] = values
    status_var = nc.createVariable("cell_status", "i1", tuple(dimensions))
    status_var.flag_meanings = (
        "1=observed_usable 0=observed_quality_unusable "
        "-1=missing_fill -2=not_fetched"
    )
    status_var[:] = status
    prov_var = nc.createVariable("source_version_index", "i1", tuple(dimensions))
    prov_var.version_lookup = json.dumps(version_codes, ensure_ascii=False)
    prov_var[:] = provenance
    nc.flush()
    payload = buffer.getvalue()  # scipy 关闭时会连带关闭 buffer，须在关闭前取字节
    nc.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _diff_quality_across_versions(loaded: list[dict], dimensions: list[str]) -> list[dict]:
    """同一物理坐标若在多个版次切片中出现，报告 quality 差异（掩码变化演练用）。"""
    latest: dict[tuple, dict] = {}
    changes: list[dict] = []
    for item in loaded:
        record = item["record"]
        grids = np.meshgrid(
            *[np.asarray(record["coordinates"][d]) for d in dimensions],
            indexing="ij",
        )
        flat = [g.reshape(-1) for g in grids]
        quality_flat = item["quality"].reshape(-1)
        for i in range(quality_flat.size):
            key = tuple(float(x[i]) for x in flat)
            current = {"version": record["version"], "quality": int(quality_flat[i])}
            if key in latest and latest[key]["quality"] != current["quality"]:
                changes.append(
                    {
                        "coordinates": dict(zip(dimensions, key)),
                        "from": latest[key],
                        "to": current,
                    }
                )
            latest[key] = current
    return changes


def assemble_job(store: JobStore) -> dict:
    """拼装全部变量并做跨变量网格检查，返回完整完成清单。"""
    checklists = []
    grid_signatures = []
    for variable in store.job["variables"]:
        checklist = assemble_variable(store, variable)
        checklists.append(checklist)
        state = store.job["variables"][variable]
        grid_signatures.append(
            (variable, tuple(state["dimensions"]),
             tuple(_merged_shape(store, variable)))
        )

    # 形状相同但维度不同（temperature(time,station) 与 current(station,sample)）
    # 必须显式禁止按形状对齐：这里不静默广播，而是写入告警并保留各自独立网格。
    alignment_warnings: list[str] = []
    for i, (name_a, dims_a, shape_a) in enumerate(grid_signatures):
        for name_b, dims_b, shape_b in grid_signatures[i + 1:]:
            if shape_a == shape_b and dims_a != dims_b:
                alignment_warnings.append(
                    f"{name_a}{list(dims_a)} 与 {name_b}{list(dims_b)} 形状同为 "
                    f"{shape_a}，采样轴不同，未做任何按位置对齐；如需联合分析请按坐标显式连接"
                )

    summary = {
        "job_status": store.job["status"],
        "selection": store.job["selection"],
        "variables": checklists,
        "alignment_warnings": alignment_warnings,
        "alignment_notice": (
            "各变量保留自身维度与坐标；temperature 为 (time, station)、"
            "current 为 (station, sample)，形状相同也不按位置对齐"
        ),
    }
    out_path = store.root / "checklist.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _merged_shape(store: JobStore, variable: str) -> list[int]:
    state = store.job["variables"][variable]
    counts: dict[str, set[float]] = {d: set() for d in state["dimensions"]}
    for record in state["slices"]:
        for dim, coords in record["coordinates"].items():
            counts[dim].update(coords)
    return [len(counts[d]) for d in state["dimensions"]]
