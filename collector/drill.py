"""可重复的来源演练：轴中间补点、质量掩码变化、旧版退休。

默认在本进程内启动来源服务（source.server.running_source），通过契约中的
/__control 演练接口推进 stage；控制接口只用于推进演练状态，采集结果全部来自
/catalog 与 /array.nc 正式端点。数据写入调用方指定的独立目录，与代码目录分离。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .assemble import assemble_job
from .job import JobStore
from .protocol import SourceClient, VersionRetired

# 同时覆盖两个变量所需的全部物理坐标轴。
DEFAULT_SELECTION = "time=0:5,station=10:30,sample=0:4.6"


def _fresh_workdir(root: str | Path, name: str) -> Path:
    path = Path(root) / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _pending_pages(store: JobStore, variable: str) -> list[dict]:
    state = store.job["variables"][variable]
    pages = []
    for plan in state["plans"]:
        for page in plan["pages"]:
            if not store._slice_exists(variable, page["id"]):
                pages.append((plan, page))
    return pages


def run_all_drills(data_root: str | Path) -> dict:
    from source.server import running_source

    reports: dict[str, object] = {}
    with running_source() as (base, source):
        client = SourceClient(base)

        reports["stage0_r1_baseline"] = drill_baseline(
            client, source, _fresh_workdir(data_root, "drill_baseline")
        )
        reports["mid_download_retirement"] = drill_retirement(
            client, source, _fresh_workdir(data_root, "drill_retirement")
        )
        reports["axis_insertion_and_quality_mask"] = drill_insertion_and_mask(
            client, source, _fresh_workdir(data_root, "drill_insertion")
        )
    return reports


def drill_baseline(client: SourceClient, source, workdir: Path) -> dict:
    """阶段 0（r1）完整采集，验证分页、空选择、缺测与质量分类、重跑去重。"""
    source.stage = 0  # 复位，保证各演练可独立、可重复运行
    store = JobStore(workdir)
    store.create(DEFAULT_SELECTION, client)
    store.collect(client)
    first_run = {v: len(s["slices"]) for v, s in store.job["variables"].items()}

    # 重跑：不得新增任何切片或边界单元。
    store.collect(client)
    second_run = {v: len(s["slices"]) for v, s in store.job["variables"].items()}
    assert first_run == second_run, "重跑重复拼入了切片"

    summary = assemble_job(store)
    assert summary["job_status"] == "complete"

    # 契约中的已知缺测单元：temperature 目录索引 (time=1, station=1)。
    temp = next(v for v in summary["variables"] if v["variable"] == "temperature")
    assert temp["cells_missing_fill"] >= 1
    assert temp["cells_observed_usable"] + temp["cells_observed_unusable_quality"] + \
        temp["cells_missing_fill"] == _planned_cells(temp)
    return {
        "workdir": str(workdir),
        "pages_per_variable": first_run,
        "rerun_pages_per_variable": second_run,
        "job_status": summary["job_status"],
        "checklist": summary,
    }


def drill_retirement(client: SourceClient, source, workdir: Path) -> dict:
    """下载到一半时 r1 退休：已落盘数据可解释，重建后不产生混版“完整”阵列。"""
    source.stage = 0  # 从 r1 开始，保证可独立重复运行
    store = JobStore(workdir)
    store.create(DEFAULT_SELECTION, client)

    # 只取每个变量的第一页，随后推进到阶段 2（r1 退休）。
    partial_pages: dict[str, str] = {}
    for variable in store.job["variables"]:
        plan, page = _pending_pages(store, variable)[0]
        slc = client.get_slice(
            plan["version"], variable,
            {d: tuple(r) for d, r in page["index_ranges"].items()},
            store.catalog_snapshot(plan["version"]),
        )
        store._save_slice(variable, page, slc)
        partial_pages[variable] = page["id"]
    store._refresh_job_status()
    store._flush()

    source.stage = 2  # 演练控制：r1 租期结束
    retired_error = None
    try:
        store.collect(client)
    except VersionRetired as exc:
        retired_error = str(exc)

    # 中断状态下仍可拼装：只有 r1 分版局部阵列，作业不得被标为 complete。
    interrupted = assemble_job(store)
    assert interrupted["job_status"] == "retired"
    for variable in interrupted["variables"]:
        assert variable["merged_complete_array"] is None
        assert variable["outputs"], "退休中断后已落盘切片仍应可解释"

    # 旧版索引不得按新版坐标解释：直接拿旧索引请求会被服务拒绝（410）。
    old_index_denied = False
    try:
        client.get_catalog(version="r1")
    except VersionRetired:
        old_index_denied = True

    # 重建：当前目录为 r2，按相同物理坐标补齐未覆盖单元。
    rebuilt = store.rebuild_on_retirement(client)
    summary = assemble_job(store)

    versions = {
        v["variable"]: v["catalog_versions_touched"] for v in summary["variables"]
    }
    assert all("r1" in touched and "r2" in touched for touched in versions.values())
    assert summary["job_status"] == "mixed_version"
    for variable in summary["variables"]:
        assert variable["merged_complete_array"] is None, "混版结果不得标为完整阵列"
        cov = variable["cross_version_coverage"]
        assert cov["all_targets_covered_by_some_version"], (
            f"{variable['variable']} 存在跨版仍未覆盖的坐标："
            f"{cov['truly_uncovered_coordinates']}"
        )

    # 再重跑一次：重建后的 r2 页面也不允许重复拼入。
    before = {v: len(s["slices"]) for v, s in store.job["variables"].items()}
    store.collect(client)
    after = {v: len(s["slices"]) for v, s in store.job["variables"].items()}
    assert before == after

    return {
        "workdir": str(workdir),
        "partial_pages_before_retirement": partial_pages,
        "retired_error": retired_error,
        "old_catalog_read_denied": old_index_denied,
        "versions_per_variable": versions,
        "job_status": summary["job_status"],
        "interrupted_outputs_explainable": True,
        "checklist": summary,
    }


def drill_insertion_and_mask(client: SourceClient, source, workdir: Path) -> dict:
    """r2 在 time 轴中间补入 2.5 秒且更新质量掩码；坐标翻译必须识别新点。"""
    source.stage = 0  # 从 r1 开始，保证可独立重复运行
    r1_catalog = client.get_catalog()
    assert 2.5 not in r1_catalog["axes"]["time"]

    store = JobStore(workdir)
    store.create("time=0:5,station=10:30,sample=0:4.6", client)
    store.collect(client)
    r1_slices = sum(len(s["slices"]) for s in store.job["variables"].values())

    source.stage = 1  # 演练控制：发布 r2（r1 仍在租期内）
    r2_catalog = client.get_catalog()
    assert r2_catalog["version"] == "r2"
    assert r2_catalog["axes"]["time"][3] == 2.5  # 中间补点，不是尾部增长

    # 新作业直接面向 r2：物理坐标选择应同时覆盖旧点与中间补入点。
    r2_workdir = workdir.parent / "drill_insertion_r2"
    store2 = JobStore(r2_workdir)
    store2.create("time=0:5,station=10:30,sample=0:4.6", client)
    store2.collect(client)
    summary2 = assemble_job(store2)
    temp_plan = store2.job["variables"]["temperature"]["plans"][-1]
    time_selected = temp_plan["axis_selections"]["time"]["selected"]
    assert 2.5 in time_selected and time_selected.index(2.5) == 3

    # 质量掩码变化：契约规定 r2 在目录索引 (time=2, station=0) 新增 quality=0。
    temp_checklist = next(
        v for v in summary2["variables"] if v["variable"] == "temperature"
    )
    r2_bad = temp_checklist["cells_observed_unusable_quality"]

    # 对照 r1 同坐标范围的 quality=0 数量（仅 (1,1) 一处）。
    r1_bad = _count_quality_zero(store, "temperature")
    assert r2_bad > r1_bad, "r2 质量掩码变化未被采集结果反映"

    # 降序 station 轴与非等间隔 sample 轴的坐标必须按目录原样保留。
    station_axis = r2_catalog["axes"]["station"]
    assert station_axis == sorted(station_axis, reverse=True)
    sample_diffs = np.diff(r2_catalog["axes"]["sample"])
    assert not np.allclose(sample_diffs, sample_diffs[0])

    return {
        "workdir": str(r2_workdir),
        "r1_time_axis": r1_catalog["axes"]["time"],
        "r2_time_axis": r2_catalog["axes"]["time"],
        "inserted_coordinate": 2.5,
        "r1_slices": r1_slices,
        "r1_quality_zero_cells": r1_bad,
        "r2_quality_zero_cells": r2_bad,
        "station_descending": station_axis,
        "sample_nonuniform": r2_catalog["axes"]["sample"],
        "checklist": summary2,
    }


def _count_quality_zero(store: JobStore, variable: str) -> int:
    """统计“有值但 quality=0”的单元；fill 哨兵单元不计入（其 quality 也为 0）。"""
    total = 0
    for record in store.job["variables"][variable]["slices"]:
        path = store.root / "slices" / record["data_file"]
        fill = np.int16(record["fill_value"])
        with np.load(path) as bundle:
            raw, quality = bundle["raw"], bundle["quality"]
            total += int(np.count_nonzero((raw != fill) & (quality == 0)))
    return total


def _planned_cells(checklist_variable: dict) -> int:
    return (
        checklist_variable["cells_observed_usable"]
        + checklist_variable["cells_observed_unusable_quality"]
        + checklist_variable["cells_missing_fill"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行来源演练（轴补点/掩码变化/版次退休）")
    parser.add_argument("--data-root", default="./drill-data", help="演练数据根目录（独立于代码）")
    args = parser.parse_args(argv)
    reports = run_all_drills(args.data_root)
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
