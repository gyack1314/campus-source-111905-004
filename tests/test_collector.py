"""采集客户端测试：覆盖契约中的分页、坐标、质量与版次语义。"""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

from collector.assemble import assemble_job
from collector.coords import (
    guard_distinct_grids,
    parse_selection,
    plan_pages,
    select_indices,
)
from collector.job import JobStore
from collector.protocol import (
    SourceClient,
    TooLarge,
    VersionMismatch,
    VersionRetired,
)
from tests.support import SourceCase

SELECT = "time=0:5,station=10:30,sample=0:4.6"


class CoordinateTests(unittest.TestCase):
    def test_descending_axis_returns_catalog_order_indices(self):
        sel = select_indices("station", [30, 20, 10], 10, 30)
        self.assertEqual(sel.indices, (0, 3))
        self.assertEqual(sel.selected, [30, 20, 10])

    def test_descending_partial_range(self):
        sel = select_indices("station", [30, 20, 10], 10, 20)
        self.assertEqual(sel.indices, (1, 3))
        self.assertEqual(sel.selected, [20, 10])

    def test_nonuniform_axis_is_flagged_not_treated_as_gap(self):
        from collector.coords import describe_axis

        sel = select_indices("sample", [0, 0.4, 1.8, 4.6], 0, 4.6)
        # 全部请求坐标都有采样点：没有“未提供观测”的缺口。
        self.assertEqual(sel.gaps, [])
        self.assertEqual(sel.indices, (0, 4))
        description = describe_axis([0, 0.4, 1.8, 4.6])
        self.assertFalse(description["uniform_spacing"])

    def test_range_outside_coverage_is_gap_and_empty(self):
        sel = select_indices("time", [0, 1, 2], 8, 9)
        self.assertTrue(sel.empty)
        self.assertEqual(sel.gaps, [(8, 9)])

    def test_partial_outside_coverage(self):
        sel = select_indices("time", [0, 1, 2], 1.5, 9)
        self.assertEqual(sel.indices, (2, 3))
        self.assertEqual(sel.selected, [2])
        # 缺口端点沿用请求边界：[1.5,2) 与 (2,9] 均无观测。
        self.assertEqual(sel.gaps, [(1.5, 2.0), (2.0, 9)])

    def test_plan_pages_respects_max_cells(self):
        pages = plan_pages(["a", "b"], {"a": (0, 3), "b": (0, 3)}, 4)
        for page in pages:
            cells = np.prod([e - s for s, e in page.values()])
            self.assertLessEqual(cells, 4)
        # 确定性：同样输入得到同样页面序列。
        self.assertEqual(pages, plan_pages(["a", "b"], {"a": (0, 3), "b": (0, 3)}, 4))

    def test_plan_pages_empty_selection(self):
        self.assertEqual(plan_pages(["a"], {"a": (2, 2)}, 4), [])

    def test_parse_selection(self):
        self.assertEqual(
            parse_selection("time=0:5, station=10:30"),
            {"time": (0.0, 5.0), "station": (10.0, 30.0)},
        )

    def test_shape_guard_rejects_cross_grid_alignment(self):
        with self.assertRaises(ValueError):
            guard_distinct_grids(
                "temperature", ("time", "station"), (3, 3),
                "current", ("station", "sample"), (3, 3),
            )


class PaginationTests(SourceCase):
    def test_request_over_limit_is_413_and_planner_avoids_it(self):
        client = SourceClient(self.base)
        catalog = client.get_catalog()
        # 全量 temperature 为 6x3=18 > max_cells=4，直接请求会被拒。
        with self.assertRaises(TooLarge):
            client.get_slice(
                catalog["version"], "temperature",
                {"time": (0, 6), "station": (0, 3)}, catalog,
            )
        pages = plan_pages(
            catalog["variables"]["temperature"]["dimensions"],
            {"time": (0, 6), "station": (0, 3)}, catalog["max_cells"],
        )
        self.assertTrue(pages)
        for page in pages:
            slc = client.get_slice(catalog["version"], "temperature", page, catalog)
            self.assertIsNotNone(slc)
            self.assertLessEqual(slc.raw.size, catalog["max_cells"])


class CollectionTests(SourceCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name) / "job"
        self.client = SourceClient(self.base)

    def test_full_collection_checklist_and_netcdf(self):
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        store.collect(self.client)
        summary = assemble_job(store)
        self.assertEqual(summary["job_status"], "complete")

        temp = next(v for v in summary["variables"] if v["variable"] == "temperature")
        curr = next(v for v in summary["variables"] if v["variable"] == "current")
        # temperature 选择 time 6 点 x station 3 点；current 为 station 3 x sample 4。
        self.assertEqual(
            temp["cells_observed_usable"]
            + temp["cells_observed_unusable_quality"]
            + temp["cells_missing_fill"],
            18,
        )
        self.assertEqual(
            curr["cells_observed_usable"]
            + curr["cells_observed_unusable_quality"]
            + curr["cells_missing_fill"],
            12,
        )
        # 契约规定 (time=1, station=1) 是 fill 且 quality=0：按互斥分类归为缺测，
        # r1 不存在“有值但质控不可用”的单元。
        self.assertEqual(temp["cells_missing_fill"], 1)
        self.assertEqual(temp["cells_observed_unusable_quality"], 0)
        self.assertTrue(Path(self.workdir, temp["merged_complete_array"]).exists())

        with netcdf_file(
            io.BytesIO(Path(self.workdir, temp["merged_complete_array"]).read_bytes()),
            mmap=False,
        ) as nc:
            self.assertEqual(nc.collection_complete, b"true")
            self.assertEqual(nc.variables["station"][:].tolist(), [30, 20, 10])
            status = np.array(nc.variables["cell_status"][:], dtype=np.int8)
            self.assertIn(-1, [int(x) for x in status.reshape(-1)])  # 缺测单元状态保留
            self.assertTrue(np.isnan(nc.variables["temperature"][:][status == -1]).all())
            # 非缺测单元按缩放/偏移解释，且不保留 fill 哨兵的缩放值。
            self.assertNotIn(-9999, nc.variables["temperature"][:].tolist())

    def test_fill_value_is_not_scaled(self):
        store = JobStore(self.workdir)
        store.create("time=1:1.5,station=20:20,sample=0:0", self.client)
        store.collect(self.client)
        record = store.job["variables"]["temperature"]["slices"][0]
        data = np.load(Path(self.workdir, "slices", record["data_file"]))
        self.assertEqual(int(data["raw"].item()), -9999)  # 原始字节保留哨兵
        slc = self.client.get_slice(
            record["version"], "temperature",
            {d: tuple(r) for d, r in record["index_ranges"].items()},
            store.catalog_snapshot(record["version"]),
        )
        values = slc.values_masked()
        self.assertTrue(np.ma.is_masked(values))
        self.assertTrue(values.mask.all())

    def test_empty_coordinate_range_is_legal(self):
        store = JobStore(self.workdir)
        # 没有任何采样点落在 [8, 9]。
        store.create("time=8:9,station=10:30,sample=8:9", self.client)
        store.collect(self.client)
        summary = assemble_job(store)
        self.assertEqual(summary["job_status"], "complete")
        for variable in summary["variables"]:
            self.assertEqual(variable["planned_pages"], 0)
            self.assertEqual(variable["fetched_pages"], 0)
            self.assertTrue(variable["coordinates_not_observed"])
            self.assertIsNone(variable["merged_complete_array"])

    def test_empty_page_rerun_is_idempotent(self):
        # 空坐标选择产生零页面作业；重跑必须得到字节级一致的作业状态，
        # 不重复请求、不重复登记。
        store = JobStore(Path(self.tmp.name) / "job_empty_page")
        store.create("time=8:9,station=10:30,sample=8:9", self.client)
        store.collect(self.client)
        requests_before = len(self.source.log)
        before = json.dumps(store.job, sort_keys=True)
        store.collect(self.client)
        self.assertEqual(json.dumps(store.job, sort_keys=True), before)
        self.assertEqual(len(self.source.log), requests_before)

    def test_rerun_does_not_duplicate_boundary_cells(self):
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        store.collect(self.client)
        counts_first = {
            v: len(s["slices"]) for v, s in store.job["variables"].items()
        }
        requests_before = len(self.source.log)
        store.collect(self.client)
        assemble_job(store)  # 内含重叠检测，重复坐标会直接抛错
        counts_second = {
            v: len(s["slices"]) for v, s in store.job["variables"].items()
        }
        self.assertEqual(counts_first, counts_second)
        self.assertEqual(len(self.source.log), requests_before)

    def test_all_missing_segment_remains_explainable(self):
        # (time=1, station=1) 是全缺测片段：合法取到但无有效观测。
        store = JobStore(self.workdir)
        store.create("time=1:1.5,station=20:20,sample=0:0", self.client)
        store.collect(self.client)
        summary = assemble_job(store)
        temp = next(v for v in summary["variables"] if v["variable"] == "temperature")
        self.assertEqual(temp["cells_missing_fill"], 1)
        self.assertEqual(temp["cells_observed_usable"], 0)


class VersionTests(SourceCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = Path(self.tmp.name) / "job"
        self.client = SourceClient(self.base)

    def test_mid_axis_insertion_uses_physical_coordinates(self):
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        store.collect(self.client)
        self.control(1)  # r2 发布，time 轴中间补 2.5
        r2 = self.client.get_catalog()
        self.assertEqual(r2["axes"]["time"][3], 2.5)

        # 旧版索引不得按新版坐标解释：r1 索引 time=3:4 的物理坐标是 3，
        # 而 r2 同索引位置已是中间补入的 2.5。用 r1 目录快照解释 r2 响应时，
        # 客户端的坐标核对必须拦截这种版次错配。
        shifted = {"time": (3, 4), "station": (0, 1)}
        r1_snapshot = store.catalog_snapshot("r1")
        with self.assertRaises(VersionMismatch):
            self.client.get_slice("r2", "temperature", shifted, r1_snapshot)
        # 同一索引区间在 r1 目录下合法且坐标确为 3，证明拦截来自坐标漂移。
        ok = self.client.get_slice("r1", "temperature", shifted, r1_snapshot)
        self.assertEqual([float(v) for v in ok.coordinates["time"]], [3.0])
        # 用 r2 自己的目录解释则正常，但坐标是 2.5 而非 3。
        r2_slc = self.client.get_slice("r2", "temperature", shifted, r2)
        self.assertEqual([float(v) for v in r2_slc.coordinates["time"]], [2.5])

    def test_retirement_mid_download_keeps_partial_explainable(self):
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        # 落一个切片后退休。
        variable = "temperature"
        plan, page = store.job["variables"][variable]["plans"][0], None
        page = store.job["variables"][variable]["plans"][0]["pages"][0]
        slc = self.client.get_slice(
            plan["version"], variable,
            {d: tuple(r) for d, r in page["index_ranges"].items()},
            store.catalog_snapshot(plan["version"]),
        )
        store._save_slice(variable, page, slc)
        self.control(2)  # r1 退休
        with self.assertRaises(VersionRetired):
            store.collect(self.client)
        summary = assemble_job(store)
        self.assertEqual(summary["job_status"], "retired")
        temp = next(v for v in summary["variables"] if v["variable"] == variable)
        self.assertIsNone(temp["merged_complete_array"])
        self.assertTrue(temp["outputs"])  # 已落盘部分仍可解释

    def test_rebuild_after_retirement_covers_inserted_coordinate(self):
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        # 先完成 temperature 的第一页，再退休。
        state = store.job["variables"]["temperature"]
        plan = state["plans"][0]
        page = plan["pages"][0]
        slc = self.client.get_slice(
            plan["version"], "temperature",
            {d: tuple(r) for d, r in page["index_ranges"].items()},
            store.catalog_snapshot(plan["version"]),
        )
        store._save_slice("temperature", page, slc)
        self.control(2)
        with self.assertRaises(VersionRetired):
            store.collect(self.client)

        summary = store.rebuild_on_retirement(self.client)
        self.assertEqual(summary["status"], "mixed_version")
        final = assemble_job(store)
        temp = next(v for v in final["variables"] if v["variable"] == "temperature")
        self.assertEqual(sorted(temp["catalog_versions_touched"]), ["r1", "r2"])
        self.assertIsNone(temp["merged_complete_array"])  # 混版不得标完整
        # r2 轴中间的 2.5 秒点必须被新计划覆盖。
        r2_plan = state["plans"][-1]
        self.assertIn(2.5, r2_plan["axis_selections"]["time"]["selected"])
        # 单元总数按 r2 网格核对（7 个 time x 3 个 station）。
        classified = (
            temp["cells_observed_usable"]
            + temp["cells_observed_unusable_quality"]
            + temp["cells_missing_fill"]
        )
        self.assertEqual(classified, 21)
        # 跨版覆盖：21 个目标坐标每个都被 r1 或 r2 实际取到，无真实遗漏。
        cov = temp["cross_version_coverage"]
        self.assertTrue(cov["all_targets_covered_by_some_version"])
        self.assertEqual(cov["target_cells"], 21)
        self.assertEqual(cov["covered_cells"], 21)
        self.assertEqual(cov["truly_uncovered_coordinates"], [])
        # 重跑重建作业不会重复拼入。
        before = len(state["slices"])
        store.collect(self.client)
        self.assertEqual(len(state["slices"]), before)

    def test_quality_mask_change_visible_in_checklist(self):
        self.control(1)  # r2：(time=2, station=0) 新增 quality=0
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        store.collect(self.client)
        summary = assemble_job(store)
        temp = next(v for v in summary["variables"] if v["variable"] == "temperature")
        # r1 无“有值但 quality=0”单元；r2 在 (time=2, station=0) 新增一处。
        self.assertEqual(temp["cells_observed_unusable_quality"], 1)
        self.assertEqual(temp["cells_missing_fill"], 1)

    def test_catalog_snapshot_is_immutable(self):
        store = JobStore(self.workdir)
        store.create(SELECT, self.client)
        snapshot_path = self.workdir / "catalogs" / "r1.json"
        before = snapshot_path.read_bytes()
        store.collect(self.client)
        self.assertEqual(snapshot_path.read_bytes(), before)
        self.control(1)
        r2 = self.client.get_catalog()
        store._save_catalog("r2", r2)
        self.assertTrue((self.workdir / "catalogs" / "r2.json").exists())


class AlignmentGuardTests(SourceCase):
    def test_same_shape_distinct_axes_warns_not_aligned(self):
        # 构造两个同形不同轴的变量：temperature 取 3 个 time x 2 个 station，
        # current 取 2 个 station x 3 个 sample，形状同为 (2,3) 但语义轴不同。
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = JobStore(Path(tmp.name) / "job")
        store.create("time=0:2,station=10:30,sample=0:1.8", SourceClient(self.base))
        store.collect(SourceClient(self.base))
        summary = assemble_job(store)
        shapes = {
            v["variable"]: tuple(
                np.array(
                    netcdf_file(
                        io.BytesIO(
                            (Path(store.root) / v["merged_complete_array"]).read_bytes()
                        ),
                        mmap=False,
                    ).variables[v["variable"]].shape
                )
            )
            for v in summary["variables"]
        }
        self.assertEqual(shapes["temperature"], shapes["current"])
        self.assertTrue(
            any("temperature" in w and "current" in w for w in summary["alignment_warnings"]),
            summary["alignment_warnings"],
        )


class DrillIntegrationTests(SourceCase):
    def test_run_all_drills_covers_three_scenarios(self):
        from collector.drill import (
            drill_baseline,
            drill_insertion_and_mask,
            drill_retirement,
        )

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        client = SourceClient(self.base)

        baseline = drill_baseline(client, self.source, root / "base")
        self.assertEqual(baseline["job_status"], "complete")
        self.assertEqual(
            baseline["pages_per_variable"], baseline["rerun_pages_per_variable"]
        )

        retirement = drill_retirement(client, self.source, root / "retire")
        self.assertTrue(retirement["old_catalog_read_denied"])
        self.assertEqual(retirement["job_status"], "mixed_version")
        self.assertTrue(retirement["interrupted_outputs_explainable"])

        insertion = drill_insertion_and_mask(client, self.source, root / "insert")
        self.assertEqual(insertion["inserted_coordinate"], 2.5)
        self.assertGreater(
            insertion["r2_quality_zero_cells"], insertion["r1_quality_zero_cells"]
        )


if __name__ == "__main__":
    unittest.main()
