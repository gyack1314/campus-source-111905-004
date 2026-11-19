"""来源服务 HTTP 协议客户端与 NetCDF 切片解码。

严格按 docs/来源契约.md 实现：目录读取、按维度索引切片、状态码约定
（204 合法空选择、400 越界、410 版次退休、413 单次超限），
并在本地手动解释 scale_factor/add_offset/_FillValue，缺测原始字节不经过缩放。
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import requests
from scipy.io import netcdf_file


class SourceError(RuntimeError):
    """来源服务返回了无法继续处理的响应。"""


class BadRequest(SourceError):
    """400：请求不符合来源契约（如索引越界）。"""


class VersionRetired(SourceError):
    """410：该发布版租期已结束，旧版索引不得再按新版坐标解释。"""


class TooLarge(SourceError):
    """413：单次单元数超过 max_cells，服务不代为分页。"""


class VersionMismatch(SourceError):
    """响应版次与请求版次或响应头不一致，拒绝接受该切片。"""


@dataclass(frozen=True)
class Slice:
    """一次成功切片的完整本地来源记录。"""

    variable: str
    version: str
    dimensions: tuple[str, ...]
    coordinates: dict[str, list[float]]
    raw: np.ndarray  # int16 原始字节，含 _FillValue 哨兵，未做缩放
    quality: np.ndarray  # int8，0=存在但质控不可用，1=质控可用
    scale_factor: float
    add_offset: float
    fill_value: int
    index_ranges: dict[str, tuple[int, int]]  # 该版目录轴上的索引区间

    @property
    def cell_coordinates(self):
        """逐单元的物理坐标元组迭代器，按数组行优先顺序展开。"""
        grids = np.meshgrid(
            *[np.asarray(self.coordinates[d]) for d in self.dimensions],
            indexing="ij",
        )
        flat = [g.reshape(-1) for g in grids]
        for i in range(self.raw.size):
            yield tuple(float(x[i]) for x in flat)

    def values_masked(self) -> np.ma.MaskedArray:
        """按契约手动解释缩放与缺测：fill 哨兵保持缺测，绝不先乘缩放。"""
        raw = self.raw.astype(np.float64, copy=True)
        mask = self.raw == np.int16(self.fill_value)
        scaled = raw * self.scale_factor + self.add_offset
        return np.ma.MaskedArray(scaled, mask=mask)


class SourceClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        response = requests.get(self.base_url + "/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_catalog(self, version: str | None = None) -> dict:
        params = {} if version is None else {"version": version}
        response = requests.get(
            self.base_url + "/catalog", params=params, timeout=self.timeout
        )
        if response.status_code == 410:
            raise VersionRetired(
                f"目录版次 {version} 已退休：{response.text.strip() or '租期结束'}"
            )
        response.raise_for_status()
        catalog = response.json()
        if version is not None and catalog.get("version") != version:
            raise VersionMismatch(
                f"请求版次 {version} 但目录返回 {catalog.get('version')}"
            )
        return catalog

    def get_slice(
        self,
        version: str,
        variable: str,
        index_ranges: dict[str, tuple[int, int]],
        catalog: dict,
    ) -> Slice | None:
        """按该版目录的索引区间取一个切片；合法空选择返回 None。"""
        meta = catalog["variables"][variable]
        dimensions = list(meta["dimensions"])

        params: list[tuple[str, str]] = [("version", version), ("variable", variable)]
        for dim in dimensions:
            start, stop = index_ranges[dim]
            params.append((dim, f"{start}:{stop}"))

        response = requests.get(
            self.base_url + "/array.nc", params=params, timeout=self.timeout
        )
        if response.status_code == 204:
            header_version = response.headers.get("X-Source-Version")
            if header_version != version:
                raise VersionMismatch(
                    f"空切片响应版次 {header_version} 与请求 {version} 不一致"
                )
            return None
        if response.status_code == 400:
            raise BadRequest(response.text)
        if response.status_code == 410:
            raise VersionRetired(
                f"版次 {version} 在租期结束后被读取：{response.text.strip()}"
            )
        if response.status_code == 413:
            raise TooLarge(response.text)
        response.raise_for_status()

        header_version = response.headers.get("X-Source-Version")
        if header_version != version:
            raise VersionMismatch(
                f"切片响应版次 {header_version} 与请求 {version} 不一致"
            )

        return self._decode_netcdf(
            response.content, version, variable, dimensions, index_ranges, catalog
        )

    @staticmethod
    def _decode_netcdf(
        content: bytes,
        version: str,
        variable: str,
        dimensions: list[str],
        index_ranges: dict[str, tuple[int, int]],
        catalog: dict,
    ) -> Slice:
        # maskandscale=False：保留原始字节与属性，由本工具自行解释，
        # 避免缺测哨兵被自动缩放后误当有效观测。
        with netcdf_file(io.BytesIO(content), mmap=False, maskandscale=False) as nc:
            if getattr(nc, "source_version", b"").decode() != version:
                raise VersionMismatch(
                    f"NetCDF source_version={nc.source_version!r} 与请求 {version} 不一致"
                )
            if variable not in nc.variables or "quality" not in nc.variables:
                raise SourceError("切片缺少目标变量或 quality 变量")
            value_var = nc.variables[variable]
            quality_var = nc.variables["quality"]
            nc_dims = tuple(value_var.dimensions)
            if list(nc_dims) != dimensions:
                raise VersionMismatch(
                    f"切片维度顺序 {list(nc_dims)} 与目录 {dimensions} 不一致"
                )
            if quality_var.dimensions != value_var.dimensions:
                raise SourceError("quality 与目标变量不同形，无法逐单元标注质量")
            raw = np.array(value_var[:], dtype=np.int16, copy=True)
            quality = np.array(quality_var[:], dtype=np.int8, copy=True)
            coordinates = {dim: list(nc.variables[dim][:].astype(float)) for dim in nc_dims}

            scale_factor = float(value_var.scale_factor)
            add_offset = float(value_var.add_offset)
            fill_value = int(np.int16(value_var._FillValue))

        meta = catalog["variables"][variable]
        # NetCDF classic 属性回读可能有浮点表示误差，按容差比较。
        if not np.isclose(scale_factor, float(meta["scale_factor"])) or not np.isclose(
            add_offset, float(meta["add_offset"])
        ) or fill_value != int(meta["fill"]):
            raise VersionMismatch("切片缩放/缺测约定与目录记录不一致")

        # 坐标必须与该版目录按索引区间取出的坐标逐点相等，
        # 防止旧版索引被解释成新版坐标（例如 r2 在 time 轴中间补入 2.5 秒）。
        for dim in nc_dims:
            start, stop = index_ranges[dim]
            axis = catalog["axes"][dim]
            coords = axis["coordinates"] if isinstance(axis, dict) else axis
            expected = [float(v) for v in coords[start:stop]]
            actual = [float(v) for v in coordinates[dim]]
            if actual != expected:
                raise VersionMismatch(
                    f"变量 {variable} 轴 {dim} 的切片坐标与 {version} 目录不符："
                    f"{actual} != {expected}"
                )

        return Slice(
            variable=variable,
            version=version,
            dimensions=tuple(dimensions),
            coordinates=coordinates,
            raw=raw,
            quality=quality,
            scale_factor=scale_factor,
            add_offset=add_offset,
            fill_value=fill_value,
            index_ranges=dict(index_ranges),
        )
