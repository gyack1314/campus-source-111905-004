"""实验阵列 NetCDF 切片采集工具。

只依赖 requests / numpy / scipy，按物理坐标向来源服务申请切片，
本地保留每个切片的版次与坐标来源，并生成对应实际覆盖单元、缺口与版次的完成清单。
"""
from .coords import (
    AxisSelection,
    describe_axis,
    guard_distinct_grids,
    parse_selection,
    plan_pages,
    select_indices,
)
from .protocol import (
    BadRequest,
    Slice,
    SourceClient,
    SourceError,
    TooLarge,
    VersionMismatch,
    VersionRetired,
)
from .job import JobStore, open_or_create_job
from .assemble import assemble_job, assemble_variable

__all__ = [
    "AxisSelection",
    "BadRequest",
    "JobStore",
    "Slice",
    "SourceClient",
    "SourceError",
    "TooLarge",
    "VersionMismatch",
    "VersionRetired",
    "assemble_job",
    "assemble_variable",
    "describe_axis",
    "guard_distinct_grids",
    "open_or_create_job",
    "parse_selection",
    "plan_pages",
    "select_indices",
]
