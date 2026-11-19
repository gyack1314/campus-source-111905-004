"""采集工具命令行。

用法示例（脚本与数据目录分离，数据目录由参数显式给出）：

    python -I -m collector.cli catalog --base http://127.0.0.1:8765
    python -I -m collector.cli collect --base http://127.0.0.1:8765 \
        --workdir ./data/job1 --select "time=0:5,station=10:30,sample=0:4.6"
    python -I -m collector.cli rebuild --base http://127.0.0.1:8765 --workdir ./data/job1
    python -I -m collector.cli status --workdir ./data/job1
"""
from __future__ import annotations

import argparse
import json
import sys

from .assemble import assemble_job
from .job import open_or_create_job
from .protocol import SourceClient, VersionRetired


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="实验阵列 NetCDF 切片采集工具")
    parser.add_argument("--base", default="http://127.0.0.1:8765", help="来源服务地址")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("catalog", help="读取当前目录")

    p_collect = sub.add_parser("collect", help="按物理坐标创建并执行采集作业")
    p_collect.add_argument("--workdir", required=True, help="作业数据目录（独立于代码目录）")
    p_collect.add_argument("--select", required=True, help="如 time=0:5,station=10:30,sample=0:4.6")

    p_rebuild = sub.add_parser("rebuild", help="旧版退休后按当前版次重建未覆盖坐标")
    p_rebuild.add_argument("--workdir", required=True)

    p_status = sub.add_parser("status", help="拼装（或重拼）作业并输出完成清单")
    p_status.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)
    client = SourceClient(args.base)

    if args.command == "catalog":
        _print(client.get_catalog())
        return 0

    if args.command == "collect":
        store = open_or_create_job(args.workdir, client, args.select)
        try:
            store.collect(client)
        except VersionRetired as exc:
            print(f"[中断] {exc}；已落盘切片保留可解释，请执行 rebuild 或查看 status", file=sys.stderr)
        summary = assemble_job(store)
        _print(summary)
        return 0 if summary["job_status"] == "complete" else 2

    if args.command == "rebuild":
        store = open_or_create_job(args.workdir, client)
        store.rebuild_on_retirement(client)
        summary = assemble_job(store)
        _print(summary)
        return 0 if summary["job_status"] in ("complete", "mixed_version") else 2

    if args.command == "status":
        store = open_or_create_job(args.workdir, client)
        _print(assemble_job(store))
        return 0

    parser.error(f"未知命令 {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
