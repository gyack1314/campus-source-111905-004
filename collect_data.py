#!/usr/bin/env python3
"""采集命令行启动器：以脚本绝对路径配合 python -I 运行。

    python -I /workspace/collect_data.py collect \
        --base http://127.0.0.1:8765 --workdir /data/job1 \
        --select "time=0:5,station=10:30,sample=0:4.6"

脚本所在目录（受信任代码目录）会进入 sys.path[0]；当前工作目录与
PYTHONPATH 在 -I 下均不参与导入，因此数据目录内被植入的同名模块不会被加载。
数据目录一律通过参数显式传入。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collector.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
