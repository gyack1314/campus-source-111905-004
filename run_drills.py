#!/usr/bin/env python3
"""演练启动器：在进程内启动来源服务并跑全部演练，数据写入参数指定目录。

    python -I /workspace/run_drills.py --data-root /data/drill-data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collector.drill import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
