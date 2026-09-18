#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------
"""占用指定 NPU 卡的 demo，用于验证空闲卡探测逻辑。

运行后进程会在目标卡上分配显存并做一次矩阵乘，然后保持存活，使
`npu-smi info` 中该卡不再显示「No running processes found in NPU N」，
从而能被 activate_npu.sh / select_npu.py 判定为「非空闲」。

依赖：容器内需已装 torch + torch_npu（CANN 工具链通常自带）。

用法：
    python3 occupy_npu.py                       # 占用卡 0，无限期
    python3 occupy_npu.py -d 3                  # 占用卡 3
    python3 occupy_npu.py -d 2 -t 300           # 占用卡 2，300 秒后自动释放
    python3 occupy_npu.py -d 1 -s 8192          # 占用卡 1，分配 8192x8192 显存

参数：
    -d, --device  目标 NPU 卡号（默认 0）
    -t, --timeout 占用时长，秒；0 表示无限期（默认 0，Ctrl+C 释放）
    -s, --size    分配的 fp32 方阵边长，控制显存占用量（默认 4096）
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="占用指定 NPU 卡（demo），用于验证空闲卡探测。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="要占用的 NPU 卡号")
    parser.add_argument("-t", "--timeout", type=int, default=0,
                        help="占用时长（秒），0 表示无限期")
    parser.add_argument("-s", "--size", type=int, default=4096,
                        help="分配的 fp32 方阵边长，控制显存占用量")
    args = parser.parse_args()

    try:
        import torch
        import torch_npu  # noqa: F401 - 注册 NPU 后端，供 torch.npu 使用
    except ImportError as e:
        sys.exit(f"缺少 torch / torch_npu 依赖: {e}（需在带 CANN 工具链的镜像内运行）")

    # 把当前进程绑定到指定卡，后面的显存分配与计算都落在这张卡上。
    torch.npu.set_device(args.device)

    # 分配一块显存并做一次真实计算，确保驱动记录该卡被本进程占用。
    x = torch.randn(args.size, args.size, device="npu")
    y = x @ x
    torch.npu.synchronize()

    print(f"[occupy] 已占用 NPU {args.device}，pid={os.getpid()}")
    print(f"[occupy] 已分配 {args.size}x{args.size} fp32 显存（约 "
          f"{args.size * args.size * 4 / 1024 / 1024:.0f} MB）")
    print("[occupy] 可在另一个终端执行 `npu-smi info` 观察该卡已非空闲")

    try:
        if args.timeout > 0:
            print(f"[occupy] 将保持 {args.timeout} 秒后自动释放...")
            time.sleep(args.timeout)
        else:
            print("[occupy] 无限期占用中，按 Ctrl+C 释放...")
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        # 显式释放并同步，便于观察占用结束后卡回到空闲态。
        del x, y
        torch.npu.synchronize()
        print(f"[occupy] 已释放 NPU {args.device}，进程退出")


if __name__ == "__main__":
    main()