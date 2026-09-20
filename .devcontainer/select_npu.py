#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------
"""分配空闲 NPU 卡：探测 → 写回 devcontainer.json → 输出卡号。

在宿主机、创建容器之前执行（由 start-container.sh 调用）。/dev/davinciN 为
独占设备，必须先在宿主机确定分给本容器哪些卡，再写入 devcontainer.json 的
runArgs，使随后的 docker create 只挂载这些卡。

支持多个 devcontainer 配置：通过 --config 指定要写回的 devcontainer.json，
不再硬编码固定路径；配合 --exclude-container-label 可在探测时排除「本次即将
覆盖（重建）的旧容器」——旧容器在删除前仍挂载着旧卡，若不排除，这些本应归还
的卡会被误判为「被占用」，导致 rebuild 时可用卡不足或卡号漂移。

空闲判定（同时满足四条件）：
  1. 卡设备存在（/dev/davinci<N>）；
  2. npu-smi info 显示健康状态 OK；
  3. npu-smi info 显示该卡无计算进程（排除宿主机直接跑的进程）；
  4. 不被任何运行中容器挂载（docker inspect，权威来源——覆盖「挂载但空跑」
     这种 npu-smi 和 fuser 都检测不到的情形），但排除 --exclude-container-label
     命中的旧容器。

用法：
    python3 select_npu.py                                          # 默认主配置，1 张卡
    python3 select_npu.py --config .devcontainer/gpu/devcontainer.json
    NPU_REQUEST_COUNT=2 python3 select_npu.py --exclude-container-label 'msopprof.devconfig=/x.json'

行为：
    - stdout 只输出选中的卡号，逗号分隔（例如 "0,2"），供调用方直接使用。
    - 诊断日志一律输出到 stderr，不污染 stdout。
    - 退出码：0 表示成功；非 0 表示失败（未发现设备 / npu-smi 执行失败 /
      空闲卡不足 / 写回配置失败）。
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# 可在多容器间共享的控制/管理设备，不参与分配、始终保留在 runArgs 中。
CARD_DEVICE_RE = re.compile(r"^--device=/dev/davinci(\d+)$")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "devcontainer.json")


def docker_occupied_cards(exclude_label=None):
    """返回被运行中容器挂载的 /dev/davinciN 卡号集合（可排除指定 label 的容器）。

    /dev/davinciN 为独占设备，一旦被某个容器通过 docker create 挂载即被占用
    ——无论容器内是否有计算进程，npu-smi info（只反映计算进程）和 fuser（只
    反映打开 fd）都检测不到这种「挂载但空跑」的占用，只有 Docker 的挂载信息
    能精确反映。

    exclude_label: `key=value` 形式的 docker label。命中的容器属于「本次即将
    重建」的旧容器，其占用的卡应视为即将释放、可被复用，故从占用集合中排除。
    """
    occupied = set()
    exclude_ids = set()
    if exclude_label:
        try:
            exclude_ids = set(subprocess.run(
                ["docker", "ps", "-q", "--filter", "label=" + exclude_label],
                capture_output=True, text=True, timeout=15,
            ).stdout.split())
        except Exception as e:  # noqa: BLE001
            print("docker ps (exclude label) failed: %s" % e, file=sys.stderr)

    try:
        ids = subprocess.run(
            ["docker", "ps", "-q"], capture_output=True, text=True, timeout=15,
        ).stdout.split()
    except Exception as e:  # noqa: BLE001
        print("docker ps failed: %s (assume no docker occupancy)" % e, file=sys.stderr)
        return occupied

    for cid in ids:
        if cid in exclude_ids:
            continue
        try:
            raw = subprocess.run(
                ["docker", "inspect", cid],
                capture_output=True, text=True, timeout=15,
            ).stdout
            info = json.loads(raw)
            for dev in info[0]["HostConfig"].get("Devices") or []:
                m = re.fullmatch(r"/dev/davinci(\d+)", dev.get("PathOnHost") or "")
                if m:
                    occupied.add(int(m.group(1)))
        except Exception as e:  # noqa: BLE001 - 单个容器 inspect 失败不应中断
            print("docker inspect %s failed: %s" % (cid[:12], e), file=sys.stderr)
            continue
    return occupied


def detect_free_cards(count, exclude_label=None):
    """返回按卡号升序的空闲卡列表，最多 count 张。"""
    # 从 /dev/davinci<N> 枚举宿主机真实存在的卡，不做固定数量假设。
    cards = set()
    for path in glob.glob("/dev/davinci[0-9]*"):
        m = re.fullmatch(r"/dev/davinci(\d+)", path)
        if m:
            cards.add(int(m.group(1)))
    cards = sorted(cards)
    if not cards:
        print("no /dev/davinci<N> device found", file=sys.stderr)
        sys.exit(1)

    # 查询 npu-smi info（用于健康状态 + 宿主机直接运行的进程）。
    try:
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception as e:  # noqa: BLE001 - 此处需上报任何探测失败原因
        print("npu-smi info failed: %s" % e, file=sys.stderr)
        sys.exit(1)

    # 解析板卡表头行的 Health，非 OK 的卡不参与分配。
    # 板卡表头形如：| 0     910B3    | OK    | ...（名称列非空，能命中）；
    # Chip 行形如：| 0              | 0000:C1:00.0 | ...（名称列为空，不会命中）。
    bad_health = {
        int(m.group(1))
        for m in re.finditer(r"^\|\s*(\d+)\s+\S+\s+\|\s*(\S+)", out, re.MULTILINE)
        if m.group(2) != "OK"
    }

    # 有计算进程的卡（宿主机直接跑的进程）也排除。
    busy_by_process = set(cards) - {
        int(m.group(1))
        for m in re.finditer(r"No running processes found in NPU\s+(\d+)", out)
    }

    # 被运行中容器挂载的卡排除——最关键的一层，覆盖「挂载但无进程」。
    occupied = docker_occupied_cards(exclude_label)

    free = [
        c for c in cards
        if c not in bad_health
        and c not in busy_by_process
        and c not in occupied
    ]
    return free[:count]


def find_runargs_block(text):
    """定位 "runArgs" 后第一个完整 [ ... ] 数组，返回切片下标 (start, end)。

    start 指向 '['，end 指向闭合 ']' 之后的位置。按字符扫描并感知字符串
    字面量，值内出现方括号时不会误判。"""
    key_idx = text.index('"runArgs"')
    start = text.index("[", key_idx)
    depth = 0
    in_str = False
    escape = False
    i = start
    while i < len(text):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return start, i + 1
        i += 1
    raise ValueError("runArgs 数组未闭合")


def apply_runargs(cards, config_path):
    """把 cards 写回指定 devcontainer.json 的 runArgs，保留注释与格式。"""
    with open(config_path, encoding="utf-8") as f:
        text = f.read()

    start, end = find_runargs_block(text)
    run_args = json.loads(text[start:end])

    # 清除独占计算设备（精确匹配 --device=/dev/davinci<数字>），保留共享管理
    # 设备与其他 runArgs（--network、--privileged 等）。
    run_args = [a for a in run_args if not CARD_DEVICE_RE.match(a)]
    for card in cards:
        run_args.append("--device=/dev/davinci%d" % card)

    # 重新生成数组文本，缩进与源文件一致（元素 4 空格、闭合括号 2 空格）。
    block_lines = json.dumps(run_args, indent=4, ensure_ascii=False).splitlines()
    block_lines[-1] = "  ]"
    new_block = "\n".join(block_lines)

    text = text[:start] + new_block + text[end:]

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser(
        description="分配空闲 NPU 卡并写回 devcontainer.json 的 runArgs。",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="要写回的 devcontainer.json 路径（默认 .devcontainer/devcontainer.json）")
    parser.add_argument("--exclude-container-label", default=None,
                        help="docker label(key=value)；命中的容器在空闲卡检测时排除（用于覆盖旧容器）")
    args = parser.parse_args()

    count = int(os.environ.get("NPU_REQUEST_COUNT", "1"))
    chosen = detect_free_cards(count, exclude_label=args.exclude_container_label)

    if len(chosen) < count:
        print(
            "insufficient free NPU cards: requested %d, got %d (free: %s)"
            % (count, len(chosen), chosen),
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        apply_runargs(chosen, args.config)
    except Exception as e:  # noqa: BLE001 - 写回失败需上报并终止
        print("failed to update devcontainer.json: %s" % e, file=sys.stderr)
        sys.exit(1)

    print("updated runArgs with NPU cards: %s" % chosen, file=sys.stderr)

    # stdout 只输出卡号，供 shell 捕获。
    print(",".join(map(str, chosen)))


if __name__ == "__main__":
    main()