#!/usr/bin/env bash
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# -------------------------------------------------------------------------
# start-container.sh - 分配空闲 NPU 卡并拉起 devcontainer 的统一入口
#
# 背景：
#   /dev/davinciN 是独占计算设备，同一张卡不能被多个容器同时挂载（否则后启动
#   的容器 npu-smi 报 -8020）。因此必须在 docker create 之前确定分给本容器
#   哪些卡，并把它们写回 devcontainer.json 的 runArgs——这是唯一能让本次
#   启动生效的时机。
#
# 流程：
#   1. select_npu.py     在宿主机探测空闲卡（健康 OK、无计算进程、且不被任何
#                        运行中容器挂载）；随后把分到的卡写回 devcontainer.json
#                        的 runArgs（保留注释），并在 stdout 输出卡号。
#                        空闲卡不足 NPU_REQUEST_COUNT 时以非 0 退出，本脚本
#                        因 set -e 立即终止，不启动容器。
#   2. devcontainer up    唯一一次拉起，runArgs 已含本次分配的卡。
#
# 用法：
#   bash .devcontainer/start-container.sh                 # 默认 1 张空闲卡
#   NPU_REQUEST_COUNT=2 bash .devcontainer/start-container.sh
#
# 说明：
#   - 依赖宿主机存在 npu-smi 与 devcontainer CLI。
#   - 必须先于「Open Folder in Container」执行本脚本；直接把仓库在 VS Code 里
#     Reopen 会使用上一次写回的 runArgs，不会重新分配。
# -------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COUNT="${NPU_REQUEST_COUNT:-1}"

# 1. 探测空闲卡并写回 devcontainer.json；空闲卡不足时非 0 退出，set -e 终止。
ids="$(NPU_REQUEST_COUNT="$COUNT" python3 "$SCRIPT_DIR/select_npu.py")"

echo "==> Allocated NPU card(s): $ids"

# 2. 拉起容器。devcontainer CLI 需在仓库根目录执行，initializeCommand
#    会在此时拉取镜像并准备挂载源。
cd "$REPO_ROOT"
exec devcontainer up --workspace-folder "$REPO_ROOT"