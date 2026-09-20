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
#   哪些卡，并把它们写回指定 devcontainer.json 的 runArgs。
#
# 流程：
#   1. 用 devcontainer CLI 读取指定配置（--config），并给容器打一个稳定的
#      --id-label。该 label 既是识别「本次配置对应的旧容器」的锚点（rebuild
#      时旧容器在删除前仍挂着旧卡，需在探测时排除，避免旧卡无法归还）。
#   2. select_npu.py     探测空闲卡（健康 OK、无计算进程、且不被任何运行中
#                        容器挂载，但排除上面的旧容器），写回 devcontainer.json
#                        的 runArgs，并输出卡号。空闲卡不足时非 0 退出。
#   3. devcontainer up    唯一一次拉起，runArgs 已含本次分配的卡。
#
# 用法（支持多个 devcontainer 配置，通过 --config 指定）：
#   bash .devcontainer/start-container.sh
#   bash .devcontainer/start-container.sh --config .devcontainer/gpu/devcontainer.json
#   NPU_REQUEST_COUNT=2 bash .devcontainer/start-container.sh --config path/to/devcontainer.json
#
# 说明：
#   - 依赖宿主机存在 npu-smi、python3 与 devcontainer CLI。
#   - --config 指定的是实际生效的 devcontainer.json，脚本借此同时完成
#     「写回卡号」与「拉起容器」，二者指向同一个配置。
# -------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COUNT="${NPU_REQUEST_COUNT:-1}"

# 解析 --config。默认主配置 .devcontainer/devcontainer.json。
CONFIG=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            [ "$#" -ge 2 ] || { echo "error: --config 需要参数" >&2; exit 1; }
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "error: 不支持的参数: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    CONFIG="$SCRIPT_DIR/devcontainer.json"
fi

# 配置文件需相对于仓库根目录解析为绝对路径，供 devcontainer --config 使用。
case "$CONFIG" in
    /*) CONFIG_ABS="$CONFIG" ;;
    *)  CONFIG_ABS="$REPO_ROOT/$CONFIG" ;;
esac

# label 用「脚本目录名」做锚点，保证同一份配置（同一 .devcontainer 场景）每次
# 重建都命中同一个旧容器；不同配置之间互不干扰。
ID_LABEL="vllm-ascend.devconfig=${CONFIG_ABS}"
EXCLUDE_LABEL="vllm-ascend.devconfig=${CONFIG_ABS}"

# 1. 探测空闲卡并写回 devcontainer.json（旧容器占用的卡会被排除，可被复用）。
ids="$(NPU_REQUEST_COUNT="$COUNT" python3 "$SCRIPT_DIR/select_npu.py" \
    --config "$CONFIG_ABS" \
    --exclude-container-label "$EXCLUDE_LABEL")"

echo "==> Allocated NPU card(s): $ids"

# 2. 拉起容器。--remove-existing-container 配合稳定 --id-label，可正确覆盖
#    旧容器并释放其占用的卡；否则旧容器不删，卡依旧被占用。
cd "$REPO_ROOT"
exec devcontainer up \
    --workspace-folder "$REPO_ROOT" \
    --config "$CONFIG_ABS" \
    --id-label "$ID_LABEL" \
    --remove-existing-container