#!/usr/bin/env bash
# 复现 .github/workflows/pr_test.yaml 门禁中的 NPU 阶段（A3 双卡）。
# 对应 _selected_tests.yaml 里 a3 分区（npu_type=a3, num_npus=2）在
# linux-aarch64 双卡 runner 上执行的 job：
#   - 'Check NPU availability'                         -> npu-smi info
#   - 'Install vllm-project/vllm from source'          -> checkout pin + VLLM_TARGET_DEVICE=empty uv pip install .
#   - 'Install vllm-project/vllm-ascend with device'   -> triton-ascend + 编译自定义内核
#   - 'Run selected tests with device'                 -> run_selected_tests.sh a3 2 with-device tests/e2e/pull_request/two_card
#
# 镜像：m.daocloud.io/quay.io/ascend/vllm-ascend:nightly-main-a3（A3 芯片，需宿主挂载 NPU 卡）。
# nightly-main-a3 是 all-in-one 镜像，内置的 vllm 是 v0.28.0 tag，其 API 与门禁 pin
# （.github/vllm-main-verified.commit）不一致，因此与 a2/cpu 脚本一致先重装 vllm pin。
set -euo pipefail

PROJECT_DIR="/workspace/vllm-ascend"
cd "${PROJECT_DIR}"

echo "[1/6] 标记 git safe.directory"
git config --global --add safe.directory "${PROJECT_DIR}"

echo "[2/6] 检查 NPU 可用性 (门禁: 'Check NPU availability')"
npu-smi info

echo "[3/6] 激活 CANN 环境并安装开发依赖 (门禁: 'Install packages')"
# shellcheck disable=SC1091
. /usr/local/Ascend/ascend-toolkit/set_env.sh

pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
pip config set global.extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi
pip install uv uc-manager
export UV_SYSTEM_PYTHON=1
export UV_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
export UV_EXTRA_INDEX_URL=https://repo.huaweicloud.com/ascend/repos/pypi

echo "[4/6] 从门禁 pin 提交重装 vllm (门禁: 'Install vllm-project/vllm from source')"
# 门禁 _selected_tests.yaml 里的命令是 `VLLM_TARGET_DEVICE=empty uv pip install .`
# （没有 --no-deps，会按 pin 解析并安装 vllm 依赖）。镜像内置 vllm 是 v0.28.0 tag，
# 需要先切到 pin 并覆盖重装；这里保留 --force-reinstall 覆盖镜像内置 vllm，
# 去掉 --no-deps 让依赖与门禁一致（VLLM_TARGET_DEVICE=empty 只装 common.txt，
# 不含 flashinfer，flashinfer 仅在 cuda/rocm 设备下才装）。
VLLM_PIN="$(tr -d '[:space:]' < "${PROJECT_DIR}/.github/vllm-main-verified.commit")"
VLLM_SRC="/vllm-workspace/vllm"
git config --global --add safe.directory "${VLLM_SRC}"
git -C "${VLLM_SRC}" fetch --depth 1 https://github.com/vllm-project/vllm.git "${VLLM_PIN}"
git -C "${VLLM_SRC}" checkout -f FETCH_HEAD
( cd "${VLLM_SRC}" && VLLM_TARGET_DEVICE=empty uv pip install . --force-reinstall --no-build-isolation )
pip uninstall -y triton

echo "[5/6] 安装 triton-ascend 并带设备编译安装 vllm-ascend (门禁: 'Install ... with device')"
cd "${PROJECT_DIR}"
uv pip install -r requirements-dev.txt
uv pip install --force-reinstall --no-deps triton-ascend==3.2.2
export MAX_JOBS=46
uv pip install -e .

echo "[6/6] 运行 A3 双卡测试 (门禁: 'Run selected tests with device', a3-2 分区)"
# 与门禁一致用 ModelScope 下载模型，避免容器网络直连 huggingface.co 失败。
VLLM_USE_MODELSCOPE=True VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .github/workflows/scripts/run_selected_tests.sh a3 2 with-device tests/e2e/pull_request/two_card

echo "门禁 NPU(A3) 阶段复现完成"