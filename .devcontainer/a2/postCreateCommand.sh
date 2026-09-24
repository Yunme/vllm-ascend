#!/usr/bin/env bash
# 复现 .github/workflows/pr_test.yaml 门禁中的 NPU 阶段（A2 单卡 910B）。
# 对应 _selected_tests.yaml 里 a2 分区（npu_type=a2, num_npus=1）在
# linux-aarch64-a2b3-1 runner 上执行的 job：
#   - 'Check NPU availability'                         -> npu-smi info
#   - 'Install vllm-project/vllm from source'          -> checkout pin + VLLM_TARGET_DEVICE=empty uv pip install .
#   - 'Install vllm-project/vllm-ascend with device'   -> triton-ascend + 编译自定义内核
#   - 'Run selected tests with device'                 -> run_selected_tests.sh a2 1 with-device tests/e2e/pull_request/one_card
#
# 镜像：quay.io/ascend/vllm-ascend:nightly-main（A2 910B 芯片，需宿主挂载 NPU 卡）。
# nightly-main 是 all-in-one 镜像，内置的 vllm 是 v0.28.0 tag，其 API 与门禁 pin
# （.github/vllm-main-verified.commit）不一致，因此与 cpu 脚本一致先重装 vllm pin。
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
export MAX_JOBS=23
uv pip install -e . --no-build-isolation

echo "[6/6] 预下载 A2 分区模型并运行单卡测试 (门禁: 'Run selected tests with device', a2-1 分区)"
# 门禁在跑测试前会由 labeled_download_model_dataset.yaml（[model-dataset-download] label）
# 把 model_dataset_list.json 里 a2 分区的模型预下载到 /root/.cache/modelscope，测试阶段再
# 以 HF_HUB_OFFLINE=1 + VLLM_USE_MODELSCOPE=True 离线读取。devcontainer 的 /root/.cache
# 挂载宿主机缓存，但可能尚未包含本次测试需要的模型（例如 test_lora_with_spec_decode.py 的
# Qwen/Qwen3-1.7B、vllm-ascend/Qwen3-1.7B_eagle3、vllm-ascend/qwen-linear-algebra-coder），
# 离线 snapshot_download 会报 "Cannot find the requested files in the cached path"。
# 这里只下载当前选中测试实际引用的 org/repo，而非 model_dataset_list.json 的全量列表：列表含
# 300+ 模型，既有 ModelScope 上不存在的条目（如 ByteDance-Seed/BAGEL-7B-MoT 会 404 中断
# 整个脚本），也含 70B/235B/397B 等超大模型，全量下载不现实。对已缓存模型只校验，对无法
# 下载的条目跳过、不中断。
python3 - <<'PY'
import pathlib
import re
import sys

from modelscope import snapshot_download

files = [
    f
    for f in pathlib.Path("tests/e2e/pull_request/one_card").rglob("test_*.py")
    if "_310p" not in str(f)
    and f.name
    not in {
        "test_uva.py",
        "test_llama32_lora.py",
        "test_qwen3_multi_loras.py",
        "test_qwen3_5_0_8b.py",
    }
]
# conftest.py 里定义的 session fixture（如 lora 的 *_files、test_vlm 的 vl_config）引用
# 的 repo 不出现在 test_*.py 里，必须一并扫描，否则会漏下载（曾导致 test_ilama_lora 失败）。
files.append(pathlib.Path("tests/e2e/conftest.py"))
files += list(pathlib.Path("tests/e2e/pull_request/one_card").rglob("conftest.py"))
pat = re.compile(r'["\']([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+?)["\']')
models = {
    m
    for f in files
    for m in pat.findall(f.read_text(errors="ignore"))
    if m.count("/") == 1 and " " not in m
}
for model in sorted(models):
    print(f"Downloading {model}")
    try:
        snapshot_download(model_id=model)
    except Exception as exc:
        print(f"WARN: skip {model}: {exc}", file=sys.stderr)
PY

# 门禁 select_tests.py 会把 one_card 目录展开为逐个 test_*.py 文件，再由
# run_selected_tests.sh 每个文件单独 pytest（独立进程）。若直接把目录整体传给
# pytest，单进程会按字母序收集整个目录：test_minimax_m3_sparse_attn.py 先于
# test_model_runner_v1_with_device.py 被收集，其 import vllm_ascend.models.minimax_m3
# 会触发 minimax_m3_vl.py 顶层注入一个缺少 _AR_RESIDUAL_RMS_NORM 的假模块
# (vllm.model_executor.layers.fused_allreduce_gemma_rms_norm)，污染后续收集，导致
# test_model_runner_v1_with_device.py 报
# "cannot import name '_AR_RESIDUAL_RMS_NORM' ... (unknown location)"。
# 因此与门禁一致，展开为文件列表逐个执行。同时排除 _310p 子目录（门禁中它们由
# 310p-1 runner 单独执行，非 a2b3-1 单卡 910B）、skip_tests 里的 test_uva.py，
# 以及三个强制走 HuggingFace Hub 的测试：lora/test_llama32_lora.py、
# lora/test_qwen3_multi_loras.py、test_qwen3_5_0_8b.py。它们 fixture/顶部
# @patch VLLM_USE_MODELSCOPE=False 硬编码从 HF Hub 读取模型（如
# jeeejeee/llama32-3b-text2sql-spider、charent/self_cognition_*、Qwen3.5-0.8B），
# 而 devcontainer 与门禁一致通过 ModelScope 下载模型（VLLM_USE_MODELSCOPE=True）
# 且离线（HF_HUB_OFFLINE=1），本地无 HF 缓存，会报 LocalEntryNotFoundError，
# 无法在容器内复现，故跳过。
A2_TESTS=()
while IFS= read -r _t; do
  A2_TESTS+=("${_t}")
done < <(find tests/e2e/pull_request/one_card -name 'test_*.py' \
  -not -path '*/_310p/*' \
  -not -name 'test_uva.py' \
  -not -name 'test_llama32_lora.py' \
  -not -name 'test_qwen3_multi_loras.py' \
  -not -name 'test_qwen3_5_0_8b.py' \
  | sort)
HF_HUB_OFFLINE=1 VLLM_USE_MODELSCOPE=True VLLM_WORKER_MULTIPROC_METHOD=spawn \
  .github/workflows/scripts/run_selected_tests.sh a2 1 with-device "${A2_TESTS[@]}"

echo "门禁 NPU(A2) 阶段复现完成"