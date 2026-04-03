#!/bin/bash
# ============================================================
#  LagMemo yjr-develop 分支 — GPU 1 测试脚本
#  生成时间: 2026-04-03
#
#  用法:
#    bash run_eval_gpu1.sh [mode] [scene] [input_data]
#
#    mode      : glue (默认) | data_record | goat
#    scene     : 场景名，默认 5cdEh9F2hJL
#                可选: 4ok3usBNeis / Nfvxx8J5NCo / TEEsavR23oF / all
#    input_data: 数据集目录，默认 3_episode_data
#                可选: new_data3 / 3_episode_data
#
#  示例:
#    bash run_eval_gpu1.sh                              # glue, 5cdEh9F2hJL, 3_episode_data
#    bash run_eval_gpu1.sh glue 5cdEh9F2hJL new_data3
#    bash run_eval_gpu1.sh data_record 5cdEh9F2hJL 3_episode_data
#    bash run_eval_gpu1.sh goat 5cdEh9F2hJL new_data3
# ============================================================

set -e

# ---------- 参数 ----------
MODE=${1:-"glue"}
SCENE=${2:-"5cdEh9F2hJL"}
INPUT_DATA=${3:-"3_episode_data"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="datadump/test_gpu1_${MODE}_${TIMESTAMP}"

# ---------- 路径 ----------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SEEM_CKPT="./src/third_party/seem/checkpoints/seem_focall_v0.pt"
SEEM_YAML="./src/third_party/seem/configs/seem/focall_unicl_lang_demo.yaml"
CLIP_CKPT="./src/third_party/ml-mobileclip/checkpoints/mobileclip_s0.pt"

# ---------- 根据 mode 选脚本和 config ----------
case "$MODE" in
  glue)
    SCRIPT="project/habitat_lagmemo/eval_episode_glue.py"
    BASELINE_CFG="project/config/agent/hm3d_eval.yaml"
    ;;
  data_record)
    SCRIPT="project/habitat_lagmemo/eval_episode_data_record.py"
    BASELINE_CFG="project/config/agent/hm3d_eval.yaml"
    ;;
  goat)
    SCRIPT="project/habitat_lagmemo/eval_episode_goat.py"
    BASELINE_CFG="project/config/agent/hm3d_eval_goat.yaml"
    ;;
  *)
    echo "Unknown mode: $MODE. Choose from: glue / data_record / goat"
    exit 1
    ;;
esac

# ---------- GPU 设置 ----------
# GPU 1: A100 80GB PCIe（基本空闲）
export CUDA_VISIBLE_DEVICES=1
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# ---------- 运行 ----------
cd "${REPO_ROOT}"

echo "================================================"
echo " LagMemo Eval — yjr-develop"
echo " Mode        : ${MODE} (${SCRIPT})"
echo " Scene       : ${SCENE}"
echo " Dataset     : ${INPUT_DATA}"
echo " Output      : ${OUTPUT_PATH}"
echo " GPU (phys)  : 1 (CUDA_VISIBLE_DEVICES=1)"
echo " Start time  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lagmemo

python "${SCRIPT}" \
    --scenes "${SCENE}" \
    --input_data "${INPUT_DATA}" \
    --output_path "${OUTPUT_PATH}" \
    --seem_ckpt_path "${SEEM_CKPT}" \
    --seem_yaml_path "${SEEM_YAML}" \
    --mobileclip_ckpt_path "${CLIP_CKPT}" \
    --baseline_config_path "${BASELINE_CFG}" \
    --habitat_config_path "project/config/habitat/lagmemo_hm3d.yaml"

echo "================================================"
echo " Done. Results → ${OUTPUT_PATH}"
echo " End time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
