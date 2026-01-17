#!/bin/bash
# 脚本名：train_cosyvoice.sh
# 功能：CosyVoice2.0 标贝女声模型训练全自动化脚本（Linux适配版）
# 编码设置（Linux UTF-8 全局生效）
export LC_ALL=en_US.UTF-8
export PYTHONIOENCODING=utf-8

# ======================== 配置项（根据服务器实际路径修改） ========================
PROJECT_ROOT="/home/student/work/linjunru/CosyVoice2"
CONDA_ENV_NAME="cosyvoice_env"
# 关键：手动指定conda的安装路径（从之前的安装日志来）
CONDA_PATH="/home/student/miniconda3"
DATA_DIR="${PROJECT_ROOT}/my_biaobei_dataset"
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_models/Qwen2-0.5B"
TRAIN_SCRIPT="${PROJECT_ROOT}/cosyvoice/bin/train.py"
CONFIG_FILE="${PROJECT_ROOT}/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
EXP_MODEL_DIR="${PROJECT_ROOT}/exp/cosyvoice2_biaobei/checkpoints"
EXP_LOG_DIR="${PROJECT_ROOT}/exp/cosyvoice2_biaobei/logs"
CUDA_VISIBLE_DEVICES="0"
NUM_GPUS="1"
NUM_WORKERS="0"
PREFETCH="100"

# ======================== 函数：打印彩色日志 ========================
info() {
    echo -e "\033[32mInfo: $1\033[0m"
}
error() {
    echo -e "\033[31mError: $1\033[0m"
    exit 1
}

# ======================== 第一步：初始化并验证conda环境（核心修复） ========================
info "The script will automatically activate the CosyVoice conda environment."
echo ""

# 步骤1：强制加载conda环境变量（非交互式shell必备）
export PATH="${CONDA_PATH}/bin:${PATH}"
# 步骤2：初始化conda（一次性操作，重复执行无影响）
conda init bash > /dev/null 2>&1
# 步骤3：重新加载bash配置，让conda生效
source ~/.bashrc

# 步骤4：检查conda环境是否存在
info "Checking Conda environment (${CONDA_ENV_NAME})..."
if ! conda info --envs | grep -q "${CONDA_ENV_NAME}"; then
    error "Conda environment not found! Name: ${CONDA_ENV_NAME}\nPlease create it first: conda create -n ${CONDA_ENV_NAME} python=3.10 -y"
fi

# 步骤5：获取conda环境的绝对路径（非交互式shell激活核心）
CONDA_ENV_PATH=$(conda info --envs | grep "${CONDA_ENV_NAME}" | awk '{print $2}')
if [ -z "${CONDA_ENV_PATH}" ]; then
    error "Failed to get path of Conda environment: ${CONDA_ENV_NAME}"
fi
info "Conda environment path: ${CONDA_ENV_PATH}"

# ======================== 第二步：检查所有关键文件/目录 ========================
info "======================================"
info "[Step 1: Check File and Directory Paths]"
info "======================================"

# 检查项目根目录
if [ ! -d "${PROJECT_ROOT}" ]; then
    error "Project root directory not found! Path: ${PROJECT_ROOT}"
fi

# 检查配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    error "Configuration file not found! Path: ${CONFIG_FILE}\nPlease confirm that cosyvoice2.yaml exists in examples/libritts/cosyvoice2/conf/"
fi

# 检查数据集目录
if [ ! -d "${DATA_DIR}" ]; then
    error "Dataset directory not found! Path: ${DATA_DIR}"
fi

# 检查训练音频目录
if [ ! -d "${DATA_DIR}/train/wavs" ]; then
    error "Training audio directory not found! Path: ${DATA_DIR}/train/wavs"
fi

# 检查训练元数据
if [ ! -f "${DATA_DIR}/train/metadata.csv" ]; then
    error "Training metadata file not found! Path: ${DATA_DIR}/train/metadata.csv"
fi

# 检查预训练模型
if [ ! -d "${PRETRAINED_MODEL_DIR}" ]; then
    error "Pretrained model directory not found! Path: ${PRETRAINED_MODEL_DIR}"
fi

# 检查训练脚本
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    error "Training script not found! Path: ${TRAIN_SCRIPT}\nPlease confirm that train.py exists in cosyvoice/bin/"
fi

# 自动创建训练结果目录
if [ ! -d "${EXP_MODEL_DIR}" ]; then
    mkdir -p "${EXP_MODEL_DIR}"
    mkdir -p "${EXP_LOG_DIR}"
    info "Created experiment directory successfully: ${PROJECT_ROOT}/exp/cosyvoice2_biaobei"
fi

# ======================== 第三步：数据预处理（用conda run执行，非交互式shell必备） ========================
echo ""
info "======================================"
info "[Step 2: Data Preprocessing (Generate Model Input Features)]"
info "======================================"

# # 2.1 提取说话人嵌入特征（campplus.onnx）
# info "Extracting speaker embedding features..."
# CAMPPLUS_ONNX="${PRETRAINED_MODEL_DIR}/campplus.onnx"
# if [ ! -f "${CAMPPLUS_ONNX}" ]; then
#     error "Pretrained model file not found! Path: ${CAMPPLUS_ONNX}\nPlease confirm that campplus.onnx exists in CosyVoice2-0.5B directory!"
# fi

# for split in train dev; do
#     if [ ! -f "${DATA_DIR}/${split}/spk2embedding.pt" ]; then
#         # 关键：用conda run替代conda activate，适配非交互式shell
#         conda run -n "${CONDA_ENV_NAME}" python "${PROJECT_ROOT}/tools/extract_embedding.py" \
#             --dir "${DATA_DIR}/${split}" \
#             --onnx_path "${CAMPPLUS_ONNX}"
#         if [ $? -ne 0 ]; then
#             error "Failed to extract speaker embedding for ${split} dataset!"
#         fi
#     fi
# done

# # 2.2 提取离散语音Token（speech_tokenizer_v2.onnx）
# info "Extracting discrete speech tokens..."
# TOKENIZER_ONNX="${PRETRAINED_MODEL_DIR}/speech_tokenizer_v2.onnx"
# for split in train dev; do
#     if [ ! -f "${DATA_DIR}/${split}/utt2speech_token.pt" ]; then
#         conda run -n "${CONDA_ENV_NAME}" python "${PROJECT_ROOT}/tools/extract_speech_token.py" \
#             --dir "${DATA_DIR}/${split}" \
#             --onnx_path "${TOKENIZER_ONNX}"
#         if [ $? -ne 0 ]; then
#             error "Failed to extract speech tokens for ${split} dataset!"
#         fi
#     fi
# done

# # 2.3 生成Parquet高效训练文件
# info "Generating Parquet format training files..."
# for split in train dev; do
#     if [ ! -d "${DATA_DIR}/${split}/parquet" ]; then
#         mkdir -p "${DATA_DIR}/${split}/parquet"
#         conda run -n "${CONDA_ENV_NAME}" python "${PROJECT_ROOT}/tools/make_parquet_list.py" \
#             --num_utts_per_parquet 1000 \
#             --num_processes 10 \
#             --src_dir "${DATA_DIR}/${split}" \
#             --des_dir "${DATA_DIR}/${split}/parquet"
#         if [ $? -ne 0 ]; then
#             error "Failed to generate Parquet files for ${split} dataset!"
#         fi
#     fi
# done

# # 2.4 生成训练/验证集数据列表
# info "Generating training/validation data lists..."
# for split in train dev; do
#     if [ ! -f "${DATA_DIR}/${split}.data.list" ]; then
#         cat "${DATA_DIR}/${split}/parquet/data.list" > "${DATA_DIR}/${split}.data.list"
#         if [ $? -ne 0 ]; then
#             error "Failed to generate data list for ${split} dataset!"
#         fi
#     fi
# done

# ======================== 第四步：启动模型训练 ========================
echo ""
info "======================================"
info "[Step 3: Start Model Training (Fully Automated)]"
info "======================================"
info "Configuration file path: ${CONFIG_FILE}"
info "Training dataset: ${DATA_DIR}/train.data.list"
info "Validation dataset: ${DATA_DIR}/dev.data.list"
info "Model save directory: ${EXP_MODEL_DIR}"
info "Training log directory: ${EXP_LOG_DIR}"
info "Available GPU devices: ${CUDA_VISIBLE_DEVICES}"
info "======================================"
info "Training takes a long time (several days for 10,000 samples on a single GPU), do not close the terminal!"
info "Loss values will be printed in real time, a continuous decrease indicates normal training~"
echo ""

# 设置GPU环境变量
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
# 核心新增：强制禁用Deepspeed
export TRANSFORMERS_NO_DEEPSPEED=1
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_AIO=0
export DS_BUILD_UTILS=0
# 即使没有CUDA，也强制使用CPU模式
export CUDA_VISIBLE_DEVICES="0"
export FORCE_CPU=0
export PYTHONWARNINGS="ignore"
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

# 核心修改1：执行训练命令并捕获真实退出码
info "Starting model training..."
conda run -n "${CONDA_ENV_NAME}" python -u "${TRAIN_SCRIPT}" \
    --model llm \
    --config "${CONFIG_FILE}" \
    --qwen_pretrain_path "${PRETRAINED_MODEL_DIR}" \
    --train_data "${DATA_DIR}/train.data.list" \
    --cv_data "${DATA_DIR}/dev.data.list" \
    --model_dir "${EXP_MODEL_DIR}" \
    --tensorboard_dir "${EXP_LOG_DIR}" \
    --use_amp \
    --num_workers "${NUM_WORKERS}"
    --save_per_epoch 15 \          # 新增
    --save_per_step -1 \           # 新增
    --max_frames_in_batch 3000     # 新增

# 保存训练命令的退出码（核心！0=成功，非0=失败）
TRAIN_EXIT_CODE=$?

# ======================== 第五步：训练结果判断（最终版） ========================
echo ""
info "======================================"
info "[Step 4: Training Process Completed]"
info "======================================"

# 核心修复：新增变量判空逻辑
if [ -z "${TRAIN_EXIT_CODE}" ]; then
    # 未执行训练步骤时，提示预处理完成
    echo -e "\033[32m✅ Data preprocessing completed successfully! (No training executed)\033[0m"
    exit 0
elif [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    # 训练成功
    echo -e "\033[32m🎉 Congratulations! Training completed successfully!\033[0m"
    echo -e "\033[32m👉 Final model weights: ${EXP_MODEL_DIR} (best_model.pth is the optimal model)\033[0m"
    echo -e "\033[32m👉 Complete training logs: ${EXP_LOG_DIR}\033[0m"
    exit 0
else
    # 训练失败 - 替换原CUDA_HOME误导性提示为真实单GPU错误排查指引
    echo -e "\033[31m❌ Training terminated abnormally! Exit code: ${TRAIN_EXIT_CODE}\033[0m"
    echo -e "\033[31m======================================\033[0m"
    echo -e "\033[31mKey Tips (Single GPU Training):\033[0m"
    echo -e "\033[31m1. CUDA is working normally (GPU 0 detected), ignore CUDA_HOME prompts\033[0m"
    echo -e "\033[31m2. Check if distributed training code is commented: cosyvoice/utils/executor.py (dist.barrier())\033[0m"
    echo -e "\033[31m3. Check info_dict fields: cosyvoice/utils/train_utils.py (tag/lr fields initialized)\033[0m"
    echo -e "\033[31m4. Check dataset field compatibility: cosyvoice/dataset/processor.py (audio_data → wav_path)\033[0m"
    echo -e "\033[31m5. Check process group destroy: cosyvoice/bin/train.py (try-except for dist.destroy_process_group)\033[0m"
    exit ${TRAIN_EXIT_CODE}
fi