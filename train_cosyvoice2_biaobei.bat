@echo off
chcp 65001 > nul
set "PYTHONIOENCODING=utf-8"
setlocal enabledelayedexpansion
set "ENABLE_LONG_PATH=1"

echo Info: The script will automatically activate the CosyVoice virtual environment.
echo.

:: ======================================
:: ======================================
set "PROJECT_ROOT=D:\CosyVoice2"
set "VENV_ROOT=D:\CosyVoice2\.venv"  :: 匹配你的实际虚拟环境路径
set "DATA_DIR=!PROJECT_ROOT!\my_biaobei_dataset"
set "PRETRAINED_MODEL_DIR=!PROJECT_ROOT!\pretrained_models\CosyVoice2-0.5B"
set "TRAIN_SCRIPT=!PROJECT_ROOT!\cosyvoice\bin\train.py"
set "CONFIG_FILE=!PROJECT_ROOT!\examples\libritts\cosyvoice2\conf\cosyvoice2.yaml"
set "EXP_MODEL_DIR=!PROJECT_ROOT!\exp\cosyvoice2_biaobei\checkpoints"
set "EXP_LOG_DIR=!PROJECT_ROOT!\exp\cosyvoice2_biaobei\logs"
set "CUDA_VISIBLE_DEVICES=0"
set "NUM_GPUS=1"
set "NUM_WORKERS=0"
set "PREFETCH=100"


:: ======================================
:: ======================================
echo Activating CosyVoice virtual environment...
if not exist "!VENV_ROOT!\Scripts\activate.bat" (
    echo Error: Virtual environment activation script not found! Path: !VENV_ROOT!\Scripts\activate.bat
    echo Please confirm that the .venv environment exists at: D:\cosyvoice\.venv\
    pause
    exit /b 1
)

call "!VENV_ROOT!\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated successfully!
echo.

:: ======================================
:: 第一步：检查所有关键文件/目录是否存在（避免运行报错）
:: ======================================
echo ======================================
echo [Step 1: Check File and Directory Paths]
echo ======================================
if not exist "!PROJECT_ROOT!" (
    echo Error: Project root directory not found! Path: !PROJECT_ROOT!
    pause
    exit /b 1
)

if not exist "!CONFIG_FILE!" (
    echo Error: Configuration file not found! Path: !CONFIG_FILE!
    echo Please confirm that cosyvoice2.yaml exists in examples\libritts\cosyvoice2\conf\
    pause
    exit /b 1
)

if not exist "!BIAOBEI_DATA!" (
    echo Error: Dataset directory not found! Path: !BIAOBEI_DATA!
    pause
    exit /b 1
)

if not exist "!BIAOBEI_DATA!\train\wavs" (
    echo Error: Training audio directory not found! Path: !BIAOBEI_DATA!\train\wavs
    pause
    exit /b 1
)

if not exist "!BIAOBEI_DATA!\train\metadata.csv" (
    echo Error: Training metadata file not found! Path: !BIAOBEI_DATA!\train\metadata.csv
    pause
    exit /b 1
)

if not exist "!PRETRAINED_MODEL!" (
    echo Error: Pretrained model directory not found! Path: !PRETRAINED_MODEL!
    pause
    exit /b 1
)

if not exist "!TRAIN_SCRIPT!" (
    echo Error: Training script not found! Path: !TRAIN_SCRIPT!
    echo Please confirm that train.py exists in cosyvoice\bin\
    pause
    exit /b 1
)

:: 自动创建训练结果保存目录（模型权重+日志）
if not exist "!EXP_DIR!" (
    mkdir "!EXP_DIR!"
    mkdir "!EXP_DIR!\checkpoints"
    mkdir "!EXP_DIR!\logs"
    echo Created experiment directory successfully: !EXP_DIR!
)

:: ======================================
:: 第二步：数据预处理（生成CosyVoice2.0必需的特征文件）
:: ======================================
echo.
echo ======================================
echo [Step 2: Data Preprocessing (Generate Model Input Features)]
echo ======================================

:: 2.1 生成 wav.scp/text/utt2spk 索引文件（已通过Python脚本手动生成，注释跳过）
:: echo Generating audio and text index files...
:: for %%x in (train dev) do (
::     if not exist "!BIAOBEI_DATA!\%%x\wav.scp" (
::         python "!PROJECT_ROOT!\tools\prepare_data.py" ^
::             --src_dir "!BIAOBEI_DATA!\%%x" ^
::             --des_dir "!BIAOBEI_DATA!\%%x" ^
::             --metadata "metadata.csv"
::         if errorlevel 1 (
::             echo Error: Failed to preprocess %%x dataset index files!
::             pause
::             exit /b 1
::         )
::     )
:: )

:: 2.2 提取说话人嵌入特征（模型识别音色必需步骤）
echo Extracting speaker embedding features...
:: 提前检查campplus.onnx文件是否存在
if not exist "!PRETRAINED_MODEL!\campplus.onnx" (
    echo Error: Pretrained model file not found! Path: !PRETRAINED_MODEL!\campplus.onnx
    echo Please confirm that campplus.onnx exists in CosyVoice2-0.5B directory!
    pause
    exit /b 1
)
for %%x in (train dev) do (
    if not exist "!BIAOBEI_DATA!\%%x\spk2embedding.pt" (
        python "!PROJECT_ROOT!\tools\extract_embedding.py" ^
            --dir "!BIAOBEI_DATA!\%%x" ^
            --onnx_path "!PRETRAINED_MODEL!\campplus.onnx"
        if errorlevel 1 (
            echo Error: Failed to extract speaker embedding for %%x dataset!
            pause
            exit /b 1
        )
    )
)

:: 2.3 提取离散语音Token（音频转为模型可识别的语义格式，必需步骤）
echo Extracting discrete speech tokens...
for %%x in (train dev) do (
    if not exist "!BIAOBEI_DATA!\%%x\utt2speech_token.pt" (
        python "!PROJECT_ROOT!\tools\extract_speech_token.py" ^
            --dir "!BIAOBEI_DATA!\%%x" ^
            --onnx_path "!PRETRAINED_MODEL!\speech_tokenizer_v2.onnx"
        if errorlevel 1 (
            echo Error: Failed to extract speech tokens for %%x dataset!
            pause
            exit /b 1
        )
    )
)

:: 2.4 生成Parquet高效训练文件（提升模型加载速度，必需步骤）
echo Generating Parquet format training files...
for %%x in (train dev) do (
    if not exist "!BIAOBEI_DATA!\%%x\parquet" (
        mkdir "!BIAOBEI_DATA!\%%x\parquet"
        python "!PROJECT_ROOT!\tools\make_parquet_list.py" ^
            --num_utts_per_parquet 1000 ^
            --num_processes 10 ^
            --src_dir "!BIAOBEI_DATA!\%%x" ^
            --des_dir "!BIAOBEI_DATA!\%%x\parquet"
        if errorlevel 1 (
            echo Error: Failed to generate Parquet files for %%x dataset!
            pause
            exit /b 1
        )
    )
)

:: 2.5 生成训练/验证集数据列表（模型批量读取数据用）
echo Generating training/validation data lists...
if not exist "!BIAOBEI_DATA!\train.data.list" (
    type "!BIAOBEI_DATA!\train\parquet\data.list" > "!BIAOBEI_DATA!\train.data.list"
)
if not exist "!BIAOBEI_DATA!\dev.data.list" (
    type "!BIAOBEI_DATA!\dev\parquet\data.list" > "!BIAOBEI_DATA!\dev.data.list"
)

:: ======================================
:: 第三步：启动CosyVoice 2.0 标贝女声模型训练
:: ======================================
echo.
echo ======================================
echo [Step 3: Start Model Training (Fully Automated)]
echo ======================================
echo Configuration file path: !CONFIG_FILE!
echo Training dataset: !BIAOBEI_DATA!\train.data.list
echo Validation dataset: !BIAOBEI_DATA!\dev.data.list
echo Model save directory: !EXP_DIR!\checkpoints
echo Training log directory: !EXP_DIR!\logs
echo Available GPU devices: !CUDA_VISIBLE_DEVICES!
echo ======================================
echo Info: Training takes a long time (several days for 10,000 samples on a single GPU), do not close the terminal!
echo Info: Loss values will be printed in real time, a continuous decrease indicates normal training~
echo.

:: 设置GPU环境变量（指定使用的GPU）
set "CUDA_VISIBLE_DEVICES=!CUDA_VISIBLE_DEVICES!"

:: 核心训练命令（调用官方train.py，直接启动训练）
:: 核心训练命令：严格规范续行符 ^，确保参数正确传递
python -u "!TRAIN_SCRIPT!" ^
    --model llm ^
    --config "!CONFIG_FILE!" ^
    --qwen_pretrain_path "D:\cosyvoice\CosyVoice-main\pretrained_models\CosyVoice2-0.5B" ^
    --train_data "!BIAOBEI_DATA!\train.data.list" ^
    --cv_data "!BIAOBEI_DATA!\dev.data.list" ^
    --model_dir "!EXP_DIR!\checkpoints" ^
    --tensorboard_dir "!EXP_DIR!\logs" ^
    --use_amp ^
    --num_workers 0

:: ======================================
:: 第四步：训练结束后处理（提示结果与排查方向）
:: ======================================
echo.
echo ======================================
echo [Step 4: Training Process Completed]
echo ======================================
if errorlevel 0 (
    echo Congratulations! Training completed successfully!
    echo 👉 Final model weights: !EXP_DIR!\checkpoints (best_model.pth is the optimal model)
    echo 👉 Complete training logs: !EXP_DIR!\logs (View loss curves via TensorBoard)
    echo 👉 Configuration file reference: !CONFIG_FILE!
    echo You can directly use best_model.pth for Chinese female voice synthesis testing later~
) else (
    echo Training terminated abnormally! Please check the above terminal error information and troubleshoot in the following directions:
    echo 1. Out of memory: Reduce batch_size in cosyvoice2.yaml (change to 2 or 1)
    echo 2. File encoding: Ensure metadata.csv and yaml configuration files are UTF-8 encoded
    echo 3. Missing dependencies: Re-run pip install -r requirements.txt to install dependencies
    echo 4. Audio format: Ensure audio files in wavs directory are 16000Hz, 16bit, mono .wav files
)

pause