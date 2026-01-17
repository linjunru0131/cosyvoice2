@echo off
:: 第一步：清空终端、切换UTF-8编码，避免乱码
cls
chcp 65001 > nul

:: 第二步：启用延迟变量扩展，解决变量未解析问题（必须在变量定义前）
setlocal enabledelayedexpansion

:: 第三步：设置核心环境变量，语法正确，无多余字符，避免换行/特殊字符干扰
set "PYTHONIOENCODING=utf-8"
set "ENABLE_LONG_PATH=1"

:: 输出配置提示
echo ======================================
echo [Info] 配置项目路径，请确认路径正确性
echo ======================================

:: 定义核心路径变量（优化语法，避免解析错误）
set "PROJECT_ROOT=D:\CosyVoice2"
set "VENV_ROOT=!PROJECT_ROOT!\venv"
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

:: 定义.list文件和parquet目录路径（自动生成相关）
set "TRAIN_LIST_FILE=!DATA_DIR!\train.data.list"
set "DEV_LIST_FILE=!DATA_DIR!\dev.data.list"
set "TRAIN_PARQUET_DIR=!DATA_DIR!\train\parquet"
set "DEV_PARQUET_DIR=!DATA_DIR!\dev\parquet"

:: 输出变量验证（优化格式，避免语法错误）
echo 项目根目录：!PROJECT_ROOT!
echo 数据集目录：!DATA_DIR!
echo 预训练模型：!PRETRAINED_MODEL_DIR!
echo 配置文件：!CONFIG_FILE!
echo 训练集parquet目录：!TRAIN_PARQUET_DIR!
echo 验证集parquet目录：!DEV_PARQUET_DIR!
echo.

:: ======================================
:: 新增步骤：自动生成 train.data.list 和 dev.data.list
:: 核心逻辑：1. 检查parquet目录 2. 遍历所有.tar(parquet)文件 3. 写入.list文件
:: ======================================
echo ======================================
echo [New Step] 自动生成 train.data.list 和 dev.data.list
echo ======================================

:: -------------- 生成 train.data.list --------------
echo [Info] 开始处理训练集，生成 !TRAIN_LIST_FILE!
if exist "!TRAIN_PARQUET_DIR!" (
    :: 先删除旧的train.data.list（避免重复内容）- 注释单独成行，避免命令与中文混排
    if exist "!TRAIN_LIST_FILE!" (
        :: 删除旧文件，屏蔽输出（> nul），注释单独写在上方
        del "!TRAIN_LIST_FILE!" > nul
        echo [Info] 删除旧的训练集列表文件：!TRAIN_LIST_FILE!
    )

    :: 遍历parquet目录下的所有.tar文件（兼容parquet_*.tar格式）
    :: /r 递归遍历 /b 仅输出文件名 /a-d 排除目录
    for /r "!TRAIN_PARQUET_DIR!" %%f in (*.tar) do (
        :: 将完整路径写入train.data.list（每行一个路径）
        echo %%~f >> "!TRAIN_LIST_FILE!"
    )

    :: 验证是否生成成功且非空
    if exist "!TRAIN_LIST_FILE!" (
        for /f %%s in ("!TRAIN_LIST_FILE!") do set "FILE_SIZE=%%~zs"
        if !FILE_SIZE! gtr 0 (
            echo [Success] 训练集列表文件生成成功：!TRAIN_LIST_FILE!
            echo [Info] 包含有效路径数据，文件大小：!FILE_SIZE! 字节
        ) else (
            echo [Warning] 训练集列表文件生成成功，但为空！请检查 !TRAIN_PARQUET_DIR! 下是否有.tar文件
        )
    ) else (
        echo [Error] 训练集列表文件生成失败：!TRAIN_LIST_FILE!
        pause
        exit /b 1
    )
) else (
    echo [Error] 训练集parquet目录不存在：!TRAIN_PARQUET_DIR!
    echo 请确认目录结构为：!DATA_DIR!\train\parquet
    pause
    exit /b 1
)

:: -------------- 生成 dev.data.list --------------
echo.
echo [Info] 开始处理验证集，生成 !DEV_LIST_FILE!
if exist "!DEV_PARQUET_DIR!" (
    :: 先删除旧的dev.data.list（避免重复内容）- 注释单独成行
    if exist "!DEV_LIST_FILE!" (
        del "!DEV_LIST_FILE!" > nul
        echo [Info] 删除旧的验证集列表文件：!DEV_LIST_FILE!
    )

    :: 遍历parquet目录下的所有.tar文件
    for /r "!DEV_PARQUET_DIR!" %%f in (*.tar) do (
        echo %%~f >> "!DEV_LIST_FILE!"
    )

    :: 验证是否生成成功且非空
    if exist "!DEV_LIST_FILE!" (
        for /f %%s in ("!DEV_LIST_FILE!") do set "FILE_SIZE=%%~zs"
        if !FILE_SIZE! gtr 0 (
            echo [Success] 验证集列表文件生成成功：!DEV_LIST_FILE!
            echo [Info] 包含有效路径数据，文件大小：!FILE_SIZE! 字节
        ) else (
            echo [Warning] 验证集列表文件生成成功，但为空！请检查 !DEV_PARQUET_DIR! 下是否有.tar文件
        )
    ) else (
        echo [Error] 验证集列表文件生成失败：!DEV_LIST_FILE!
        pause
        exit /b 1
    )
) else (
    echo [Error] 验证集parquet目录不存在：!DEV_PARQUET_DIR!
    echo 请确认目录结构为：!DATA_DIR!\dev\parquet
    pause
    exit /b 1
)
echo.

:: ======================================
:: 原有步骤1：检查虚拟环境和核心文件/目录是否存在
:: ======================================
echo ======================================
echo [Step 1] 检查关键文件和环境
echo ======================================
:: 检查虚拟环境激活脚本
if exist "!VENV_ROOT!\Scripts\activate.bat" (
    echo [Success] 虚拟环境激活脚本存在，准备激活环境
) else (
    echo [Error] 虚拟环境激活脚本不存在：!VENV_ROOT!\Scripts\activate.bat
    echo 请确认虚拟环境目录为：!VENV_ROOT!
    pause
    exit /b 1
)

:: 检查训练脚本
if exist "!TRAIN_SCRIPT!" (
    echo [Success] 训练脚本存在：!TRAIN_SCRIPT!
) else (
    echo [Error] 训练脚本不存在：!TRAIN_SCRIPT!
    pause
    exit /b 1
)

:: 检查配置文件
if exist "!CONFIG_FILE!" (
    echo [Success] 配置文件存在：!CONFIG_FILE!
) else (
    echo [Error] 配置文件不存在：!CONFIG_FILE!
    pause
    exit /b 1
)

:: 检查数据集目录
if exist "!DATA_DIR!" (
    echo [Success] 数据集目录存在：!DATA_DIR!
) else (
    echo [Error] 数据集目录不存在：!DATA_DIR!
    pause
    exit /b 1
)

:: 检查预训练模型目录
if exist "!PRETRAINED_MODEL_DIR!" (
    echo [Success] 预训练模型目录存在：!PRETRAINED_MODEL_DIR!
) else (
    echo [Error] 预训练模型目录不存在：!PRETRAINED_MODEL_DIR!
    pause
    exit /b 1
)

:: 自动创建实验结果目录（模型权重+日志）
if not exist "!EXP_MODEL_DIR!" (
    mkdir "!EXP_MODEL_DIR!"
    echo [Info] 自动创建模型保存目录：!EXP_MODEL_DIR!
)
if not exist "!EXP_LOG_DIR!" (
    mkdir "!EXP_LOG_DIR!"
    echo [Info] 自动创建日志保存目录：!EXP_LOG_DIR!
)
echo.

:: ======================================
:: 原有步骤2：激活虚拟环境
:: ======================================
echo ======================================
echo [Step 2] 激活虚拟环境
echo ======================================
if not exist "!VENV_ROOT!\Scripts\activate.bat" (
    echo [Error] 虚拟环境激活脚本不存在：!VENV_ROOT!\Scripts\activate.bat
    pause
    exit /b 1
)
echo [Info] 正在激活虚拟环境...
call "!VENV_ROOT!\Scripts\activate.bat"
echo [Success] 虚拟环境激活成功
echo.

:: ======================================
:: 原有步骤3：核心训练逻辑（对应run.sh的stage=5）
:: ======================================
echo ======================================
echo [Step 3] 启动CosyVoice模型训练（LLM/Flow/HifiGAN）
echo ======================================
echo [Info] GPU设备：!CUDA_VISIBLE_DEVICES!
echo [Info] GPU数量：!NUM_GPUS!
echo [Info] 模型保存目录：!EXP_MODEL_DIR!
echo [Info] 日志保存目录：!EXP_LOG_DIR!
echo [Info] 训练开始，请勿关闭终端（训练时间较长，视数据集大小而定）
echo ======================================
echo.

:: 循环训练3个模型（llm/flow/hifigan）
for %%m in (llm flow hifigan) do (
    echo ======================================
    echo [Training] 开始训练模型：%%m
    echo ======================================

    :: 核心训练命令（适配Windows单GPU）
    python -u "!TRAIN_SCRIPT!" ^
        --train_engine torch_ddp ^
        --config "!CONFIG_FILE!" ^
        --train_data "!TRAIN_LIST_FILE!" ^
        --cv_data "!DEV_LIST_FILE!" ^
        --qwen_pretrain_path "D:\CosyVoice2\pretrained_models\Qwen2-0.5B" ^
        --model %%m ^
        --checkpoint "!PRETRAINED_MODEL_DIR!\%%m.pt" ^
        --model_dir "!EXP_MODEL_DIR!" ^
        --tensorboard_dir "!EXP_LOG_DIR!" ^
        --ddp.dist_backend gloo ^
        --num_workers !NUM_WORKERS! ^
        --prefetch !PREFETCH! ^
        --pin_memory ^
        --use_amp

    :: 检查单个模型训练是否成功
    if errorlevel 1 (
        echo [Error] 模型%%m训练失败
        pause
        exit /b 1
    )
    echo [Success] 模型%%m训练完成
    echo.
)

:: ======================================
:: 原有步骤4：训练结束后处理
:: ======================================
echo ======================================
echo [Step 4] 训练流程全部完成
echo ======================================
echo [Success] 所有模型（llm/flow/hifigan）训练完成
echo [Info] 最终模型权重：!EXP_MODEL_DIR!
echo [Info] 完整训练日志：!EXP_LOG_DIR!
echo [Info] 自动生成的数据集列表文件：
echo [Info]  - 训练集：!TRAIN_LIST_FILE!
echo [Info]  - 验证集：!DEV_LIST_FILE!
echo [Info] 可通过TensorBoard查看损失曲线：tensorboard --logdir=!EXP_LOG_DIR!
echo.

pause
exit /b 0