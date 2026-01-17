@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion


set "PRETRAIN_DIR=pretrained_models"
set "PYTHON_SCRIPT=download_cosyvoice2_temp.py"

:: 1. 先安装核心依赖（modelscope，用于下载预训练模型）
echo 正在安装核心依赖：modelscope...
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
if errorlevel 1 (
    echo 错误：安装 modelscope 失败，请检查 Python 和 pip 是否配置正常！
    pause
    exit /b 1
)

:: 2. 创建预训练模型存放目录
if not exist "%PRETRAIN_DIR%" mkdir "%PRETRAIN_DIR%"
echo 正在创建预训练模型目录：%PRETRAIN_DIR%

:: 3. 生成临时 Python 下载脚本（专注于 CosyVoice 2.0 模型）
echo 正在生成 CosyVoice 2.0 下载脚本...
echo from modelscope import snapshot_download > "%PYTHON_SCRIPT%"
echo import os >> "%PYTHON_SCRIPT%"
echo. >> "%PYTHON_SCRIPT%"
echo # 配置下载路径 >> "%PYTHON_SCRIPT%"
echo pretrain_dir = "%PRETRAIN_DIR%" >> "%PYTHON_SCRIPT%"
echo if not os.path.exists(pretrain_dir): >> "%PYTHON_SCRIPT%"
echo     os.makedirs(pretrain_dir) >> "%PYTHON_SCRIPT%"
echo. >> "%PYTHON_SCRIPT%"
echo # 下载 CosyVoice 2.0 核心预训练模型和资源 >> "%PYTHON_SCRIPT%"
echo print("正在下载 CosyVoice2-0.5B（核心模型）...") >> "%PYTHON_SCRIPT%"
echo snapshot_download('iic/CosyVoice2-0.5B', local_dir=os.path.join(pretrain_dir, 'CosyVoice2-0.5B')) >> "%PYTHON_SCRIPT%"
echo print("正在下载 CosyVoice-ttsfrd（文本处理资源）...") >> "%PYTHON_SCRIPT%"
echo snapshot_download('iic/CosyVoice-ttsfrd', local_dir=os.path.join(pretrain_dir, 'CosyVoice-ttsfrd')) >> "%PYTHON_SCRIPT%"
echo print("CosyVoice 2.0 预训练模型下载完成！") >> "%PYTHON_SCRIPT%"

:: 4. 直接执行 Python 下载脚本（无需 Conda）
echo 正在下载 CosyVoice 2.0 预训练模型（网络较慢，耐心等待）...
python "%PYTHON_SCRIPT%"
if errorlevel 1 (
    echo 错误：下载预训练模型失败，请检查网络或 modelscope 安装是否完整！
    pause
    exit /b 1
)

:: 5. 清理临时文件
if exist "%PYTHON_SCRIPT%" del "%PYTHON_SCRIPT%"
echo 临时文件清理完成！

echo.
echo ======================================
echo 下载完成！预训练模型保存到：%PRETRAIN_DIR%
echo 包含：CosyVoice2-0.5B（核心模型）、CosyVoice-ttsfrd（文本处理）
echo ======================================
pause