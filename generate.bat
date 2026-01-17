@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: ===================== 仅需修改这 2 项（必改！）=====================
:: 1. 你的标贝合并式标注文件路径（000001-010000.txt）
set "BIAOBEI_MERGED_TXT=D:\cosyvoice\ProsodyLabeling\000001-010000.txt"
:: 2. 你的标贝数据集根目录（my_biaobei_dataset，已在根目录）
set "WAV_ROOT=D:\cosyvoice\CosyVoice-main\my_biaobei_dataset"
:: =====================================================================

:: 第一步：检查文件是否存在
if not exist "%BIAOBEI_MERGED_TXT%" (
    echo 错误：未找到合并标注文件！路径：%BIAOBEI_MERGED_TXT%
    pause
    exit /b 1
)
if not exist "%WAV_ROOT%\train\wavs" (
    echo 错误：未找到训练集音频目录！路径：%WAV_ROOT%\train\wavs
    pause
    exit /b 1
)
if not exist "%WAV_ROOT%\dev\wavs" (
    echo 错误：未找到验证集音频目录！路径：%WAV_ROOT%\dev\wavs
    pause
    exit /b 1
)

:: 第二步：生成训练集 metadata.csv
echo 正在生成训练集标注文件...
:: 修复1：del 命令添加 /f 强制删除，避免只读文件/路径解析问题
if exist "%WAV_ROOT%\train\metadata.csv" del /f "%WAV_ROOT%\train\metadata.csv"

:: 遍历训练集所有 .wav 音频文件
for %%f in ("%WAV_ROOT%\train\wavs\*.wav") do (
    set "WAV_NAME=%%~nf"  :: 获取音频文件名（如 000001，无后缀）
    set "TARGET_TEXT="   :: 初始化存储中文文本的变量

    :: 修复2：嵌套 findstr 命令用 ^" 转义引号，正确解析特殊字符路径
    :: 修复3：确保 delims= 后是 Tab 键（手动在编辑器中按 Tab 输入，不要用空格）
    for /f "delims=	tokens=1*" %%a in ('findstr /b ^"!WAV_NAME!	^" "%BIAOBEI_MERGED_TXT%"') do (
        set "TARGET_TEXT=%%b"  :: 提取中文文本（包含#韵律标注，保留不删除）
    )

    :: 若找到对应文本，写入 metadata.csv（格式：音频名|中文文本）
    :: 修复4：echo 写入时，确保路径引号完整，避免管道符 ^| 解析错误
    if not "!TARGET_TEXT!"=="" (
        echo !WAV_NAME!^|!TARGET_TEXT! >> "%WAV_ROOT%\train\metadata.csv"
    ) else (
        echo 警告：未找到音频 !WAV_NAME! 对应的中文标注
    )
)

:: 第三步：生成验证集 metadata.csv
echo 正在生成验证集标注文件...
:: 修复1：del 命令添加 /f 强制删除
if exist "%WAV_ROOT%\dev\metadata.csv" del /f "%WAV_ROOT%\dev\metadata.csv"

:: 遍历验证集所有 .wav 音频文件
for %%f in ("%WAV_ROOT%\dev\wavs\*.wav") do (
    set "WAV_NAME=%%~nf"
    set "TARGET_TEXT="

    :: 修复2+3：嵌套命令转义引号 + 保留 Tab 分隔符
    for /f "delims=	tokens=1*" %%a in ('findstr /b ^"!WAV_NAME!	^" "%BIAOBEI_MERGED_TXT%"') do (
        set "TARGET_TEXT=%%b"
    )

    if not "!TARGET_TEXT!"=="" (
        echo !WAV_NAME!^|!TARGET_TEXT! >> "%WAV_ROOT%\dev\metadata.csv"
    ) else (
        echo 警告：未找到音频 !WAV_NAME! 对应的中文标注
    )
)

:: 第四步：验证生成结果
echo.
echo ======================================
echo 标注文件生成完成！
echo 训练集标注：%WAV_ROOT%\train\metadata.csv
echo 验证集标注：%WAV_ROOT%\dev\metadata.csv
echo ======================================
pause