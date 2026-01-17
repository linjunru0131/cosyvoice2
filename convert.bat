@echo off
chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion  :: 启用延迟变量扩展，保证 COUNT 正常累加

:: 1. 创建数据集目录结构（标准 CosyVoice 2.0 目录）
mkdir D:\cosyvoice\CosyVoice-main\my_biaobei_dataset\train\wavs 2> nul
mkdir D:\cosyvoice\CosyVoice-main\my_biaobei_dataset\dev\wavs 2> nul

:: 2. 批量转码（适配 CosyVoice 2.0 要求：24000Hz 单声道 16bit PCM）
set "BIAOBEI_WAVS_PATH=D:\cosyvoice\Wave"
set "OUTPUT_TRAIN_PATH=D:\cosyvoice\CosyVoice-main\my_biaobei_dataset\train\wavs"

:: 遍历所有 .wav 文件转码（核心修改：-ar 24000，其他参数保留 CosyVoice 2.0 要求）
for %%f in ("%BIAOBEI_WAVS_PATH%\*.wav") do (
    set "WAV_NAME=%%~nf"
    ffmpeg -i "%%f" -ar 24000 -ac 1 -sample_fmt s16 -c:a pcm_s16le "%OUTPUT_TRAIN_PATH%\%%~nf.wav" -y
)

:: 3. 抽取 1000 条到验证集（路径拼接正确，无多余文件夹）
dir /b "%OUTPUT_TRAIN_PATH%\*.wav" > temp_wav_list.txt

set "COUNT=0"
for /f "tokens=*" %%w in (temp_wav_list.txt) do (
    if !COUNT! lss 1000 (
        :: 路径拼接正确：wavs 后加反斜杠，避免创建异常文件夹
        move "%OUTPUT_TRAIN_PATH%\%%w" "D:\cosyvoice\CosyVoice-main\my_biaobei_dataset\dev\wavs\%%w" > nul 2>&1
        set /a COUNT+=1
    ) else (
        goto :end_extract
    )
)
:end_extract

:: 4. 清理临时文件
del temp_wav_list.txt > nul 2>&1

echo 转码完成！（24000Hz 单声道 16bit，符合 CosyVoice 2.0 要求）
echo 数据集已保存到 my_biaobei_dataset 目录
pause