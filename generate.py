# 保存为 generate_metadata.py
import os

BIAOBEI_MERGED_TXT = r"D:\cosyvoice\ProsodyLabeling\000001-010000.txt"
WAV_ROOT = r"D:\cosyvoice\CosyVoice-main\my_biaobei_dataset"


# =====================================================================

def generate_metadata(wav_dir, output_csv):
    """生成单个目录的 metadata.csv"""
    # 读取标注文件，构建「音频名-文本」映射字典
    label_dict = {}
    with open(BIAOBEI_MERGED_TXT, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按 Tab 键分割（兼容批处理的标注文件格式）
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            wav_name, text = parts
            label_dict[wav_name] = text

    # 遍历 WAV 文件，生成 CSV
    with open(output_csv, 'w', encoding='utf-8') as csv_f:
        for wav_file in os.listdir(wav_dir):
            if not wav_file.endswith('.wav'):
                continue
            wav_name = os.path.splitext(wav_file)[0]
            # 查找对应标注
            if wav_name in label_dict:
                text = label_dict[wav_name]
                csv_f.write(f"{wav_name}|{text}\n")
            else:
                print(f"警告：未找到音频 {wav_name} 对应的中文标注")


if __name__ == "__main__":
    # 检查路径是否存在
    if not os.path.exists(BIAOBEI_MERGED_TXT):
        print(f"错误：未找到合并标注文件！路径：{BIAOBEI_MERGED_TXT}")
        exit(1)

    train_wav_dir = os.path.join(WAV_ROOT, "train", "wavs")
    dev_wav_dir = os.path.join(WAV_ROOT, "dev", "wavs")
    if not os.path.exists(train_wav_dir):
        print(f"错误：未找到训练集音频目录！路径：{train_wav_dir}")
        exit(1)
    if not os.path.exists(dev_wav_dir):
        print(f"错误：未找到验证集音频目录！路径：{dev_wav_dir}")
        exit(1)

    # 生成训练集和验证集标注
    print("正在生成训练集标注文件...")
    train_csv = os.path.join(WAV_ROOT, "train", "metadata.csv")
    generate_metadata(train_wav_dir, train_csv)

    print("正在生成验证集标注文件...")
    dev_csv = os.path.join(WAV_ROOT, "dev", "metadata.csv")
    generate_metadata(dev_wav_dir, dev_csv)

    # 验证结果
    print("\n======================================")
    print("标注文件生成完成！")
    print(f"训练集标注：{train_csv}")
    print(f"验证集标注：{dev_csv}")
    print("======================================")