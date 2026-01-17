import os
import pandas as pd


biaobei_root = "D:/cosyvoice/CosyVoice-main/my_biaobei_dataset"
speaker_id = "spk001"  # 标贝是单一女声，统一用这个说话人ID


def generate_index_files(data_split):
    """
    生成指定数据集（train/dev）的 wav.scp/text/utt2spk 文件
    data_split: "train" 或 "dev"
    """
    # 1. 读取 metadata.csv
    metadata_path = os.path.join(biaobei_root, data_split, "metadata.csv")
    df = pd.read_csv(
        metadata_path,
        sep="|",
        header=None,
        names=["utt_id", "text"],
        dtype=str,
        na_filter=False
    )

    # 2. 定义输出文件路径
    output_dir = os.path.join(biaobei_root, data_split)
    wav_scp_path = os.path.join(output_dir, "wav.scp")
    text_path = os.path.join(output_dir, "text")
    utt2spk_path = os.path.join(output_dir, "utt2spk")

    # 3. 批量生成内容
    with open(wav_scp_path, "w", encoding="utf-8") as f_wav, \
            open(text_path, "w", encoding="utf-8") as f_text, \
            open(utt2spk_path, "w", encoding="utf-8") as f_utt2spk:

        for _, row in df.iterrows():
            utt_id = str(row["utt_id"]).strip()
            text = str(row["text"]).strip()

            if not utt_id or not text:
                continue

            # 拼接音频绝对路径
            wav_path = os.path.join(output_dir, "wavs", f"{utt_id}.wav")
            wav_path = wav_path.replace("/", "\\")

            # 写入 wav.scp（音频ID 对应 音频绝对路径）
            f_wav.write(f"{utt_id} {wav_path}\n")
            # 写入 text（音频ID 对应 文本内容）
            f_text.write(f"{utt_id} {text}\n")
            # 写入 utt2spk（音频ID 对应 说话人ID）
            f_utt2spk.write(f"{utt_id} {speaker_id}\n")

    print(f"✅ 已生成 {data_split} 数据集的索引文件：")
    print(f"  - {wav_scp_path}")
    print(f"  - {text_path}")
    print(f"  - {utt2spk_path}")


if __name__ == "__main__":
    # 生成训练集和验证集的索引文件
    generate_index_files("train")
    generate_index_files("dev")
    print("\n🎉 所有索引文件生成完成！")