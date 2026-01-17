import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

# 添加项目根目录到Python路径
sys.path.append('/home/student/work/fangyouying/finaldesign/CosyVoice2')

# 配置路径
TRAIN_DIR = './train'
PARQUET_DIR = os.path.join(TRAIN_DIR, 'parquet')
METADATA_PATH = os.path.join(TRAIN_DIR, 'metadata.csv')
WAVS_DIR = os.path.join(TRAIN_DIR, 'wavs')
SPK2EMBEDDING_PATH = os.path.join(TRAIN_DIR, 'spk2embedding.pt')
UTT2TOKEN_PATH = os.path.join(TRAIN_DIR, 'utt2speech_token.pt')

# 创建Parquet目录
os.makedirs(PARQUET_DIR, exist_ok=True)

# 加载预处理中间文件
print("Loading spk2embedding.pt...")
spk2embedding = torch.load(SPK2EMBEDDING_PATH)
# 适配单说话人特征：提取唯一的特征值
if len(spk2embedding) == 1:
    spk_emb = list(spk2embedding.values())[0]
    # 转换为numpy数组
    if isinstance(spk_emb, torch.Tensor):
        spk_emb_np = spk_emb.numpy()
    elif isinstance(spk_emb, list):
        spk_emb_np = np.array(spk_emb)
    else:
        spk_emb_np = None
    print(f"  - 单说话人特征，维度：{spk_emb_np.shape if spk_emb_np is not None else '未知'}")
else:
    spk_emb_np = None
    print(f"  - 多说话人特征数量：{len(spk2embedding)}")

print("Loading utt2speech_token.pt...")
utt2token = torch.load(UTT2TOKEN_PATH)
print(f"  - 语音Token数量：{len(utt2token)}")

# 读取metadata.csv
print("Reading metadata.csv...")
metadata = []
missing_token = 0
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        utt_id, text = line.split('|', 1)
        # 只检查Token是否存在（说话人特征共用）
        if utt_id not in utt2token:
            missing_token += 1
            continue
        metadata.append({
            'utt_id': utt_id,
            'text': text,
            'wav_path': os.path.join(WAVS_DIR, f'{utt_id}.wav')
        })

print(f"  - 原始样本数：{len(metadata) + missing_token}")
print(f"  - 有效样本数（有Token）：{len(metadata)}")
print(f"  - 缺失Token的样本数：{missing_token}")

# 按每1000条样本生成一个Parquet文件
batch_size = 1000
num_batches = (len(metadata) + batch_size - 1) // batch_size

print(f"\nGenerating {num_batches} Parquet files...")
generated_files = []
for batch_idx in tqdm(range(num_batches), desc='Generating Parquet files'):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(metadata))
    batch_data = metadata[start:end]
    
    # 补充特征：所有样本共用单说话人特征 + 各自的Token
    for item in batch_data:
        utt_id = item['utt_id']
        # 共用说话人特征
        item['spk_embedding'] = spk_emb_np
        # 各自的语音Token
        token = utt2token[utt_id]
        if isinstance(token, torch.Tensor):
            item['speech_token'] = token.numpy()
        elif isinstance(token, list):
            item['speech_token'] = np.array(token)
        else:
            item['speech_token'] = None
    
    # 生成Parquet文件
    df = pd.DataFrame(batch_data)
    parquet_path = os.path.join(PARQUET_DIR, f'data_{batch_idx}.parquet')
    df.to_parquet(parquet_path, index=False)
    generated_files.append(parquet_path)

# 生成train.data.list
print("\nGenerating train.data.list...")
with open('train.data.list', 'w', encoding='utf-8') as f:
    for file_path in generated_files:
        rel_path = os.path.relpath(file_path, '.')
        f.write(f'{rel_path}\n')

# 最终验证
num_parquet = len(generated_files)
num_samples = sum([len(pd.read_parquet(f)) for f in generated_files])

print(f'\n✅ Parquet文件生成完成！')
print(f'   - Parquet目录：{PARQUET_DIR}')
print(f'   - 生成文件数：{num_parquet}个')
print(f'   - 有效样本数：{num_samples}条')
print(f'   - 数据列表：train.data.list (共{num_parquet}行)')
print(f'\n📌 验证命令：')
print(f'   ls -l {PARQUET_DIR} | head -5')
print(f'   cat train.data.list | head -3')