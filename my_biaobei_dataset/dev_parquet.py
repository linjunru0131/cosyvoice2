import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

sys.path.append('/home/student/work/fangyouying/finaldesign/CosyVoice2')

# 配置dev路径
DEV_DIR = './dev'
PARQUET_DIR = os.path.join(DEV_DIR, 'parquet')
METADATA_PATH = os.path.join(DEV_DIR, 'metadata.csv')
WAVS_DIR = os.path.join(DEV_DIR, 'wavs')
SPK2EMBEDDING_PATH = os.path.join(DEV_DIR, 'spk2embedding.pt')
UTT2TOKEN_PATH = os.path.join(DEV_DIR, 'utt2speech_token.pt')

os.makedirs(PARQUET_DIR, exist_ok=True)

# 加载单说话人特征
print("Loading spk2embedding.pt (dev)...")
spk2embedding = torch.load(SPK2EMBEDDING_PATH)
spk_emb = list(spk2embedding.values())[0]
if isinstance(spk_emb, torch.Tensor):
    spk_emb_np = spk_emb.numpy()
else:
    spk_emb_np = np.array(spk_emb)
print(f"  - 单说话人特征维度：{spk_emb_np.shape}")

# 加载Token
print("Loading utt2speech_token.pt (dev)...")
utt2token = torch.load(UTT2TOKEN_PATH)
print(f"  - 语音Token数量：{len(utt2token)}")

# 读取dev的metadata
print("Reading dev metadata.csv...")
metadata = []
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        utt_id, text = line.split('|', 1)
        if utt_id not in utt2token:
            continue
        metadata.append({
            'utt_id': utt_id,
            'text': text,
            'wav_path': os.path.join(WAVS_DIR, f'{utt_id}.wav')
        })
print(f"  - dev有效样本数：{len(metadata)}")

# 生成Parquet（每1000条一个文件）
batch_size = 1000
num_batches = (len(metadata) + batch_size - 1) // batch_size
generated_files = []
for batch_idx in tqdm(range(num_batches), desc='Generating dev Parquet'):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(metadata))
    batch_data = metadata[start:end]
    
    for item in batch_data:
        item['spk_embedding'] = spk_emb_np
        token = utt2token[item['utt_id']]
        item['speech_token'] = token.numpy() if isinstance(token, torch.Tensor) else np.array(token)
    
    df = pd.DataFrame(batch_data)
    parquet_path = os.path.join(PARQUET_DIR, f'data_{batch_idx}.parquet')
    df.to_parquet(parquet_path, index=False)
    generated_files.append(parquet_path)

# 生成dev.data.list
with open('../dev.data.list', 'w', encoding='utf-8') as f:
    for file_path in generated_files:
        f.write(f'{os.path.relpath(file_path, "..")}\n')

print(f"\n✅ dev Parquet生成完成！")
print(f"   - 文件数：{len(generated_files)}个")
print(f"   - 样本数：{len(metadata)}条")
print(f"   - 数据列表：../dev.data.list")