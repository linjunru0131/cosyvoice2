# from modelscope import snapshot_download
# snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
# snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')


from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id='Qwen/Qwen2-0.5B',
    cache_dir=r'D:\CosyVoice2\pretrained_models'
)

print("模型下载完成，路径是：", model_dir)
