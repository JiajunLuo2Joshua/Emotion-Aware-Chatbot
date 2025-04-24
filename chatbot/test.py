from TTS.api import TTS
import os
import platform

# 下载并加载模型（首次自动缓存，后续离线可用）
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# 输出文件
output_path = "output.wav"
tts.tts_to_file(text="Hi, I am your gentle companion robot.", file_path=output_path)

os.system(f'start {output_path}')

