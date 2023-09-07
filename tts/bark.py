
from transformers import AutoProcessor, AutoModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

# inputs = processor(
#     text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
#     return_tensors="pt",
# )

inputs = processor(
    text=["床前明月光，疑是地上霜。举头望明月，低头思故乡。"],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)

print(speech_values)

sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate,
                       data=speech_values.cpu().numpy().squeeze())


# # Use a pipeline as a high-level helper
# from transformers import pipeline
# import numpy as np
# from scipy.io.wavfile import write

# pipe = pipeline("text-to-speech", model="suno/bark-small", device=0)

# result = pipe("Hello, my name is Suno. And, uh — and I like pizza. [laughs]But I also have other interests such as playing tic tac toe.")

# print(result)

# # 保存错误
# # 语音数据数组
# audio_data = result['audio']

# # 采样率
# sampling_rate = result['sampling_rate']

# # 指定保存的文件名和路径
# output_file = 'generated_audio.wav'

# # 将音频数据的振幅缩放到[-1, 1]之间
# audio_data = audio_data / np.max(np.abs(audio_data))

# # 缩放音频数据，确保适合16位整数格式，同时限制振幅在[-1, 1]之间
# scaled_audio = (audio_data * 0.8 * 32767.0).astype(np.int16)

# # 将语音数据保存为.wav文件
# write(output_file, sampling_rate, scaled_audio)
