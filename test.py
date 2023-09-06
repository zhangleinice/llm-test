# from transformers import pipeline

# transcriber = pipeline(model="openai/whisper-medium", device=-1)
# result = transcriber("./data/podcast_clip.mp3")
# print(result)

from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="zh", task="transcribe")

transcriber = pipeline(model="openai/whisper-medium", device=-1,
                       generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
result = transcriber("./data/podcast_clip.mp3")
print(result)

# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# # 模型下载到本地
# model_path = "./path/to/your/model"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)

# transcriber = pipeline(model=model, tokenizer=tokenizer, device=0)
# result = transcriber("./data/podcast_clip.mp3")
# print(result)
