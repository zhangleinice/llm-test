from transformers import pipeline

transcriber = pipeline(model="openai/whisper-medium", device=0)

result = transcriber("./data/podcast_clip.mp3")

print(result)
