from transformers import pipeline

transcriber = pipeline(model="openai/whisper-medium", device=0)

result = transcriber("./podcast_clip.mp3")

print(result)
