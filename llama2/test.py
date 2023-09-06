from transformers import pipeline

transcriber = pipeline(
    model="meta-llama/Llama-2-7b-chat-hf",
    device=0
)

result = transcriber("你好")

print(result)
