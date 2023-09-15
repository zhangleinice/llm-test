# Use a pipeline as a high-level helper

from transformers import pipeline

model_id = "meta-llama/Llama-2-7b-hf"

pipe = pipeline(
    "text-generation",
    model=model_id,
)

prompt = "今天你吃了吗，用中文回答我"

res = pipe(prompt)

print(12233)

print("res", res)

print('res1', res[0])
