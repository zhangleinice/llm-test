from transformers import pipeline, AutoTokenizer

# model_id = "meta-llama/Llama-2-7b-chat-hf"

model_id = "meta-llama/Llama-2-7b-hf"

# hf_auth = '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'

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    # token=hf_auth
)

prompt = "今天你吃了吗，用中文回答我"

res = pipe(prompt)

print(12233)

print("res", res)

print('res1', res[0])