# from transformers import pipeline

# transcriber = pipeline(
#     model="meta-llama/Llama-2-7b-chat-hf",
#     device=0
# )

# result = transcriber("你好")

# print(result)

from transformers import AutoTokenizer, AutoModel

model_id = "meta-llama/Llama-2-7b-chat-hf"

hf_auth = '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'

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

model = AutoModel.from_pretrained(
    model_id,
    token=hf_auth
    # use_auth_token=hf_auth
).half().cuda()

model = model.eval()

input = tokenizer("今天你吃了吗？")

print('input', input)

promopt = "今天你吃了吗"

res = model.chat(tokenizer, promopt, history=[])

print('res', res)
