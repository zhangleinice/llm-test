

from transformers import AutoTokenizer, LlamaForCausalLM

# model_id = "meta-llama/Llama-2-7b-chat-hf"

model_id = "meta-llama/Llama-2-7b-hf"

hf_auth = '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'


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

model = LlamaForCausalLM.from_pretrained(
    model_id,
    token=hf_auth
)

model = model.eval()

prompt = "今天你吃了吗，用中文回答我"

input = tokenizer(prompt, return_tensors="pt")

res = model.generate(input_ids=input.input_ids, max_length=30)

# 将生成的输出转换为文本
generated_text = tokenizer.decode(res[0], skip_special_tokens=True)

print('input', input)

print('res', res)

print('generated_text', generated_text)


# input {'input_ids': tensor([[    1, 29871, 31482, 30408, 30919,   232,   147,   134, 30743,   232,
#            147,   154, 30214, 30406, 30275, 30333, 30742,   234,   176,   151,
#          30672]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
# res tensor([[    1, 29871, 31482, 30408, 30919,   232,   147,   134, 30743,   232,
#            147,   154, 30214, 30406, 30275, 30333, 30742,   234,   176,   151,
#          30672,    13,   233,   155,   171, 30408, 30919,   232,   147,   134]])
# generated_text 今天你吃了吗，用中文回答我
# 昨天你吃
