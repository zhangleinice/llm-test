# Use a pipeline as a high-level helper

from transformers import pipeline
import torch

model_id = "meta-llama/Llama-2-7b-hf"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

prompt = "今天你吃了吗，用中文回答我"

res = pipe(
    prompt,
    do_sample=True,
    num_return_sequences=1,
    max_length=200,
)

print('res', res)

# res [{'generated_text': '今天你吃了吗，用中文回答我。\n1. 你吃了什么？\n2. 你吃的好吗？\n3. 你吃了多少？\n4. 你吃了多少钱？\n5. 你吃了多少时间？\n6. 你吃了多少玩儿？\n7. 你吃了多少吗？\n8. 你吃了多少人？\n9. 你吃了多少酒？\n10. 你吃了多少瓜？\n11. 你吃了多少辣椒？\n12. 你吃了多少��'}]


