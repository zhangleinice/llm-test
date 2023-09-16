# 需要升级python3.10以上

# import torch
# torch.cuda.empty_cache()

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch

# 在运行模型之前释放GPU内存（没用）
# torch.cuda.empty_cache()

model_id = "meta-llama/Llama-2-7b-hf"

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    model_kwargs={"max_length": 64},
    device=0,
    batch_size=4
)

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

question_chain = LLMChain(llm=llm, prompt=prompt)

question = "What is electroencephalography?"

res = question_chain.run(question)

print('res\n', res)
