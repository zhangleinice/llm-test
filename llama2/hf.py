
# langchain连接huggingface开源模型

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

llm = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

question_chain = LLMChain(llm=llm, prompt=prompt)

question = "What is electroencephalography?"

res = question_chain.run(question)

print('res\n', res)
#   First, we need to understand what is the brain. The brain is a complex structure that is made up of neurons. Neurons are the basic building blocks of the brain. Each neuron is made up of a cell body and
