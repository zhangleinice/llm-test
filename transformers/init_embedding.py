import torch
from transformers import AutoTokenizer, AutoModel

# 选择预训练模型
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# 添加特殊标记到分词器
num_added_toks = tokenizer.add_tokens(
    ['[ENT_START]', '[ENT_END]'], special_tokens=True)

# 描述要添加的特殊标记
descriptions = ['start of entity', 'end of entity']

with torch.no_grad():
    # 遍历并添加特殊标记
    for i, token in enumerate(reversed(descriptions), start=1):
        # 对描述进行分词
        tokenized = tokenizer.tokenize(token)
        print(tokenized)
        # 将分词转换为对应的ID
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
        # 计算新的嵌入向量，取平均值
        new_embedding = model.embeddings.word_embeddings.weight[tokenized_ids].mean(
            axis=0)
        # 将新的嵌入向量应用到模型的词嵌入权重中
        model.embeddings.word_embeddings.weight[-i,
                                                :] = new_embedding.clone().detach().requires_grad_(True)

# 打印添加后的特殊标记的嵌入向量
print(model.embeddings.word_embeddings.weight[-2:, :])


# ['end', 'of', 'entity']
# ['start', 'of', 'entity']
# tensor([[-0.0340, -0.0144, -0.0441,  ..., -0.0016,  0.0318, -0.0151],
#         [-0.0060, -0.0202, -0.0312,  ..., -0.0084,  0.0193, -0.0296]],
#        grad_fn=<SliceBackward0>)
