from transformers import AutoTokenizer, AutoModel

model_id = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModel.from_pretrained(model_id)

print('vocabulary size:', len(tokenizer))

num_added_toks = tokenizer.add_tokens(
    ['[ENT_START]', '[ENT_END]'], special_tokens=True)

print("After we add", num_added_toks, "tokens")

print('vocabulary size:', len(tokenizer))

model.resize_token_embeddings(len(tokenizer))
print(model.embeddings.word_embeddings.weight.size())

# Randomly generated matrix
print(model.embeddings.word_embeddings.weight[-2:, :])

# vocabulary size: 30522
# After we add 2 tokens
# vocabulary size: 30524
# torch.Size([30524, 768])
# tensor([[-0.0186,  0.0512,  0.0323,  ...,  0.0029,  0.0260, -0.0018],
#         [-0.0153,  0.0004, -0.0041,  ...,  0.0021,  0.0544,  0.0123]],
#        grad_fn=<SliceBackward0>)
