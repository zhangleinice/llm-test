import torch
from transformers import AutoTokenizer, AutoModel

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

num_added_toks = tokenizer.add_tokens(
    ['[ENT_START]', '[ENT_END]'], special_tokens=True)

descriptions = ['start of entity', 'end of entity']

with torch.no_grad():
    for i, token in enumerate(reversed(descriptions), start=1):
        tokenized = tokenizer.tokenize(token)
        print(tokenized)
        tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
        new_embedding = model.embeddings.word_embeddings.weight[tokenized_ids].mean(
            axis=0)
        model.embeddings.word_embeddings.weight[-i,
                                                :] = new_embedding.clone().detach().requires_grad_(True)
print(model.embeddings.word_embeddings.weight[-2:, :])


# ['end', 'of', 'entity']
# ['start', 'of', 'entity']
# tensor([[-0.0340, -0.0144, -0.0441,  ..., -0.0016,  0.0318, -0.0151],
#         [-0.0060, -0.0202, -0.0312,  ..., -0.0084,  0.0193, -0.0296]],
#        grad_fn=<SliceBackward0>)
