import torch
import tokenizer
import model

# hyper-parameters
vocab_size = tokenizer.vocab_size

# ---------------------------

with open('datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print('\nEncoding data...')
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(f'len text:\t\t{len(text)}')
print(f'len data:\t\t{len(data)}')

m = model.TransformerLM()

print('\nDone.')