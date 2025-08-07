import torch
import tokenizer
import model

# hyper-parameters
training_loops = 1

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

print('\nInitializing model...')
m = model.TransformerLM()
m = m.to(model.device)
optimizer = torch.optim.AdamW(m.parameters(), lr=model.learning_rate)
n_params = sum(p.nelement() for p in m.parameters())
print(f'Num parameters:\t{n_params}')

def get_batch(split='train'):
    data = train_data if split=='train' else val_data
    idx = torch.randint(0, len(data)-model.block_size, (model.batch_size,))
    x = torch.stack([data[i : i+model.block_size] for i in idx])
    y = torch.stack([data[i+1 : i+model.block_size+1] for i in idx])
    x, y = x.to(model.device), y.to(model.device)
    return x, y

print('\nTraining model...')
for i in range(training_loops):
    xb, targets = get_batch()

    logits, loss = m(xb, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i % (training_loops/10) == 0:
        print(f'loop {i}/{training_loops} loss:\t\t{loss}')

print('Done training.')

torch.save(m.state_dict(), 'model_weights.pth')
print('\nModel weights saved.\n')