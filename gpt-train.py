import numpy as np
import torch
import tokenizer
import model

# hyper-parameters
training_loops = 100
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------

data = np.load('data.npy')
data = torch.tensor(data, dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print('\nInitializing model...')
m = model.TransformerLM()
m = m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
n_params = sum(p.nelement() for p in m.parameters())
print(f'Num parameters:\t{n_params}')

def get_batch(split='train'):
    data = train_data if split=='train' else val_data
    idx = torch.randint(0, len(data)-model.block_size, (model.batch_size,))
    x = torch.stack([data[i : i+model.block_size] for i in idx])
    y = torch.stack([data[i+1 : i+model.block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

print(f'\nTraining model on {device}...')
m.train()
for i in range(training_loops):
    xb, targets = get_batch()

    logits, loss = m(xb, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i % (training_loops/10) == 0:
        print(f'loop {i}/{training_loops} loss:\t\t{loss}')

m.eval()
print('Done training.')

torch.save(m.state_dict(), 'model_weights.pth')
print('\nModel weights saved.\n')