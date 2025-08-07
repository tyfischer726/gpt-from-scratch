import torch
import tokenizer
import model

# hyper-parameters
output_length = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'training_outputs'

# ---------------------------
print('\nInitializing model...')
m = model.TransformerLM()
print('Loading model weights...')
m.load_state_dict(torch.load(f'{output_dir}/model_weights.pth', weights_only=True))
m = m.to(device)
print(f'Model moved to {device}.')

output = ''
m.eval()
while True:
    context = input("\nEnter prompt, or type 'quit': ")
    if context.strip().lower() == 'quit':
        break
    context = '\n' if len(context)==0 else context
    context = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).view(1,-1)
    print('\nModel:\n----------------------')
    output = tokenizer.decode(m.generate(context, output_length)[0].tolist())
    print(output)
    print('----------------------')

print('\nDone.')