import torch
import tokenizer
import model

# hyper-parameters
output_length = 300
context = "\n"

# ---------------------------
m = model.TransformerLM()
m = m.to(model.device)
m.load_state_dict(torch.load('model_weights.pth', weights_only=True))

context = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=model.device).view(1,-1)
print('\nModel output:\n--------------\n')
print(tokenizer.decode(m.generate(context, output_length)[0].tolist()))
print('\n--------------\n')