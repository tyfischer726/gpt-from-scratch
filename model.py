import torch
import torch.nn as nn
from torch.nn import functional as F
import tokenizer

# hyper-parameters
vocab_size = len(tokenizer.vocab)
block_size = 16
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.001
n_emb = 32

# ---------------------------

class TransformerLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, n_emb)

        self.ln = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, xb, targets=None):
        tok_emb = self.token_embedding_layer(xb)
        emb = tok_emb
        
        x = emb

        logits = self.lm_head(self.ln(x))

        if targets is None:
            loss = None
        else:
            B, T, VS = logits.shape
            logits = logits.view(B*T, VS)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, context, output_length):
        for _ in range(output_length):
            truncated_context = context[:, -block_size:]
            logits, loss = self(truncated_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, sample], dim=-1)
        return context