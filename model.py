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
n_emb = 64
n_heads = 4
dropout = 0.2
n_blocks = 1

# ---------------------------

class SAHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.head_size = n_emb // n_heads
        self.query = nn.Linear(n_emb, self.head_size, bias=False)
        self.key = nn.Linear(n_emb, self.head_size, bias=False)
        self.value = nn.Linear(n_emb, self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        B, T, C = emb.shape

        q = self.query(emb)
        k = self.key(emb)
        v = self.value(emb)

        att_pat = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        att_pat = att_pat.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att_pat = F.softmax(att_pat, dim=-1)
        att_pat = self.dropout(att_pat)

        d_emb = att_pat @ v

        return d_emb
    
class MultiHeadedSA(nn.Module):

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(SAHead() for _ in range(n_heads))
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb):
        d_emb = torch.cat([h(emb) for h in self.heads], dim=-1)
        d_emb = self.dropout(self.proj(d_emb))
        return d_emb
    
class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, emb):
        return emb
    
class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.mhsa = MultiHeadedSA()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, emb):
        x = emb + self.mhsa(self.ln1(emb))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLM(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_layer = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, xb, targets=None):
        B, T = xb.shape

        tok_emb = self.token_embedding_layer(xb)
        pos_emb = self.position_embedding_layer(torch.arange(0, T))
        emb = tok_emb + pos_emb
        x = self.blocks(emb)
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