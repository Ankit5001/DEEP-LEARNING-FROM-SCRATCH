import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Data loading and preprocessing
with open(r"E:\DL-Scratch\Dataset\GOT_books.txt", 'r', encoding='utf-8') as f:
    txt = f.read()

tokens = len(txt)
vocab_size = len(set(txt))

stoi = {s: i for i, s in enumerate(sorted(list(set(txt))))}
itos = {i: s for i, s in enumerate(sorted(list(set(txt))))}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Hyperparameters
seq = 256
batch = 64
d_model = 64
n_blocks = 2
num_heads = 6
head_size = d_model // num_heads
learning_rate = 0.0001
epochs = 10000
dropout = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Data loader
def get_batch():
    idx = torch.randint(0, tokens - seq - 1, (batch,))
    batch_x = torch.tensor([[stoi[txt[x]] for x in range(idx[i], idx[i] + seq)] for i in range(batch)], device=device)
    batch_y = torch.tensor([[stoi[txt[x]] for x in range(idx[i] + 1, idx[i] + seq + 1)] for i in range(batch)], device=device)
    return batch_x, batch_y

# Attention Head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        wei = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        wei = wei.masked_fill(torch.tril(torch.ones(wei.shape, dtype=torch.bool, device=wei.device)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return out

# Feedforward
class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Block
class Block(nn.Module):
    def __init__(self, num_heads, head_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffn = Feedforward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.ln1(self.sa(x)))
        x = x + self.dropout(self.ln2(self.ffn(x)))
        return x

# Transformer decoder
class Transformer_decoder(nn.Module):
    def __init__(self, num_heads=6, head_size=64, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq, d_model)
        self.blocks = nn.Sequential(*[Block(num_heads, head_size, dropout) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.dropout(self.token_embedding(idx))
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits

model = Transformer_decoder(dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
def train():
    print("Total Tokens:", tokens)
    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    print('Training started...')

    for epoch in tqdm(range(epochs + 1)):
        X, y = get_batch()
        y = y.view(-1).long()
        logits = model(X)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"EPOCH: {epoch} || LOSS: {loss.item()}")
            print(generate(model, "Win", 50))

# Generation function
def generate(model, prompt, max_len):
    model.eval()
    encoded_prompt = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    generated_text = prompt

    for _ in range(max_len):
        with torch.no_grad():
            predictions = model(encoded_prompt)
            probs = F.softmax(predictions[:, -1, :], dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1).item()

        next_char = decode([next_char_idx])
        generated_text += next_char
        encoded_prompt = torch.cat([encoded_prompt, torch.tensor([[next_char_idx]], dtype=torch.long, device=device)], dim=1)

    return generated_text

train()
print(generate(model, "W", 50))