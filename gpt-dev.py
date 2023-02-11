import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, indx, targets=None):
        # indx and targets are both (B, T) tensor of integers
        # B: batch
        # T: time
        # C: channel (vocab_size in this case)
        logits = self.token_embedding_table(indx) # (B, T, C)
        if targets is None:
            loss = None
        else:
            # Because of the way that PyTorch expects the input of this tensor,
            # we need to reshape it before handing it off to the cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, indx, max_new_tokens):
        #indx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(indx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probbilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            indx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            indx = torch.cat((indx, indx_next), dim=1) # (B, T+1)
        return indx

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s : [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l : ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# time to encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) # the 1,000 characters we looked at earlier will look like this to GPT

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # First 90% will be training data, the rest will be validation
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?

def get_batch(split):
    # generatae a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(indx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for steps in range(10000):
    # sample a baatch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(indx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=400)[0].tolist()))

# consider the following exmample
torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
x.shape