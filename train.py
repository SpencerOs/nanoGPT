# We always start with a datset to traian on. Let's downlod the tiny shaakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshaakespeare/input.txt
import torch

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in chaaracters: ", len(text))

# let's look aat the first 1000 characters
print(text[:1000])

# here are aall the unique characters thata occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create  mapping from chaaracters into integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s : [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: taake a list of integers, output a string

htStr = "hii there"
print(encode(htStr))
print(decode(encode(htStr)))

# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1,000 characters we looked at earlier will look like this to GPT

# Let's now split up the dataa into train and validation sets
n = int(0.9*len(data)) # First 90% will be train, rest is validation
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f'When input is {context} the target: {target}')

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x nd targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-------')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off of the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, indx, targets): 
        # indx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(indx) # (B, T, C)

        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print(out.shape)