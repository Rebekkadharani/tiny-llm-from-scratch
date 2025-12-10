import torch
import torch.nn as nn
import torch.optim as optim

# Read data
text = open("data.txt", "r").read()

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)

# Prepare dataset
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# Tiny LLM model (character-level)
class TinyLLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, 32)
        self.rnn = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden

model = TinyLLM(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(200):
    x = data[:-1]
    y = data[1:]

    logits, _ = model(x.unsqueeze(0))
    loss = loss_fn(logits.view(-1, vocab_size), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("Epoch:", epoch, " Loss:", loss.item())

# Save model
torch.save(model.state_dict(), "tinyllm.pth")
print("Training finished! Model saved as tinyllm.pth")
