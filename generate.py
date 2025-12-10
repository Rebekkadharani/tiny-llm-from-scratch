import torch
import torch.nn as nn

# Load mappings from train.py
text = open("data.txt", "r").read()
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)

# Tiny LLM model (same as training)
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
model.load_state_dict(torch.load("tinyllm.pth"))
model.eval()

# Generate text
def generate_text(start="A", length=200):
    input_char = torch.tensor([[char_to_idx[start]]])
    hidden = None
    output_text = start

    for _ in range(length):
        logits, hidden = model(input_char, hidden)
        probs = torch.softmax(logits[0, -1], dim=0)
        next_idx = torch.multinomial(probs, 1).item()
        next_char = idx_to_char[next_idx]

        output_text += next_char
        input_char = torch.tensor([[next_idx]])

    return output_text

# Ask user
start = input("Start with a character: ")
print(generate_text(start=start))
