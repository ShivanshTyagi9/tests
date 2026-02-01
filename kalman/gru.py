import torch
import torch.nn as nn

X, Y = torch.load("train.pt")

class ResidualGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

model = ResidualGRU()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# reshape for sequence
X = X.unsqueeze(1)

for epoch in range(50):
    pred = model(X)
    loss = loss_fn(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("Epoch", epoch, loss.item())

torch.save(model.state_dict(), "gru.pth")
