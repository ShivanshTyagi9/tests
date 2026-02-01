import torch
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import torch.nn as nn

class ResidualGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

model = ResidualGRU()
model.load_state_dict(torch.load("gru.pth"))
model.eval()

def fx(x, dt):
    lon, lat, v, psi, omega = x
    lon += v * dt * np.cos(psi)
    lat += v * dt * np.sin(psi)
    return np.array([lon, lat, v, psi, omega])

def hx(x):
    return np.array([x[0], x[1], x[2], x[3]])

points = MerweScaledSigmaPoints(5, 0.1, 2., 0)
ukf = UKF(5, 4, fx=fx, hx=hx, dt=10, points=points)

ukf.x = np.array([lon0, lat0, sog0, psi0, 0])

future = []

for i in range(30):
    ukf.predict()

    inp = torch.tensor([[[
        ukf.x[0], ukf.x[1], ukf.x[2],
        cog, heading, ukf.x[4]
    ]]], dtype=torch.float32)

    residual = model(inp).detach().numpy()[0]
    corrected = ukf.x[:2] + residual
    future.append(corrected)

print("Future path (UKF + GRU):")
print(np.array(future))
