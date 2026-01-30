import pandas as pd
import numpy as np
import torch
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

def fx(x, dt):
    lon, lat, v, psi, omega = x
    lon += v * dt * np.cos(psi)
    lat += v * dt * np.sin(psi)
    return np.array([lon, lat, v, psi, omega])

def hx(x):
    return np.array([x[0], x[1], x[2], x[3]])

df = pd.read_csv("ais.csv")
mmsi = df["MMSI"].iloc[0]
track = df[df["MMSI"] == mmsi]

dt = 10.0
points = MerweScaledSigmaPoints(5, 0.1, 2., 0)
ukf = UKF(5, 4, fx=fx, hx=hx, dt=dt, points=points)

ukf.x = np.array([
    track.iloc[0]["long"],
    track.iloc[0]["lat"],
    track.iloc[0]["sog"],
    np.deg2rad(track.iloc[0]["heading"]),
    0.0
])

X, Y = [], []

for i in range(len(track)-1):
    row = track.iloc[i]
    z = np.array([
        row["long"],
        row["lat"],
        row["sog"],
        np.deg2rad(row["heading"])
    ])

    ukf.predict()
    ukf.update(z)

    true_next = track.iloc[i+1][["long","lat"]].values
    pred = ukf.x[:2]
    residual = true_next - pred

    inp = [ukf.x[0], ukf.x[1], ukf.x[2],
           row["cog"], row["heading"], ukf.x[4]]

    X.append(inp)
    Y.append(residual)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

torch.save((X, Y), "train.pt")
