import pandas as pd
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ---------------- Motion model (CTRV) ----------------
def fx(x, dt):
    lon, lat, v, psi, omega = x

    if abs(omega) < 1e-4:
        lon += v * dt * np.cos(psi)
        lat += v * dt * np.sin(psi)
    else:
        lon += (v / omega) * (np.sin(psi + omega*dt) - np.sin(psi))
        lat += (v / omega) * (-np.cos(psi + omega*dt) + np.cos(psi))

    psi += omega * dt
    return np.array([lon, lat, v, psi, omega])

# ---------------- Measurement model ----------------
def hx(x):
    lon, lat, v, psi, omega = x
    return np.array([lon, lat, v, psi])

# ---------------- Load CSV ----------------
df = pd.read_csv("ais.csv")
df = df.sort_values("MMSI")

# Pick one vessel
mmsi = df["MMSI"].iloc[0]
track = df[df["MMSI"] == mmsi]

dt = 10.0  # seconds

# ---------------- UKF Setup ----------------
points = MerweScaledSigmaPoints(5, alpha=0.1, beta=2., kappa=0)

ukf = UKF(5, 4, fx=fx, hx=hx, dt=dt, points=points)

lon0 = track.iloc[0]["long"]
lat0 = track.iloc[0]["lat"]
sog0 = track.iloc[0]["sog"]
psi0 = np.deg2rad(track.iloc[0]["heading"])

ukf.x = np.array([lon0, lat0, sog0, psi0, 0.0])
ukf.P *= 10
ukf.Q = np.diag([1e-6, 1e-6, 0.5, 0.01, 0.001])
ukf.R = np.diag([1e-5, 1e-5, 0.5, 0.05])

# ---------------- Run Filter ----------------
filtered = []

for _, row in track.iterrows():
    z = np.array([
        row["long"],
        row["lat"],
        row["sog"],
        np.deg2rad(row["heading"])
    ])
    ukf.predict()
    ukf.update(z)
    filtered.append(ukf.x.copy())

filtered = np.array(filtered)

# ---------------- Future prediction ----------------
future = []
for i in range(30):
    ukf.predict()
    future.append(ukf.x[:2])

future = np.array(future)

print("Future path (UKF only):")
print(future)
