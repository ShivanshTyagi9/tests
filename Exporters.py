def export_to_csv_with_time(future, mmsi, start_time, dt):
    times = [start_time + i*dt for i in range(len(future))]
    df = pd.DataFrame({
        "MMSI": [mmsi]*len(future),
        "time_sec": times,
        "lon": future[:,0],
        "lat": future[:,1]
    })
    df.to_csv("prediction.csv", index=False)
