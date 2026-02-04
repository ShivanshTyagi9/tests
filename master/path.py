import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from math import radians, sin, cos, asin, sqrt
from fastdtw import fastdtw
from geopy.distance import geodesic

# -----------------------------
# Haversine distance (km)
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))


# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_path_prediction(gt_csv, pred_csv, threshold_km=5):
    if isinstance(gt_csv, str):
        gt = pd.read_csv(gt_csv)
    else:
        gt = gt_csv.copy()

    if isinstance(pred_csv, str):
        pred = pd.read_csv(pred_csv)
    else:
        pred = pred_csv.copy()

    gt['base_date_time'] = pd.to_datetime(gt['base_date_time'], format='%d-%m-%y %H:%M')
    pred['base_date_time'] = pd.to_datetime(pred['base_date_time'], format='%d-%m-%y %H:%M')

    results = []

    for path_id in gt['path_id'].unique():
        gt_path = gt[gt['path_id'] == path_id]
        pred_path = pred[pred['path_id'] == path_id]

        # Split days
        last_day = gt_path['base_date_time'].dt.date.max()
        gt_day2 = gt_path[gt_path['base_date_time'].dt.date == last_day]

        merged = pd.merge(
            gt_day2, pred_path,
            on=['path_id','base_date_time'],
            how='left',
            suffixes=('_gt','_pred')
        )

        penalty = 1000
        distances = merged.apply(
            lambda r: haversine(
                r['latitude_gt'], r['longitude_gt'],
                r['latitude_pred'], r['longitude_pred']
            ) if pd.notna(r['latitude_pred']) else penalty,
            axis=1
        )

        rmse = np.sqrt(np.mean(distances**2))
        ade = distances.mean()
        fde = distances.iloc[-1]

        # F1
        y_true = np.ones(len(distances))
        y_pred = (distances <= threshold_km).astype(int)
        f1 = f1_score(y_true, y_pred)

        # Speed RMSE
        if 'sog_pred' in merged:
            speed_rmse = np.sqrt(
                np.mean((merged['sog_gt'] - merged['sog_pred'])**2)
            )
        else:
            speed_rmse = np.nan

        # Heading error
        if 'heading_pred' in merged:
            heading_err = np.mean(
                np.abs(merged['heading_gt'] - merged['heading_pred'])
            )
        else:
            heading_err = np.nan

        # DTW
        gt_points = list(zip(merged['latitude_gt'], merged['longitude_gt']))
        pred_points = list(zip(merged['latitude_pred'], merged['longitude_pred']))
        dtw_dist, _ = fastdtw(
            gt_points, pred_points,
            dist=lambda a,b: geodesic(a,b).km
        )

        results.append({
            "path_id": path_id,
            "rmse": round(rmse,4),
            "ade": round(ade,4),
            "fde": round(fde,4),
            "f1": round(f1,4),
            "speed_rmse": round(speed_rmse,4),
            "heading_error": round(heading_err,4),
            "dtw": round(dtw_dist,4)
        })

    return pd.DataFrame(results)


# -----------------------------
# CLI usage
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",default="./gt/210610000.csv", required=False)
    parser.add_argument("--pred",default="./submissions/210610000.csv", required=False)
    parser.add_argument("--out", default="path_prediction_results.csv")
    args = parser.parse_args()

    df = evaluate_path_prediction(args.gt, args.pred)
    df.to_csv(args.out, index=False)
    print("Saved:", args.out)
