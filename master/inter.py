import pandas as pd
import numpy as np
import argparse

# -----------------------------
# Haversine distance (km)
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_interpolation(gt_csv, pred_csv, penalty_distance=1000):
    if isinstance(gt_csv, str):
        gt = pd.read_csv(gt_csv)
    else:
        gt = gt_csv.copy()

    if isinstance(pred_csv, str):
        pred = pd.read_csv(pred_csv)
    else:
        pred = pred_csv.copy()

    # Merge on path_id + point_id
    merged = pd.merge(
        gt, pred,
        on=["path_id", "point_id"],
        how="left",
        suffixes=("_gt", "_pred")
    )

    # Compute per-point distance
    distances = merged.apply(
        lambda r: haversine(
            r["latitude_gt"], r["longitude_gt"],
            r["latitude_pred"], r["longitude_pred"]
        ) if pd.notna(r["latitude_pred"]) else penalty_distance,
        axis=1
    )

    merged["haversine_dist_km"] = distances

    # RMSE per path
    rmse_per_path = merged.groupby("path_id")["haversine_dist_km"].apply(
        lambda x: np.sqrt(np.mean(x**2))
    )

    # Global RMSE
    consolidated_rmse = np.sqrt(np.mean(distances**2))

    # Final output table (human friendly)
    results = merged[[
        "path_id",
        "point_id",
        "base_date_time_gt",
        "longitude_gt",
        "latitude_gt",
        "longitude_pred",
        "latitude_pred",
        "sog_gt",
        "sog_pred",
        "heading_gt",
        "heading_pred",
        "haversine_dist_km"
    ]].copy()

    results["consolidated_rmse"] = ""
    results.loc[0, "consolidated_rmse"] = str(round(consolidated_rmse, 6))

    return results, rmse_per_path, consolidated_rmse


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate interpolation (real AIS format)")
    parser.add_argument("--gt",default="./gt/210610000.csv", required=False, help="Ground truth CSV")
    parser.add_argument("--pred",default="./submissions/210610000.csv", required=False, help="Prediction CSV")
    parser.add_argument("--out", default="interpolation_results.csv")
    args = parser.parse_args()

    results_df, rmse_per_path, global_rmse = evaluate_interpolation(
        args.gt, args.pred
    )

    results_df.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print("\nRMSE per path:")
    print(rmse_per_path)
    print("\nGlobal consolidated RMSE:", round(global_rmse, 6))
