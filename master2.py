from sklearn.metrics import f1_score, root_mean_squared_error, average_precision_score
from shapely.geometry import Polygon
from collections import defaultdict
from openpyxl import load_workbook
from scipy.stats import rankdata
import zipfile, traceback, json
from datetime import datetime
import argparse, logging
from pathlib import Path
from shapely import wkt
import numpy as np
import pandas as pd
import os, sys, math
import pytz, glob
import shutil
from math import radians, sin, cos, asin, sqrt
from fastdtw import fastdtw
from geopy.distance import geodesic


np.set_printoptions(legacy='1.25')
IST = pytz.timezone('Asia/Kolkata')

def ist_time(*args):
    return datetime.now(IST).timetuple()

# Apply custom formatter with IST time
logging.Formatter.converter = ist_time
logging.basicConfig(
    filename='PS_09_logs.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s : %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S'
)

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
# utilitues
# -----------------------------
def organize_annotations(data):
    image_to_annotations = defaultdict(list)
    for ann in data['annotations']:
        image_to_annotations[ann['image_id']].append({
            'polygon': wkt.loads(ann['polygon']),
            'category_id': ann['category_id'],
            'length': float(ann['length']) if ann.get('length') not in ["", None] else np.nan,
            'width': float(ann['width']) if ann.get('width') not in ["", None] else np.nan,
            'heading': float(ann['heading']) if ann.get('heading') not in ["", None] else np.nan,
            'score': ann.get('score', 1.0),
            'used': False
        })
    return image_to_annotations


def match_detections(gt_anns, pred_anns, iou_thr=0.5):
    matches = []
    used_gt = set()

    preds = sorted(pred_anns, key=lambda x: x.get('score', 1.0), reverse=True)

    for pred in preds:
        best_iou, best_gt = 0, None
        for i, gt in enumerate(gt_anns):
            if i in used_gt:
                continue
            iou_val = iou(pred['polygon'], gt['polygon'])
            if iou_val >= iou_thr and iou_val > best_iou:
                best_iou, best_gt = iou_val, i

        if best_gt is not None:
            used_gt.add(best_gt)
            matches.append((gt_anns[best_gt], pred))
        else:
            matches.append((None, pred))

    return matches

def compute_rmse_from_matches(matches, key):
    errors = []
    for gt, pred in matches:
        if gt is None:
            continue
        if np.isnan(gt[key]) or np.isnan(pred[key]):
            continue
        errors.append((gt[key] - pred[key]) ** 2)

    return np.sqrt(np.mean(errors)) if errors else np.nan

def circular_heading_error(gt_h, pred_h):
    diff = abs(gt_h - pred_h)
    return min(diff, 360 - diff)

def compute_heading_rmse(matches):
    errors = []
    for gt, pred in matches:
        if gt is None:
            continue
        if np.isnan(gt['heading']) or np.isnan(pred['heading']):
            continue
        err = circular_heading_error(gt['heading'], pred['heading'])
        errors.append(err ** 2)

    return np.sqrt(np.mean(errors)) if errors else np.nan

from sklearn.metrics import f1_score

def compute_weighted_f1(matches):
    y_true, y_pred = [], []

    for gt, pred in matches:
        if gt is None:
            continue
        y_true.append(gt['category_id'])
        y_pred.append(pred['category_id'])

    if not y_true:
        return 0.0

    return f1_score(
        y_true,
        y_pred,
        average='weighted'
    )


# -----------------------------
# Core path pred evaluation
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

        if pred_path.empty:
            results.append({
                "path_id": path_id,
                "rmse": 1000,
                "ade": 1000,
                "fde": 1000,
                "f1": 0.0,
                "speed_rmse": 1000,
                "heading_error": 180,
                "dtw": 1000
            })
            continue


        merged = pd.merge(
            gt_day2, pred_path,
            on=['path_id','base_date_time', 'point_id'],
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
        # gt_points = list(zip(merged['latitude_gt'], merged['longitude_gt']))
        # pred_points = list(zip(merged['latitude_pred'], merged['longitude_pred']))
        # dtw_dist, _ = fastdtw(
        #     gt_points, pred_points,
        #     dist=lambda a,b: geodesic(a,b).km
        # )

        # DTW (drop rows with invalid coordinates)
        dtw_df = merged.dropna(
            subset=['latitude_gt', 'longitude_gt', 'latitude_pred', 'longitude_pred']
        )

        if dtw_df.empty:
            dtw_dist = penalty
        else:
            gt_points = list(zip(dtw_df['latitude_gt'], dtw_df['longitude_gt']))
            pred_points = list(zip(dtw_df['latitude_pred'], dtw_df['longitude_pred']))

            dtw_dist, _ = fastdtw(
                gt_points,
                pred_points,
                dist=lambda a, b: geodesic(a, b).km
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

## Interpolation data
def haversine_rmse(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    """
    Compute RMSE (in meters) between two sets of lat/lon coordinates.

    Parameters:
        lat1_arr, lon1_arr: arrays of lat/lon for prediction 1
        lat2_arr, lon2_arr: arrays of lat/lon for prediction 2

    Returns:
        RMSE (float): Root Mean Square Error in meters
    """
    R = 6371  # Earth radius in kms

    # Convert from degrees to radians
    lat1_rad = np.radians(lat1_arr)
    lon1_rad = np.radians(lon1_arr)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c  # Distance between each point in kms
    return distances



def save_result_dict_to_csv(result_dict):
    # Convert the nested dict into a list of rows with serial_no
    rows = []
    for serial_no, fields in result_dict.items():
        row = {"serial_no": serial_no}
        row.update(fields)
        rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    return df





def empty_csv_creator(pro_type, gt_csv):
    index=0
    results={}
    if pro_type=='correlation':
        for _, gt_row in gt_csv.iterrows():
            gt_lat = gt_row['vessel_latitude']
            gt_lon = gt_row['vessel_longitude']
            gt_image_name = gt_row['image_name']
        
            results[str(index+1)]={}
            results[str(index+1)]['image_name'] = gt_image_name
            results[str(index+1)]['gt_latitude'] = '0'
            results[str(index+1)]['gt_longitude'] = '0'
            results[str(index+1)]['pred_latitude'] = '0'
            results[str(index+1)]['pred_longitude'] = '0'
            results[str(index+1)]['haversine_km'] = '0'
            results[str(index+1)]['mmsi_gt'] = '0'
            results[str(index+1)]['mmsi_pred'] = '0'
            results[str(index+1)]['match'] = '0'
            index+=1
        results[str(1)]['f1_score'] = '0.0'
    
    elif pro_type=='interpolation':
        for _, row in gt_csv.iterrows():
            results[str(index+1)]={}
            results[str(index+1)]["path_id"]=row["path_id"]
            results[str(index+1)]["point_id"]=row["point_id"]
            results[str(index+1)]["point_latitude_gt"]=row["point_latitude"]
            results[str(index+1)]["point_longitude_gt"]=row["point_longitude"]
            results[str(index+1)]["point_latitude_pred"]='0'
            results[str(index+1)]["point_longitude_pred"]='0'
            results[str(index+1)]["haversine_dist"]=1000
            index+=1
        results[str(1)]["consolidated_rmse"] = np.inf
    else:
        
        images = gt_csv.get('images', [])
        rows = []
        for idx, img in enumerate(images, 1):
            rows.append({
                'serial_no': idx,
                'id': img['id'],
                'file_name': img['file_name'],
                'AP': 0,
                'mAP': 0 if idx==1 else ''
            })
        return pd.DataFrame(rows)
        
    return save_result_dict_to_csv(results)

#path prediction
def process_path(submissions_root,
        input_filename='path_prediction_results.csv',
        output_filename='all_teams_path.csv'
    ):
    submissions_path = Path(submissions_root)
    all_teams = []

    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name=="output_log": continue

            input_file = team_folder / input_filename
            if not input_file.exists():
                continue

            try:
                df = pd.read_csv(input_file)
                if df.empty:
                    continue

                score_series = (
                        df
                        .set_index("path_id")["rmse"]
                        .rename(team_name)
                    )
                all_teams.append(score_series)

            except Exception as e:
                logging.error(f"Path processing failed for {team_name}: {e}")

    if all_teams:
        final_df = pd.concat(all_teams, axis=1)
        final_df.index.name = "path"
        final_df.to_csv(submissions_path / output_filename)


## AIS Correlation data
def calculate_f1_per_image(gt_csv, pred_csv):
    pred = pd.read_csv(pred_csv)
    
    results = {}
    
    merged_data = []

    for _, gt_row in gt_csv.iterrows():
        gt_lat = gt_row['vessel_latitude']
        gt_lon = gt_row['vessel_longitude']
        gt_image_name = gt_row['image_name']
        
        pred_image_data = pred[pred['image_name'] == gt_image_name]
        closest_pred = None
        min_distance = float('inf')
        
        for _, pred_row in pred_image_data.iterrows():
            pred_lat = pred_row['vessel_latitude']
            pred_lon = pred_row['vessel_longitude']
            pred_mmsi = pred_row['mmsi']
            
            # Calculate the Haversine distance between the ground truth and prediction lat/long
            distance = haversine_rmse(gt_lat, gt_lon, pred_lat, pred_lon)
            
            # If this prediction is closer, update the closest match
            if distance < min_distance:
                min_distance = distance
                closest_pred = pred_row
        
        if closest_pred is not None:
            merged_row = {
                'image_name': gt_image_name,
                'gt_latitude': gt_lat,
                'gt_longitude': gt_lon,
                'pred_latitude': closest_pred['vessel_latitude'],
                'pred_longitude': closest_pred['vessel_longitude'],
                'mmsi_gt': gt_row['mmsi'],
                'mmsi_pred': closest_pred['mmsi'],
                'haversine_km': min_distance
            }
        else:
            merged_row = {
                'image_name': gt_image_name,
                'gt_latitude': gt_lat,
                'gt_longitude': gt_lon,
                'pred_latitude': 'Not found',
                'pred_longitude': 'Not found',
                'mmsi_gt': gt_row['mmsi'],
                'mmsi_pred': -1,
                'haversine_km': 100
            }
        merged_data.append(merged_row)
        
    # Convert merged_data into a DataFrame
    merged = pd.DataFrame(merged_data)
    
    
    logging.info(f"{len(merged)}")
    #print(merged)
    y_true, y_pred = [], []
    for index, row in merged.iterrows():
        results[str(index+1)]={}
        results[str(index+1)]['image_name'] = row['image_name']
        results[str(index+1)]['gt_latitude'] = row['gt_latitude']
        results[str(index+1)]['gt_longitude'] = row['gt_longitude']
        results[str(index+1)]['pred_latitude'] = row['pred_latitude']
        results[str(index+1)]['pred_longitude'] = row['pred_longitude']
        results[str(index+1)]['haversine_km'] = row['haversine_km']
        results[str(index+1)]['mmsi_gt'] = row['mmsi_gt']
        results[str(index+1)]['mmsi_pred'] = row['mmsi_pred']
        
        #if row['mmsi_gt'] == 0:
        #    results[str(index+1)]['match'] = -1
        #    continue
        #print(row['image_name'])
        
        
        if np.less_equal(float(row['haversine_km']),1.0) and int(row['mmsi_gt'])==int(row['mmsi_pred']):
            y_true.append(1)
            y_pred.append(1)
        else:
            y_true.append(1)
            y_pred.append(0)
        
        results[str(index+1)]['match'] = y_pred[-1]
    
    #print(np.unique(y_pred, return_counts=True))
    f1 = f1_score(y_true, y_pred, pos_label=1) if len(y_true) > 0 else np.nan
    results[str(1)]['f1_score'] = np.round(f1,6)
    return results







## Interpolation scores
def calculate_rmse_per_image(gt_csv, pred_csv):
    pred = pd.read_csv(pred_csv)
    #print(pred)
    merged = pd.merge(gt_csv, pred, on=["path_id", "point_id"], how="left", suffixes=("_pred", "_gt"))
    merged.columns = merged.columns.str.replace(' ', '_')
    
    merged['point_latitude_pred'] = merged['point_latitude_pred'].fillna(np.nan)
    merged['point_longitude_pred'] = merged['point_longitude_pred'].fillna(np.nan)

    
    #print(len(merged))
    #print(merged.head())
    rmse_fields = {}
    penalty_distance = 1000 ## if path and point id, not present
    distances = merged.apply(
        lambda row: haversine_rmse(row['point_latitude_gt'], row['point_longitude_gt'],
                             row['point_latitude_pred'], row['point_longitude_pred']) 
        if pd.notna(row['point_latitude_pred']) and pd.notna(row['point_longitude_pred'])
        else penalty_distance, axis=1
    )
    
    
    #distances = haversine_rmse(merged[f"{"point_latitude"}_gt"], merged[f"{"point_longitude"}_gt"], merged[f"{"point_latitude"}_pred"],merged[f"{"point_longitude"}_pred"])

    for index, row in merged.iterrows():
        rmse_fields[str(index+1)]={}
        rmse_fields[str(index+1)]["path_id"]=row["path_id"]
        rmse_fields[str(index+1)]["point_id"]=row["point_id"]
        rmse_fields[str(index+1)]["point_latitude_gt"]=row["point_latitude_gt"]
        rmse_fields[str(index+1)]["point_longitude_gt"]=row["point_longitude_gt"]
        rmse_fields[str(index+1)]["point_latitude_pred"]=row["point_latitude_pred"]
        rmse_fields[str(index+1)]["point_longitude_pred"]=row["point_longitude_pred"]
        rmse_fields[str(index+1)]["haversine_dist"]=distances[index]
  #long_lat = [merged[f"{"point_latitude"}_gt"],merged[f"{"point_longitude"}_gt"]]
  #long_lat_pred = [merged[f"{"point_latitude"}_pred"],merged[f"{"point_longitude"}_pred"]]


    rmse = np.sqrt(np.mean(distances ** 2))
    rmse_fields[str(1)]["consolidated_rmse"] = np.round(rmse,6)

    return rmse_fields




## Detection data - IoU Checking
def iou(poly1: Polygon, poly2: Polygon) -> float:
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union != 0 else 0.0

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def organize_annotations(data):
    image_to_annotations = defaultdict(list)
    #print(data['annotations'])
    for ann in data['annotations']:
      #print(ann)
      image_to_annotations[ann['image_id']].append({
          'polygon': wkt.loads(ann['polygon']),
          'category_id': ann['category_id'],
          'used': False
      })
    return image_to_annotations



def compute_custom_vary_ap(gt_anns, pred_anns, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    #print(iou_thresholds)
    aps = []

    for threshold in iou_thresholds:
        gt_bboxes = [gt['polygon'] for gt in gt_anns]
        gt_used = [False]*len(gt_bboxes)

        preds = sorted(pred_anns, key=lambda x: x.get('score',1.0), reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, pred in enumerate(preds):
            pred_poly = pred['polygon']

            best_iou = 0
            best_gt_idx = -1

            for j, gt_poly in enumerate(gt_bboxes):
                if gt_used[j]:
                    continue
                iou_val = iou(pred_poly, gt_poly)
                if iou_val >= threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = j

            if best_gt_idx >= 0:
                tp[i] = 1
                gt_used[best_gt_idx] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        if len(gt_bboxes) == 0:
            recalls = np.zeros(len(tp))
        else:
            recalls = cum_tp / len(gt_bboxes)

        precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

        # AP calculation using the interpolation method (trapezoidal rule)
        ap = 0.0
        prev_recall = 0
        for p, r in zip(precisions, recalls):
            ap += p * (r - prev_recall)
            prev_recall = r

        aps.append(ap)

    # average AP over all thresholds
    return np.mean(aps)

####
def evaluate_detection(gt_data, pred_json_path):
    
    pred_data = load_data(pred_json_path)

    gt_by_image = organize_annotations(gt_data)
    pred_by_image = organize_annotations(pred_data)
    
    det_results = {}
    ap_list = []
    len_rmse_list = []
    wid_rmse_list = []
    head_rmse_list = []
    f1_list = []

    for img_id, img_info in enumerate(gt_data['images']):
        
        gt_image_id = img_info['id']
        gt_file_name = img_info['file_name']
        
        gt_anns = gt_by_image.get(gt_image_id, [])
        pred_anns = pred_by_image.get(gt_image_id, [])

        det_results[str(img_id + 1)] = {
            'id': gt_image_id,
            'file_name': gt_file_name
        }

        # ---------- mAP ----------
        ap = np.round(compute_custom_vary_ap(gt_anns, pred_anns), 6)
        det_results[str(img_id + 1)]['AP'] = ap
        ap_list.append(ap)

        # ---------- Matching ----------
        matches = match_detections(gt_anns, pred_anns, iou_thr=0.5)

        # ---------- Regression ----------
        len_rmse = compute_rmse_from_matches(matches, 'length')
        wid_rmse = compute_rmse_from_matches(matches, 'width')
        head_rmse = compute_heading_rmse(matches)
        class_f1 = compute_weighted_f1(matches)

        det_results[str(img_id + 1)].update({
            'length_rmse': len_rmse,
            'width_rmse': wid_rmse,
            'heading_rmse': head_rmse,
            'class_f1': class_f1
        })

        len_rmse_list.append(len_rmse)
        wid_rmse_list.append(wid_rmse)
        head_rmse_list.append(head_rmse)
        f1_list.append(class_f1)

    # ---------- Final aggregated scores ----------
    det_results[str(1)]['mAP'] = np.nanmean(ap_list)
    det_results[str(1)]['length_rmse_mean'] = np.nanmean(len_rmse_list)
    det_results[str(1)]['width_rmse_mean'] = np.nanmean(wid_rmse_list)
    det_results[str(1)]['heading_rmse_mean'] = np.nanmean(head_rmse_list)
    det_results[str(1)]['class_f1_mean'] = np.nanmean(f1_list)

    return det_results


# Normalize scores
def normalize(arr, higher_is_better=True):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        return np.ones_like(arr)  # all same values → return 1s
    if higher_is_better:
        return (arr - min_val) / (max_val - min_val)
    else:
        return (max_val - arr) / (max_val - min_val)



def extract_zip(zip_path: Path, extract_to: Path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info(f"Extracted {zip_path.name} to {extract_to}")
    except Exception as e:
        logging.error(f"Failed to extract {zip_path.name}: {e}")
        raise







def find_files_with_word(directory, word):
    matching_files = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        
        dirs[:] = [d for d in dirs if 'MACOSX' not in d]
        
        for file in files:
            if word in file.lower():  # case-insensitive match
                matching_files.append(os.path.join(root, file))
    
    return matching_files[0]






def evaluate_team(team_zip: Path, extract_dir: Path, ground_truth_path: Path):
    team_name = team_zip.stem
    team_folder = extract_dir / team_name
    os.makedirs(team_folder, exist_ok=True)
    print(team_zip,extract_dir,team_folder)
    team_log = ""
    
    # try:
    #     logging.info(f"")
    #     logging.info(f"Loading ground-truth..")
    #     gt_ais_csv = pd.read_csv(os.path.join(ground_truth_path, 'AIS_correlation.csv'))
    #     gt_inter_csv = pd.read_csv(os.path.join(ground_truth_path, 'interpolated_data.csv'))
    #     gt_detect = load_data(os.path.join(ground_truth_path, 'detection.json'))
    #     gt_path_csv = pd.read_csv(os.path.join(ground_truth_path, 'path_prediction.csv'))
    # except Exception as e:
    #     logging.error(f"Ground truth loading failed : {e}")
    
    # try:
    #     logging.info(f"")
    #     logging.info(f"Loading ground-truth path and interpolation..")
    #     gt_inter_csv = pd.read_csv(os.path.join(ground_truth_path, 'interpolated_data.csv'))
    #     gt_path_csv = pd.read_csv(os.path.join(ground_truth_path, 'path_prediction.csv'))
    # except Exception as e:
    #     logging.error(f"Ground truth loading failed : {e}")

    gt_ais_csv = None
    gt_inter_csv = None
    gt_detect = None
    gt_path_csv = None

    # AIS
    try:
        gt_ais_csv = pd.read_csv(os.path.join(ground_truth_path, 'AIS_correlation.csv'))
    except Exception as e:
        logging.warning(f"AIS GT missing: {e}")

    # Interpolation
    try:
        gt_inter_csv = pd.read_csv(os.path.join(ground_truth_path, 'interpolated_data.csv'))
    except Exception as e:
        logging.warning(f"Interpolation GT missing: {e}")

    # Detection
    try:
        gt_detect = load_data(os.path.join(ground_truth_path, 'detection.json'))
    except Exception as e:
        logging.warning(f"Detection GT missing: {e}")

    # Path
    try:
        gt_path_csv = pd.read_csv(os.path.join(ground_truth_path, 'path_prediction.csv'))
    except Exception as e:
        logging.warning(f"Path GT missing: {e}")


    try:
        
        try:
            extract_zip(team_zip, team_folder)
            current_team_path = os.path.join(os.getcwd(),team_folder)
            logging.info(current_team_path)
        except Exception as e:
            empty_df = empty_csv_creator('correlation',gt_ais_csv)
            output_csv = team_folder / 'AIS_correlation_results.csv'
            empty_df.to_csv(output_csv, index=False)
            
            empty_df = empty_csv_creator('interpolation',gt_inter_csv)
            output_csv = team_folder / 'interpolated_data_results.csv'
            empty_df.to_csv(output_csv, index=False)
            
            empty_df = empty_csv_creator("",gt_detect)
            output_csv = team_folder / 'detection_results.csv'
            empty_df.to_csv(output_csv, index=False)
            team_log+=f"File corrupted : {e}\n"
            logging.error(f"File corrupted : {e}")

        if gt_ais_csv is not None:
            try:
                target_string = 'correlation'
                ais_pred_path = find_files_with_word(current_team_path,target_string)
                logging.info(ais_pred_path)
                _, actual_extension = os.path.splitext(ais_pred_path)
                expected_extension = ".csv"
                assert actual_extension == expected_extension, f"File extension mismatch: Expected '{expected_extension}', got '{actual_extension}'"
            except Exception as e:
                empty_df = empty_csv_creator('correlation',gt_ais_csv)
                output_csv = team_folder / 'AIS_correlation_results.csv'
                empty_df.to_csv(output_csv, index=False)
                logging.error(f"{team_name} - correlation file not found: {e}")
                team_log+=f"correlation file not found: {e}\n"
            
        try:
            target_string = 'interpolated'
            interpolation_pred_path = find_files_with_word(current_team_path,target_string)
            logging.info(interpolation_pred_path)
            _, actual_extension = os.path.splitext(interpolation_pred_path)
            expected_extension = ".csv"
            assert actual_extension == expected_extension, f"File extension mismatch: Expected '{expected_extension}', got '{actual_extension}'"
        except Exception as e:
            empty_df = empty_csv_creator('interpolation',gt_inter_csv)
            output_csv = team_folder / 'interpolated_data_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - interpolation file not found: {e}")
            team_log+=f"interpolation file not found: {e}\n"

        if gt_detect is not None:    
            try:
                target_string = 'detection'
                detection_pred_path = find_files_with_word(current_team_path,target_string)
                logging.info(detection_pred_path)
                _, actual_extension = os.path.splitext(detection_pred_path)
                expected_extension = ".json"
                assert actual_extension == expected_extension, f"File extension mismatch: Expected '{expected_extension}', got '{actual_extension}'"
            except Exception as e:
                empty_df = empty_csv_creator("",gt_detect)
                output_csv = team_folder / 'detection_results.csv'
                empty_df.to_csv(output_csv, index=False)
                logging.error(f"{team_name} - detection file not found: {e}")
                team_log+=f"detection file not found: {e}\n"
        
        try:
            target_string = 'path'
            path_pred_path = find_files_with_word(current_team_path, target_string)
            logging.info(path_pred_path)
            _, actual_extension = os.path.splitext(path_pred_path)
            assert actual_extension == ".csv"
        except Exception as e:
            empty_df = empty_csv_creator("",gt_path_csv)
            output_csv = team_folder / 'path_prediction_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - path file not found: {e}")
            team_log += f"path file not found: {e}\n"

            
        

        
        # Paths to team predictions (modify if needed)
        #ais_pred_path = team_folder / 'AIS_correlation.csv'
        #interpolation_pred_path = team_folder / 'interpolated_data.csv'
        #detection_pred_path = team_folder / 'detection.json'

        # AIS Correlation
        if gt_ais_csv is not None:
            try:
                ais_result = calculate_f1_per_image(gt_ais_csv, ais_pred_path)
                df = save_result_dict_to_csv(ais_result)
                output_csv = team_folder / 'AIS_correlation_results.csv'
                df.to_csv(output_csv, index=False)
                logging.info(f"{team_name} - Results saved to {output_csv}")
                logging.info(f"{team_name} - AIS Correlation evaluated.")
            except Exception as e:
                empty_df = empty_csv_creator('correlation',gt_ais_csv)
                output_csv = team_folder / 'AIS_correlation_results.csv'
                empty_df.to_csv(output_csv, index=False)
                logging.error(f"{team_name} - AIS Correlation failed: {e}")
                team_log+=f"AIS Correlation failed: {e}\n"
            

        # Interpolation
        try:
            interp_result = calculate_rmse_per_image(gt_inter_csv, interpolation_pred_path)
            df = save_result_dict_to_csv(interp_result)
            output_csv = team_folder / 'interpolated_data_results.csv'
            df.to_csv(output_csv, index=False)
            logging.info(f"{team_name} - Results saved to {output_csv}")
            logging.info(f"{team_name} - Interpolation evaluated.")
        except Exception as e:
            empty_df = empty_csv_creator('interpolation',gt_inter_csv)
            output_csv = team_folder / 'interpolated_data_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - Interpolation failed: {e}")
            team_log+=f"Interpolation failed: {e}\n"
            

        # Object Detection
        if gt_detect is not None:
            try:
                detection_result = evaluate_detection(gt_detect, detection_pred_path)
                df = save_result_dict_to_csv(detection_result)
                output_csv = team_folder / 'detection_results.csv'
                df.to_csv(output_csv, index=False)
                logging.info(f"{team_name} - Results saved to {output_csv}")
                logging.info(f"{team_name} - Object Detection evaluated.")
            except Exception as e:
                empty_df = empty_csv_creator("",gt_detect)
                output_csv = team_folder / 'detection_results.csv'
                empty_df.to_csv(output_csv, index=False)
                logging.error(f"{team_name} - Object Detection failed: {e}")
                team_log+=f"Object Detection failed: {e}\n"

        # Path Prediction
        try:
            path_df = evaluate_path_prediction(gt_path_csv, path_pred_path)
            output_csv = team_folder / 'path_prediction_results.csv'
            path_df.to_csv(output_csv, index=False)
            logging.info(f"{team_name} - Path Prediction evaluated.")
        except Exception as e:
            empty_df = empty_csv_creator("",gt_path_csv)
            output_csv = team_folder / 'path_prediction_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - Path Prediction failed: {e}")
            team_log += f"Path Prediction failed: {e}\n"

        
        team_log+=f"Interpolation of 1000 implies no matching/submission.\n"
        output_log_folder = os.path.join(extract_dir, "output_log")
        os.makedirs(output_log_folder, exist_ok=True)
        output_xlsx = os.path.join(output_log_folder, f"{team_name}_output.xlsx")
        sheet_name='Logs'
        log_lines = team_log.strip().split("\n")
        #logging.info(team_log)
        df_log = pd.DataFrame({'Log Messages': log_lines})
        with pd.ExcelWriter(output_xlsx, engine='openpyxl', mode='w') as writer:
            df_log.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logging.info(f"Log written to '{sheet_name}' sheet in {output_xlsx}")
        
    except Exception as e:
        logging.error(f"{team_name} - Evaluation failed: {e}\n{traceback.format_exc()}")
    
    














def compute_f1_per_image(df: pd.DataFrame, team_name: str) -> pd.Series:
    image_level_stats = []
    ## ignoring -1 in match, 0 mmsi in gt rows
    df = df[df['match'].isin([0, 1])]
    
    image_f1_scores = {}
    
    for image_name, group in df.groupby('image_name'):
        y_true = [1] * len(group) 
        y_pred = group['match']

        f1 = f1_score(y_true, y_pred, pos_label=1) if len(y_true) > 0 else np.nan

        image_f1_scores[image_name] = np.round(f1, 6)

    return pd.Series(image_f1_scores, name=team_name)


def generate_all_teams_AIS(
        submissions_root: str,
        input_filename='AIS_correlation_results.csv',
        output_filename='all_teams_AIS_correlation.csv'
    ):
    submissions_path = Path(submissions_root)
    all_results = []

    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name=="output_log": continue
            input_file = team_folder / input_filename

            try:
                df = pd.read_csv(input_file)
                if df.empty:
                    logging.info(f"[ERROR] {team_name}: Empty file")
                    continue

                team_results = compute_f1_per_image(df, team_name)
                all_results.append(team_results)

                logging.info(f"[OK] Processed correlation {team_name}")

            except Exception as e:
                logging.error(f"[ERROR] {team_name}: {e}")

    if all_results:
        final_df = pd.concat(all_results, axis=1)
        final_df.index.name = 'image_name'
        output_file = submissions_path / output_filename
        final_df.reset_index().to_csv(output_file, index=False)
        logging.info(f"✅ All teams correlation F1 saved to: {output_file}")
    else:
        logging.error("⚠️ No results to save.")



# def process_interpolation(submissions_root,
#         input_filename='interpolated_data_results.csv',
#         output_filename='all_teams_interpolation.csv'
#     ):
#     submissions_path = Path(submissions_root)
#     all_teams = []

#     for team_folder in submissions_path.iterdir():
#         if team_folder.is_dir():
#             team_name = team_folder.name
#             if team_name=="output_log": continue
#             input_file = team_folder / input_filename

#             if not input_file.exists():
#                 logging.error(f"[WARN] {team_name}: Missing {input_filename}")
#                 continue

#             try:
#                 df = pd.read_csv(input_file)
#                 if df.empty:
#                     logging.error(f"[WARN] {team_name}: Empty interpolation result")
#                     continue

#                 # Group by path_id and calculate RMSE
#                 rmse_per_path = df.groupby('path_id')['haversine_dist'].apply(
#                     lambda x: np.sqrt((x**2).mean())
#                 )
#                 rmse_series = pd.Series(rmse_per_path, name=team_name)
#                 all_teams.append(rmse_series)

#                 logging.info(f"[OK] Processed interpolation for {team_name}")

#             except Exception as e:
#                 logging.error(f"[ERROR] Interpolation processing failed for {team_name}: {e}")

#     if all_teams:
#         final_df = pd.concat(all_teams, axis=1)
#         final_df.index.name = 'path_id'
#         final_df.reset_index().to_csv(submissions_path / output_filename, index=False)
#         logging.info(f"✅ Interpolation RMSE table saved to: {output_filename}")
#     else:
#         logging.info("\n⚠️ No interpolation results processed.")

def process_interpolation(
        submissions_root,
        input_filename='interpolated_data_results.csv',
        output_filename='all_teams_interpolation.csv'
    ):
    submissions_path = Path(submissions_root)
    all_teams = []

    for team_folder in submissions_path.iterdir():
        if not team_folder.is_dir():
            continue

        team_name = team_folder.name
        if team_name == "output_log":
            continue

        input_file = team_folder / input_filename
        if not input_file.exists():
            logging.warning(f"[WARN] {team_name}: Missing {input_filename}")
            continue

        try:
            df = pd.read_csv(input_file)
            if df.empty:
                continue

            # Compute RMSE per path_id
            rmse_per_path = (
                df.groupby("path_id")["haversine_dist"]
                  .apply(lambda x: np.sqrt((x ** 2).mean()))
                  .rename(team_name)
            )

            all_teams.append(rmse_per_path)
            logging.info(f"[OK] Processed interpolation for {team_name}")

        except Exception as e:
            logging.error(f"[ERROR] Interpolation processing failed for {team_name}: {e}")

    if all_teams:
        final_df = pd.concat(all_teams, axis=1)
        final_df.index.name = "path_id"
        final_df.reset_index().to_csv(
            submissions_path / output_filename,
            index=False
        )
        logging.info(f"✅ Interpolation RMSE table saved to: {output_filename}")
    else:
        logging.warning("⚠️ No interpolation results processed.")


# def process_detection(submissions_root,
#         input_filename='detection_results.csv',
#         output_filename='all_teams_detection.csv'
#     ):
#     submissions_path = Path(submissions_root)
#     all_teams = []

#     for team_folder in submissions_path.iterdir():
#         if team_folder.is_dir():
#             team_name = team_folder.name
#             if team_name=="output_log": continue
#             input_file = team_folder / input_filename

#             if not input_file.exists():
#                 logging.error(f"[WARN] {team_name}: Missing {input_filename}")
#                 continue

#             try:
#                 df = pd.read_csv(input_file)
#                 if df.empty:
#                     logging.error(f"[WARN] {team_name}: Empty detection result")
#                     continue

#                 ap_per_file = df.groupby('file_name')['AP'].mean()
#                 ap_series = pd.Series(ap_per_file, name=team_name)
#                 all_teams.append(ap_series)

#                 logging.info(f"[OK] Processed detection for {team_name}")

#             except Exception as e:
#                 logging.error(f"[ERROR] Detection processing failed for {team_name}: {e}")

#     if all_teams:
#         final_df = pd.concat(all_teams, axis=1)
#         final_df.index.name = 'file_name'
#         final_df.reset_index().to_csv(submissions_path / output_filename, index=False)
#         logging.info(f"✅ Detection AP table saved to: {output_filename}")
#     else:
#         logging.info("\n⚠️ No detection results processed.") 

def process_detection(
        submissions_root,
        input_filename='detection_results.csv',
        output_filename='all_teams_detection.csv'
    ):
    submissions_path = Path(submissions_root)
    all_teams = []

    METRICS = [
        'AP',
        'length_rmse',
        'width_rmse',
        'heading_rmse',
        'class_f1'
    ]

    for team_folder in submissions_path.iterdir():
        if not team_folder.is_dir():
            continue

        team_name = team_folder.name
        if team_name == "output_log":
            continue

        input_file = team_folder / input_filename
        if not input_file.exists():
            logging.warning(f"[WARN] {team_name}: Missing {input_filename}")
            continue

        try:
            df = pd.read_csv(input_file)
            if df.empty:
                logging.warning(f"[WARN] {team_name}: Empty detection result")
                continue

            # keep only valid per-image rows
            df = df[df['file_name'].notna()]

            grouped = df.groupby('file_name')[METRICS].mean()

            # flatten column names -> metric_team
            grouped.columns = [f"{m}_{team_name}" for m in grouped.columns]

            all_teams.append(grouped)

            logging.info(f"[OK] Processed detection metrics for {team_name}")

        except Exception as e:
            logging.error(f"[ERROR] Detection processing failed for {team_name}: {e}")

    if not all_teams:
        logging.warning("⚠️ No detection results processed.")
        return

    # merge all teams on file_name
    final_df = pd.concat(all_teams, axis=1)
    final_df.index.name = 'file_name'
    final_df.reset_index(inplace=True)

    output_path = submissions_path / output_filename
    final_df.to_csv(output_path, index=False)

    logging.info(f"✅ Detection metrics table saved to: {output_path}")



def handle_inf_rmse(series, penalty=1e6):
    series_clean = series.copy()
    finite_max = series[np.isfinite(series)].max() if np.any(np.isfinite(series)) else None
    
    if finite_max is not None:
        replacement = max(penalty, finite_max * 10)
    else:
        replacement = penalty
    
    series_clean[~np.isfinite(series_clean)] = replacement
    return series_clean



def normalize(series, reverse=False):
    try:
        if series.empty:
            return series  # return empty as is
        if reverse:  # lower is better (e.g., RMSE)
            series = handle_inf_rmse(series)
            denom = series.max() - series.min()
            if denom == 0:
                return pd.Series(1.0, index=series.index)
            return (series.max() - series) / denom
        else:  # higher is better (e.g., F1, mAP)
            denom = series.max() - series.min()
            if denom == 0:
                return pd.Series(1.0, index=series.index)
            return (series - series.min()) / denom
    except Exception as e:
        logging.error(f"[ERROR] Normalization failed: {e}")
        return series





def generate_final_leaderboard(submissions_root: str,
        output_filename='final_leaderboard.csv'
    ):
    submissions_path = Path(submissions_root)
    
    WEIGHTS = {
        'f1': 0.3,
        'rmse_i': 0.15,
        'map': 0.4,
        'rmse_p': 0.15
    }

    
    FILE_NAMES = {
        'f1': 'AIS_correlation_results.csv',
        'rmse_i': 'interpolated_data_results.csv',
        'map': 'detection_results.csv',
        'rmse_p': 'path_prediction_results.csv'
    }
    
    rows = []

    if not submissions_path.exists() or not submissions_path.is_dir():
        logging.error(f"[ERROR] Submissions root path {submissions_root} does not exist or is not a directory.")
        return

    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name=="output_log": continue
            try:
                f1_path = team_folder / FILE_NAMES['f1']
                rmse_i_path = team_folder / FILE_NAMES['rmse_i']
                map_path = team_folder / FILE_NAMES['map']
                rmse_p_path = team_folder / FILE_NAMES['rmse_p']

                
                f1_csv = pd.read_csv(f1_path)
                f1 = f1_csv['f1_score'].iloc[0]
                
                rmse_i_csv = pd.read_csv(rmse_i_path)
                rmse_i = rmse_i_csv['consolidated_rmse'].iloc[0]
                
                map_csv = pd.read_csv(map_path)
                map_score = map_csv['mAP'].iloc[0]
                rmse_l = map_csv['mAP'].iloc[0]

                rmse_p_csv = pd.read_csv(rmse_p_path)
                rmse_p = rmse_p_csv['rmse'].iloc[0]

                if None in (f1, rmse_i, map_score):
                    logging.error(f"[WARN] Skipping {team_name} due to missing one or more scores.")
                    continue

                rows.append({
                    'team_name': team_name,
                    'f1_score': f1,
                    'rmse_i': rmse_i,
                    'mAP': map_score,
                    'rmse_p': rmse_p
                })

            except Exception as e:
                logging.error(f"[ERROR] Failed processing team '{team_name}': {e}")

    if not rows:
        logging.error("[WARN] No valid teams found with complete scores.")
        return

    df = pd.DataFrame(rows)

    try:
        # Normalize the scores
        df['norm_f1'] = normalize(df['f1_score'])
        df['norm_rmse_i'] = normalize(df['rmse_i'], reverse=True)
        df['norm_mAP'] = normalize(df['mAP'])
        df['norm_rmse_p'] = normalize(df['rmse_i'], reverse=True)

        # Calculate weighted final score
        df['final_score'] = (
            df['norm_f1'] * WEIGHTS['f1'] +
            df['norm_rmse_i'] * WEIGHTS['rmse_i'] +
            df['norm_mAP'] * WEIGHTS['map'] +
            df['norm_rmse_p'] * WEIGHTS['rmse_i']
        ).round(10)

        # Sort and assign dense ranks
        df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        df['rank'] = df['final_score'].rank(method='dense', ascending=False).astype(int)
        df['percentile'] = np.round(100 * (1 - (df['rank'] - 1) / (df['rank'].max() - 1)),6)
        
        # Save output CSV
        output_path = submissions_path / output_filename
        df.to_csv(output_path, index=False)
        logging.info(f"\n✅ Final leaderboard saved to: {output_path}")

    except Exception as e:
        logging.error(f"[ERROR] Failed during normalization/ranking/saving: {e}")



# def extract_team_logs(csv_files, save_dir):
#     output_folder = os.path.join(save_dir,"output_log/")
#     os.makedirs(output_folder,exist_ok=True)
    
#     col_names = {
#         'correlation':'F1',
#         'interpolation':'RMSE',
#         'detection':'AP'
#     }
    
#     submissions_path = Path(save_dir)
#     for team_folder in submissions_path.iterdir():
#         if team_folder.is_dir():
#             team_name = team_folder.name
#             if team_name=="output_log": continue
            
#             output_xlsx = os.path.join(output_folder,f"{team_name}_output.xlsx")
#             print(output_xlsx)
            
#             with pd.ExcelWriter(output_xlsx, engine='openpyxl', mode='a' if os.path.exists(output_xlsx) else 'w') as writer:
#                 for idx, csv_file in enumerate(csv_files, 1):
#                     try:
#                         csv_file_paths = os.path.join("results/",csv_file)
#                         df = pd.read_csv(csv_file_paths)
#                         first_col = df.columns[0]

#                         if team_name not in df.columns:
#                             logging.error(f"⚠️ {team_name} not found in {csv_file}, skipping.")
#                             continue
                        
#                         sheet_name = csv_file[:-4].split("_")[-1]
#                         df_selected = df[[first_col, team_name]].rename(columns={first_col: 'file_name', team_name: col_names[sheet_name]})
                        
#                         df_selected.to_excel(writer, sheet_name=sheet_name, index=False)
#                         #logging.info(f"Wrote {csv_file} → {team_name}_submission.xlsx ({sheet_name})")

#                     except Exception as e:
#                         logging.error(f"❌ Error processing {csv_file} for {team_name}: {e}")

def extract_team_logs(csv_files, save_dir):
    output_folder = os.path.join(save_dir, "output_log/")
    os.makedirs(output_folder, exist_ok=True)

    submissions_path = Path(save_dir)

    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name == "output_log":
                continue

            all_logs = []

            for csv_file in csv_files:
                csv_path = os.path.join(save_dir, csv_file)
                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    first_col = df.columns[0]

                    if team_name not in df.columns:
                        continue

                    df_selected = df[[first_col, team_name]]
                    df_selected["source"] = csv_file
                    all_logs.append(df_selected)

                except Exception as e:
                    logging.error(f"Error reading {csv_file} for {team_name}: {e}")

            if len(all_logs) == 0:
                logging.warning(f"No logs for {team_name}")
                continue

            final_df = pd.concat(all_logs, ignore_index=True)
            output_csv = os.path.join(output_folder, f"{team_name}_log.csv")
            final_df.to_csv(output_csv, index=False)
            print("Saved:", output_csv)







def evaluate_all_submissions(submissions_folder: str, ground_truth_folder: str, save_dir: str, tasks: str):
    submission_path = Path(submissions_folder)
    ground_truth_path = Path(ground_truth_folder)
    save_dir = Path(save_dir)

    if not submission_path.exists():
        logging.error(f"Submission folder does not exist: {submissions_folder}")
        return

    if not ground_truth_path.exists():
        logging.error(f"Ground truth folder does not exist: {ground_truth_folder}")
        return
    
    if not save_dir.exists():
        os.makedirs(save_dir,exist_ok=True)
        logging.info(f"Folder created for results")
    
    logging.info(f"Evaluating zip files..")
    for zip_file in submission_path.glob('*.zip'):
        evaluate_team(zip_file, save_dir, ground_truth_path)
        
    
    logging.info("collating scores for all teams")
    generate_all_teams_AIS(save_dir)
    process_interpolation(save_dir)
    process_detection(save_dir)
    process_path(save_dir)
    
    
    csv_files_info = ["all_teams_AIS_correlation.csv","all_teams_interpolation.csv","all_teams_detection.csv", "all_teams_path.csv"]
    extract_team_logs(csv_files_info, save_dir)

    logging.info("Final score generation for all teams..")
    generate_final_leaderboard(save_dir)
    
    
    
        
        
def main(args):
    
    if os.path.exists('results/'):
        shutil.rmtree('results/')
    
    logging.info(f"Evaluating submissions for PS09")
    evaluate_all_submissions(args.subs_dir, args.gt_dir, args.out_dir, args.tasks)
    logging.info(f"Evaluation complete. Logs saved to PS_09_logs")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate PS09.")
    parser.add_argument("--subs_dir", required=False, default="./submissions", help="Directory containing submission zip files")
    parser.add_argument("--gt_dir", required=False, default="./gt", help="Directory containing ground-truth")
    parser.add_argument("--out_dir", required=False, default="./results", help="Directory to store results")
    parser.add_argument(
    "--tasks",
    default="ais,interpolation,detection,path",
    help="Comma separated: ais, interpolation, detection, path"
    )

    args = parser.parse_args()
    
    main(args)