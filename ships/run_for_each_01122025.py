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
import warnings
import shutil

warnings.filterwarnings("ignore")
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
    
    
    #logging.info(f"{len(merged)}")
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
    
    #logging.info(f"merged len = ,{len(merged)}")
    
    merged['point_latitude_pred'].fillna(pd.NA, inplace=True)
    merged['point_longitude_pred'].fillna(pd.NA, inplace=True)
    
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
    
    distances = distances.fillna(1000)
    
    #print(distances[:5])
    
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
    #logging.info(f"rmse {rmse}")
    if not((distances==1000.0).all()):
        rmse_fields[str(1)]["consolidated_rmse"] = np.round(rmse,6)
    else:
        rmse_fields[str(1)]["consolidated_rmse"] = np.inf
        
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
          'bbox': wkt.loads(ann['bbox']),
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
        gt_bboxes = [gt['bbox'] for gt in gt_anns]
        gt_used = [False]*len(gt_bboxes)

        preds = sorted(pred_anns, key=lambda x: x.get('score',1.0), reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, pred in enumerate(preds):
            pred_poly = pred['bbox']

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






def evaluate_detection(gt_data, pred_json_path):
    
    pred_data = load_data(pred_json_path)

    gt_by_image = organize_annotations(gt_data)
    pred_by_image = organize_annotations(pred_data)
    
    det_results = {}
    #print(pred_json_path)
    ap_list = []
    for img_id, img_info in enumerate(gt_data['images']):
        #print(img_id, img_info)
        
        gt_image_id = img_info['id']
        gt_file_name = img_info['file_name']
        
        gt_anns = gt_by_image.get(gt_image_id, [])
        det_results[str(img_id + 1)] = {'id': gt_image_id, 'file_name': gt_file_name}
        
        
        # Checking for a matching image in pred_data['images']
        # pred_img.get('file_name') == gt_file_name)
        
        
        matched_pred_img = next(
            (pred_img for pred_img in pred_data['images']
             if int(pred_img.get('id')) == gt_image_id and os.path.splitext(pred_img.get('file_name').lower())[0] == os.path.splitext(gt_file_name.lower())[0]),
            None
        )
        
        if not matched_pred_img:
            print(f"[WARNING] No matching prediction image found for GT image id={gt_image_id}, file_name={gt_file_name}")
            det_results[str(img_id + 1)]['AP'] = 0.0
            ap_list.append(0.0)
            continue
        
        pred_anns = pred_by_image.get(gt_image_id, [])
        # Reset 'used' flag for ground truth annotations
        for ann in gt_anns:
            ann['used'] = False

        # Compute the average precision for this image
        ap = np.round(compute_custom_vary_ap(gt_anns, pred_anns), 6)
        det_results[str(img_id + 1)]['AP'] = ap
        ap_list.append(ap)
    
    mAP = np.round(np.mean(ap_list),6) if ap_list else 0.0
    det_results[str(1)]['mAP']=mAP
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











def find_files_with_word(directory, words, extension):
    matching_files = []
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        
        dirs[:] = [d for d in dirs if 'MACOSX' not in d]
        
        for file in files:
            if file.lower().endswith(extension.lower()):
                if any(word.lower() in file.lower() for word in words):
                    matching_files.append(os.path.join(root, file))
    
    return matching_files[0]



def extract_zip(zip_path, destination_folder):
    """Extracts a zip file into the specified folder."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        logging.info(f"Extracted {zip_path} into {destination_folder}")
    except zipfile.BadZipFile:
        logging.info(f"Error: {zip_path} is not a valid ZIP file.")
    except PermissionError:
        logging.info(f"Permission error: Cannot extract {zip_path}")
    except Exception as e:
        logging.info(f"Error extracting {zip_path}: {e}")

def extract_nested_zips(directory, extracted_zips):
    """Recursively searches for zip files in the directory and extracts them once."""
    for root, _, files in os.walk(directory):
        for file in files:
            if str(file).lower().endswith('.zip'):
                zip_path = os.path.join(root, file)

                # Avoid reprocessing the same zip file
                if zip_path in extracted_zips:
                    logging.info(f"Nested zip file : ,{zip_path}")
                    continue
                
                extracted_zips.add(zip_path)

                extract_folder = os.path.join(root, os.path.splitext(file)[0])
                if not os.path.exists(extract_folder):
                    os.mkdir(extract_folder)

                extract_zip(zip_path, extract_folder)
                extract_nested_zips(extract_folder, extracted_zips)


def extract_zip_in_named_folder(zip_file_path, destination_folder):
    """Extract a zip file into a named folder and handle nested zips."""
    if not os.path.exists(zip_file_path):
        logging.info(f"Error: {zip_file_path} does not exist.")
        return

    if not str(zip_file_path).lower().endswith('.zip'):
        logging.info(f"Skipped non-zip file: {zip_file_path}")
        return

    extracted_zips = set()
    extracted_zips.add(zip_file_path)
    #logging.info(f"{zip_file_path}, {destination_folder}")
    extract_zip(zip_file_path, destination_folder)
    extract_nested_zips(destination_folder, extracted_zips)





def evaluate_team(team_zip: Path, extract_dir: Path, ground_truth_path: Path):
    team_name = team_zip.stem
    team_folder = extract_dir / team_name
    os.makedirs(team_folder, exist_ok=True)
    print(team_zip,extract_dir,team_folder)
    team_log = ""
    
    try:
        logging.info(f"")
        logging.info(f"Loading ground-truth..")
        gt_ais_csv = pd.read_csv(os.path.join(ground_truth_path, 'AIS_correlation.csv'))
        gt_inter_csv = pd.read_csv(os.path.join(ground_truth_path, 'interpolated_data.csv'))
        gt_detect = load_data(os.path.join(ground_truth_path, 'detection.json'))
    except Exception as e:
        logging.error(f"Ground truth loading failed : {e}")
    
    try:
        
        try:
            extract_zip_in_named_folder(team_zip, team_folder)
            #os.remove(os.path.join(team_folder,'AIS_correlation_results.csv'))
            #os.remove(os.path.join(team_folder,'interpolated_data_results.csv'))
            #os.remove(os.path.join(team_folder,'detection_results.csv'))
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
        
        try:
            target_string = ['correlation']
            expected_extension = ".csv"
            ais_pred_path = find_files_with_word(current_team_path, target_string, expected_extension)
            logging.info(ais_pred_path)
            _, actual_extension = os.path.splitext(ais_pred_path)
            
            assert actual_extension == expected_extension, f"File extension mismatch: Expected '{expected_extension}', got '{actual_extension}'"
        except Exception as e:
            empty_df = empty_csv_creator('correlation',gt_ais_csv)
            output_csv = team_folder / 'AIS_correlation_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - correlation file not found: {e}")
            team_log+=f"correlation file not found: {e}\n"
            
        try:
            target_string = ['interpolated','interpolation']
            expected_extension = ".csv"
            interpolation_pred_path = find_files_with_word(current_team_path,target_string, expected_extension)
            logging.info(interpolation_pred_path)
            _, actual_extension = os.path.splitext(interpolation_pred_path)
            
            assert actual_extension == expected_extension, f"File extension mismatch: Expected '{expected_extension}', got '{actual_extension}'"
        except Exception as e:
            empty_df = empty_csv_creator('interpolation',gt_inter_csv)
            output_csv = team_folder / 'interpolated_data_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - interpolation file not found: {e}")
            team_log+=f"interpolation file not found: {e}\n"
            
        try:
            target_string = ['detection']
            expected_extension = ".json"
            detection_pred_path = find_files_with_word(current_team_path,target_string,expected_extension)
            logging.info(detection_pred_path)
            _, actual_extension = os.path.splitext(detection_pred_path)
            
            assert actual_extension == expected_extension, f"File extension mismatch: Expected '{expected_extension}', got '{actual_extension}'"
        except Exception as e:
            empty_df = empty_csv_creator("",gt_detect)
            output_csv = team_folder / 'detection_results.csv'
            empty_df.to_csv(output_csv, index=False)
            logging.error(f"{team_name} - detection file not found: {e}")
            team_log+=f"detection file not found: {e}\n"
            
        

        
        # Paths to team predictions (modify if needed)
        #ais_pred_path = team_folder / 'AIS_correlation.csv'
        #interpolation_pred_path = team_folder / 'interpolated_data.csv'
        #detection_pred_path = team_folder / 'detection.json'

        # AIS Correlation
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



def process_interpolation(submissions_root,
        input_filename='interpolated_data_results.csv',
        output_filename='all_teams_interpolation.csv'
    ):
    submissions_path = Path(submissions_root)
    all_teams = []

    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name=="output_log": continue
            input_file = team_folder / input_filename

            if not input_file.exists():
                logging.error(f"[WARN] {team_name}: Missing {input_filename}")
                continue

            try:
                df = pd.read_csv(input_file)
                if df.empty:
                    logging.error(f"[WARN] {team_name}: Empty interpolation result")
                    continue

                # Group by path_id and calculate RMSE
                rmse_per_path = df.groupby('path_id')['haversine_dist'].apply(
                    lambda x: np.sqrt((x**2).mean())
                )
                rmse_series = pd.Series(rmse_per_path, name=team_name)
                all_teams.append(rmse_series)

                logging.info(f"[OK] Processed interpolation for {team_name}")

            except Exception as e:
                logging.error(f"[ERROR] Interpolation processing failed for {team_name}: {e}")

    if all_teams:
        final_df = pd.concat(all_teams, axis=1)
        final_df.index.name = 'path_id'
        final_df.reset_index().to_csv(submissions_path / output_filename, index=False)
        logging.info(f"✅ Interpolation RMSE table saved to: {output_filename}")
    else:
        logging.info("\n⚠️ No interpolation results processed.")



def process_detection(submissions_root,
        input_filename='detection_results.csv',
        output_filename='all_teams_detection.csv'
    ):
    submissions_path = Path(submissions_root)
    all_teams = []

    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name=="output_log": continue
            input_file = team_folder / input_filename

            if not input_file.exists():
                logging.error(f"[WARN] {team_name}: Missing {input_filename}")
                continue

            try:
                df = pd.read_csv(input_file)
                if df.empty:
                    logging.error(f"[WARN] {team_name}: Empty detection result")
                    continue

                ap_per_file = df.groupby('file_name')['AP'].mean()
                ap_series = pd.Series(ap_per_file, name=team_name)
                all_teams.append(ap_series)

                logging.info(f"[OK] Processed detection for {team_name}")

            except Exception as e:
                logging.error(f"[ERROR] Detection processing failed for {team_name}: {e}")

    if all_teams:
        final_df = pd.concat(all_teams, axis=1)
        final_df.index.name = 'file_name'
        final_df.reset_index().to_csv(submissions_path / output_filename, index=False)
        logging.info(f"✅ Detection AP table saved to: {output_filename}")
    else:
        logging.info("\n⚠️ No detection results processed.") 


'''
def handle_inf_rmse(series, penalty=1e6):
    series_clean = series.copy()
    finite_max = series[np.isfinite(series)].max() if np.any(np.isfinite(series)) else None
    
    if finite_max is not None:
        replacement = max(penalty, finite_max * 10)
    else:
        replacement = penalty
    
    series_clean[~np.isfinite(series_clean)] = replacement
    return series_clean
'''

def handle_inf_rmse(series, penalty=1e6, upper_threshold=100, reverse=True):
    series_clean = series.replace(np.inf, penalty)
    series_clean = series_clean.fillna(penalty)
    
    # Normalize to [0, 0.8] range
    below_threshold = series_clean[series_clean <= upper_threshold]
    if not below_threshold.empty:
        min_val = below_threshold.min()
        max_val = below_threshold.max()
        denom = max_val - min_val
        if denom == 0:
            below_threshold_normalized = pd.Series(0.4, index=below_threshold.index)  # Middle value
        else:
            below_threshold_normalized = (below_threshold - min_val) / denom * 0.8
    
    # Normalize to [0.9, 1.0] range
    above_threshold = series_clean[series_clean > upper_threshold]
    if not above_threshold.empty:
        min_val = above_threshold.min()
        max_val = above_threshold.max()
        denom = max_val - min_val
        if denom == 0:
            above_threshold_normalized = pd.Series(0.95, index=above_threshold.index)  # Middle value
        else:
            above_threshold_normalized = 0.9 + (above_threshold - min_val) / denom * 0.1
    
    if above_threshold.empty:
        series_normalized = below_threshold_normalized
    else:
        series_normalized = pd.concat([below_threshold_normalized, above_threshold_normalized])
    
    if reverse:  # Lower RMSE is better
        return 1 - series_normalized 
    return series_normalized




def normalize(series, reverse=False):
    try:
        if series.empty:
            return series  # return empty as is
        if reverse:  # lower is better (e.g., RMSE)
            series = handle_inf_rmse(series, reverse=True)
            return series
        else:  # higher is better (e.g., F1, mAP)
            denom = series.max() - series.min()
            if denom == 0:
                return pd.Series(1.0, index=series.index)
            return (series - series.min()) / denom
    except Exception as e:
        logging.error(f"[ERROR] Normalization failed: {e}")
        return series




def generate_final_leaderboard(submissions_root: str,
        ground_truth_path: str,
        output_filename='final_leaderboard.csv'
    ):
    submissions_path = Path(submissions_root)
    
    WEIGHTS = {
        'f1': 0.3,
        'rmse': 0.2,
        'map': 0.5
    }
    
    team_names_csv = pd.read_csv(os.path.join(ground_truth_path,'team_names.csv'))
    team_names = team_names_csv.set_index('Startup_Id')['Startup'].to_dict()
    
    FILE_NAMES = {
        'f1': 'AIS_correlation_results.csv',
        'rmse': 'interpolated_data_results.csv',
        'map': 'detection_results.csv'
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
                rmse_path = team_folder / FILE_NAMES['rmse']
                map_path = team_folder / FILE_NAMES['map']
                
                f1_csv = pd.read_csv(f1_path)
                f1 = f1_csv['f1_score'].iloc[0]
                
                rmse_csv = pd.read_csv(rmse_path)
                rmse = rmse_csv['consolidated_rmse'].iloc[0]
                
                map_csv = pd.read_csv(map_path)
                map_score = map_csv['mAP'].iloc[0]

                if None in (f1, rmse, map_score):
                    logging.error(f"[WARN] Skipping {team_name} due to missing one or more scores.")
                    continue

                rows.append({
                    'team_name': team_name,
                    'team_desc':team_names[team_name],
                    'f1_score': f1,
                    'rmse_km': rmse,
                    'mAP': map_score
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
        df['norm_rmse'] = normalize(df['rmse_km'], reverse=True)
        df['norm_mAP'] = normalize(df['mAP'])

        # Calculate weighted final score
        df['final_score'] = (
            df['norm_f1'] * WEIGHTS['f1'] +
            df['norm_rmse'] * WEIGHTS['rmse'] +
            df['norm_mAP'] * WEIGHTS['map']
        ).round(10)

        # Sort and assign dense ranks
        df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
        df['rank'] = df['final_score'].rank(method='dense', ascending=False).astype(int)
        #df['percentile'] = np.round(100 * (1 - (df['rank'] - 1) / (df['rank'].max() - 1)),6)
        
        # Save output CSV
        output_path = submissions_path / output_filename
        df.to_csv(output_path, index=False)
        logging.info(f"Total Submissions: {len(df)-1}")
        logging.info(f"\n✅ Final leaderboard saved to: {output_path}")

    except Exception as e:
        logging.error(f"[ERROR] Failed during normalization/ranking/saving: {e}")



def extract_team_logs(csv_files, save_dir):
    output_folder = os.path.join(save_dir,"output_log/")
    os.makedirs(output_folder,exist_ok=True)
    
    col_names = {
        'correlation':'F1',
        'interpolation':'RMSE',
        'detection':'AP'
    }
    
    submissions_path = Path(save_dir)
    for team_folder in submissions_path.iterdir():
        if team_folder.is_dir():
            team_name = team_folder.name
            if team_name=="output_log": continue
            
            output_xlsx = os.path.join(output_folder,f"{team_name}_output.xlsx")
            print(output_xlsx)
            
            with pd.ExcelWriter(output_xlsx, engine='openpyxl', mode='a' if os.path.exists(output_xlsx) else 'w') as writer:
                for idx, csv_file in enumerate(csv_files, 1):
                    try:
                        csv_file_paths = os.path.join("results/",csv_file)
                        df = pd.read_csv(csv_file_paths)
                        first_col = df.columns[0]

                        if team_name not in df.columns:
                            logging.error(f"⚠️ {team_name} not found in {csv_file}, skipping.")
                            continue
                        
                        sheet_name = csv_file[:-4].split("_")[-1]
                        df_selected = df[[first_col, team_name]].rename(columns={first_col: 'file_name', team_name: col_names[sheet_name]})
                        
                        df_selected.to_excel(writer, sheet_name=sheet_name, index=False)
                        #logging.info(f"Wrote {csv_file} → {team_name}_submission.xlsx ({sheet_name})")

                    except Exception as e:
                        logging.error(f"❌ Error processing {csv_file} for {team_name}: {e}")





def evaluate_all_submissions(submissions_folder: str, ground_truth_folder: str, save_dir: str):
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
    #os.remove(os.path.join(save_dir,'all_teams_AIS_correlation.csv'))
    #os.remove(os.path.join(save_dir,'all_teams_detection.csv'))
    #os.remove(os.path.join(save_dir,'all_teams_interpolation.csv'))
    #os.remove(os.path.join(save_dir,'final_leaderboard.csv'))
                                    
    generate_all_teams_AIS(save_dir)
    process_interpolation(save_dir)
    process_detection(save_dir)
    
    
    csv_files_info = ["all_teams_AIS_correlation.csv","all_teams_interpolation.csv","all_teams_detection.csv"]
    extract_team_logs(csv_files_info, save_dir)

    logging.info("Final score generation for all teams..")
    generate_final_leaderboard(save_dir, ground_truth_path)
        
        
def main(args):
    
    if os.path.exists('results/'):
        shutil.rmtree('results/')
    
    logging.info(f"Evaluating submissions for PS09")
    evaluate_all_submissions(args.subs_dir, args.gt_dir, args.out_dir)
    logging.info(f"Evaluation complete. Logs saved to PS_09_logs")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate PS09.")
    parser.add_argument("--subs_dir", required=False, default="./submissions", help="Directory containing submission zip files")
    parser.add_argument("--gt_dir", required=False, default="./gt", help="Directory containing ground-truth")
    parser.add_argument("--out_dir", required=False, default="./results", help="Directory to store results")
    args = parser.parse_args()
    
    main(args)