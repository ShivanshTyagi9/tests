import pandas as pd
import re

file_a_path = "AIS_2025_05_29_T160509_RPK.csv" 
file_b_path = "Layer 1.csv"


df_a = pd.read_csv(file_a_path)
df_b = pd.read_csv(file_b_path)

ais_lookup = {}

for _, row in df_a.iterrows():
    vid = row.get("vessel_id")

    try:
        vid = int(vid)
    except:
        continue
    ais_lookup[str(vid)] = row


#main part
def build_description(row):
    def safe(val):
        if pd.isna(val):
            return ""
        return str(val)

    category = safe(row.get("category"))
    MMSI = safe(row.get("MMSI"))
    vessel_name = safe(row.get("vessel_name"))
    vessel_class = "" 

    length = safe(row.get("length"))
    width = safe(row.get("width"))
    heading = safe(row.get("heading"))
    cog = safe(row.get("cog"))
    sog = safe(row.get("sog"))

    parts = []
    if length: parts.append(f"L-{length}")
    if width: parts.append(f"W-{width}")
    if heading: parts.append(f"H-{heading}")
    if cog: parts.append(f"C-{cog}")
    if sog: parts.append(f"S-{sog}")

    lwchs_str = "_".join(parts)

    return (
        f"ship_{category}_{MMSI}_{vessel_name}_{vessel_class}_L-{length}_W-{width}_H-{heading}_C-{cog}_S-{sog}_source"
        #f"{lwchs_str}_source"
    )


#matching vessel_<number>
vessel_pattern = re.compile(r"^vessel_(\d+)$")

for idx, row in df_b.iterrows():
    name = str(row["Name"])

    match = vessel_pattern.match(name)
    if not match:
        continue  #leave length and width 

    vessel_id = match.group(1)

    if vessel_id in ais_lookup:
        desc = build_description(ais_lookup[vessel_id])
    else:
        desc = "ship_unverified_________visual"

    df_b.at[idx, "Description"] = desc


df_b.to_csv(file_b_path, index=False)

print("File B updated in-place.")
