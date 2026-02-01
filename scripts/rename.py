import pandas as pd

# Load CSV
df = pd.read_csv("Layer 1.csv")

vessel_count = 0
polyline_count = 0

new_names = []

for name in df["Name"]:
    name = name.lower()

    if "rectangle" in name:
        vessel_count += 1
        polyline_count = 0
        new_names.append(f"vessel_{vessel_count}")

    elif "polyline" in name:
        polyline_count += 1
        if polyline_count == 1:
            new_names.append(f"vessel_{vessel_count}_length")
        elif polyline_count == 2:
            new_names.append(f"vessel_{vessel_count}_width")
        else:
            # in case more polylines appear unexpectedly
            new_names.append(f"vessel_{vessel_count}_polyline_{polyline_count}")

    else:
        new_names.append(name)  # untouched if something else

df["Name"] = new_names

# Save result
df.to_csv("output.csv", index=False)
