import cv2
import os
import pandas as pd
import glob
import shutil
import sys
import matplotlib.pyplot as plt

frames_dir = "frames"
template_path = "template.png"
csv_output_dir = "matching_results"
plots_output_dir = "plots"

# ==========================================
# Prepare output directories
# ==========================================

for directory in [csv_output_dir, plots_output_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# ==========================================
# Load frames
# ==========================================

frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

if len(frame_files) == 0:
    print("Error: No frames found in 'frames' directory.")
    print("Run 1_extract_frames.py first.")
    sys.exit(1)

# ==========================================
# Load template
# ==========================================

if not os.path.exists(template_path):
    print("Error: template.png not found.")
    print("Run 2_create_template.py before this step.")
    sys.exit(1)

template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

if template is None:
    print("Error: Could not load template image.")
    sys.exit(1)

print("Template loaded successfully.")

# ==========================================
# Template matching methods
# ==========================================

methods = {
    'TM_CCOEFF': cv2.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
    'TM_CCORR': cv2.TM_CCORR,
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    'TM_SQDIFF': cv2.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
}

results = {meth: [] for meth in methods.keys()}

print("Processing frames and computing template matching scores...")

# ==========================================
# Process frames
# ==========================================

for frame_path in frame_files:

    frame_name = os.path.basename(frame_path)

    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Warning: Could not read {frame_name}. Skipping.")
        continue

    for meth_name, meth_flag in methods.items():

        res = cv2.matchTemplate(img, template, meth_flag)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        results[meth_name].append({
            'frame': frame_name,
            'min_val': min_val,
            'max_val': max_val
        })

# ==========================================
# Export CSV files
# ==========================================

print("Exporting CSV files...")

for meth_name, data in results.items():

    df = pd.DataFrame(data)

    csv_path = os.path.join(csv_output_dir, f"{meth_name}.csv")

    df.to_csv(csv_path, index=False, sep=';', decimal=',')

    print(f"CSV generated: {csv_path}")

print("CSV generation completed.")

# ==========================================
# Generate plots
# ==========================================

print("Generating graphs...")

for meth_name in methods.keys():

    csv_file = os.path.join(csv_output_dir, f"{meth_name}.csv")

    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Skipping.")
        continue

    df = pd.read_csv(csv_file, sep=';', decimal=',')

    # Extract frame number
    df['frame_index'] = df['frame'].str.extract('(\\d+)').astype(int)

    plt.figure(figsize=(10,6))

    plt.plot(df['frame_index'], df['min_val'], label='min_val', linewidth=1.5)
    plt.plot(df['frame_index'], df['max_val'], label='max_val', linewidth=1.5)

    plt.title(f"Method Response: {meth_name}")
    plt.xlabel("Frame Number")
    plt.ylabel("Response Value")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(plots_output_dir, f"{meth_name}_plot.png")

    plt.savefig(save_path)
    plt.close()

    print(f"Graph saved: {save_path}")

print("All graphs generated successfully.")