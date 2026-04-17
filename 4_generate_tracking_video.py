import cv2
import os
import glob
import sys

frames_dir = "frames"
template_path = "template.png"
output_video = "tracking_video.mp4"

# ==========================================
# Available template matching methods
# ==========================================

methods = {
    "TM_CCOEFF": cv2.TM_CCOEFF,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR": cv2.TM_CCORR,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_SQDIFF": cv2.TM_SQDIFF,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
}

# ==========================================
# Ask user for method
# ==========================================

print("Available template matching methods:\n")

for i, name in enumerate(methods.keys(), start=1):
    print(f"{i} - {name}")

choice = input("\nSelect the method number: ").strip()

if not choice.isdigit() or int(choice) < 1 or int(choice) > len(methods):
    print("Invalid selection.")
    sys.exit(1)

method_name = list(methods.keys())[int(choice) - 1]
method_flag = methods[method_name]

print(f"\nSelected method: {method_name}")

# ==========================================
# Load template
# ==========================================

if not os.path.exists(template_path):
    print("Error: template.png not found.")
    print("Run 2_create_template.py first.")
    sys.exit(1)

template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

if template is None:
    print("Error: Failed to load template.")
    sys.exit(1)

w, h = template.shape[::-1]

# ==========================================
# Load frame list
# ==========================================

frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

if len(frame_files) == 0:
    print("Error: No frames found in 'frames' directory.")
    print("Run 1_extract_frames.py first.")
    sys.exit(1)

# ==========================================
# Configure video writer
# ==========================================

first_frame = cv2.imread(frame_files[0])

if first_frame is None:
    print("Error: Could not read first frame.")
    sys.exit(1)

height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

print("\nGenerating tracking video...")

# ==========================================
# Process frames
# ==========================================

for path in frame_files:

    img_color = cv2.imread(path)

    if img_color is None:
        print(f"Warning: Could not read {path}. Skipping.")
        continue

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, template, method_flag)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Determine best match location
    if method_name in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
        top_left = min_loc
        score = min_val
    else:
        top_left = max_loc
        score = max_val

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw bounding box
    cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)

    # Overlay text
    cv2.putText(
        img_color,
        f"Method: {method_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img_color,
        f"Frame: {os.path.basename(path)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img_color,
        f"Score: {score:.4f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    out.write(img_color)

out.release()

print(f"\nTracking video successfully saved as: {output_video}")