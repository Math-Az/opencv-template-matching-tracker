import cv2
import os
import shutil

# Ask user for video name
video_name = input("Enter the video name (without extension): ").strip()
video_path = f"{video_name}.mp4"

output_folder = "frames"

# Check if video exists
if not os.path.isfile(video_path):
    print(f"Error: video '{video_path}' was not found.")
    exit(1)

# Prepare frames directory
if os.path.exists(output_folder):
    print("Cleaning existing 'frames' directory...")
    shutil.rmtree(output_folder)

os.makedirs(output_folder)

print(f"Opening video: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: could not open the video.")
    exit(1)

frame_count = 0

print("Extracting frames and converting to grayscale...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")

    cv2.imwrite(filename, gray_frame)

    frame_count += 1

cap.release()

print(f"Done. {frame_count} frames were extracted to the '{output_folder}' folder.")