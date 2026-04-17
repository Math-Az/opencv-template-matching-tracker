import cv2
import os
import sys

FRAME_PATH = "frames/frame_0000.png"
TEMPLATE_PATH = "template.png"

def main():
    if not os.path.exists(FRAME_PATH):
        print("Error: frames/frame_0000.png not found.")
        print("Run 1_extract_frames.py before creating the template.")
        sys.exit(1)

    print("Loading first frame...")
    image = cv2.imread(FRAME_PATH)

    if image is None:
        print("Error: Failed to load the frame image.")
        sys.exit(1)

    print("Select the template region using the mouse.")
    print("Drag a rectangle and press ENTER or SPACE to confirm.")
    print("Press C to cancel selection.")

    roi = cv2.selectROI("Select Template Region", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("No region selected. Exiting.")
        sys.exit(1)

    template = image[y:y+h, x:x+w]

    cv2.imwrite(TEMPLATE_PATH, template)

    print(f"Template saved as '{TEMPLATE_PATH}'.")
    print(f"Template size: {w} x {h} pixels")

if __name__ == "__main__":
    main()