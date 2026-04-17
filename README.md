# Object Tracking via Template Matching (OpenCV)

A four-stage Python pipeline that performs object tracking in video sequences using OpenCV's `matchTemplate` function. The pipeline extracts frames from a video, defines a target template, evaluates all six OpenCV matching methods with comparative analysis, and renders an annotated tracking video.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Template Matching Methods](#template-matching-methods)
  - [TM\_CCOEFF](#1-tm_ccoeff)
  - [TM\_CCOEFF\_NORMED](#2-tm_ccoeff_normed)
  - [TM\_CCORR](#3-tm_ccorr)
  - [TM\_CCORR\_NORMED](#4-tm_ccorr_normed)
  - [TM\_SQDIFF](#5-tm_sqdiff)
  - [TM\_SQDIFF\_NORMED](#6-tm_sqdiff_normed)
- [Method Comparison Summary](#method-comparison-summary)
- [How to Choose the Best Method](#how-to-choose-the-best-method)
- [Requirements & Installation](#requirements--installation)
- [How to Run](#how-to-run)
- [Output Structure](#output-structure)
- [Limitations](#limitations)

---

## Project Overview

Template Matching is a classical computer vision technique for locating a predefined patch (the **template**) within a larger image. When applied frame-by-frame across a video sequence, it serves as a simple yet effective object tracker: for each frame, the algorithm slides the template over the entire image, computes a similarity (or dissimilarity) score at every position, and identifies the location with the best score as the object's current position.

This project implements a complete, reproducible tracking pipeline divided into four sequential scripts. It targets static-camera scenarios where a single object moves against a relatively stable background.

---

## Pipeline Architecture

```
[Input Video]
      │
      ▼
┌─────────────────────────────────────────┐
│  Script 1 — Extract Frames              │
│  Reads .mp4, converts each frame to     │
│  grayscale, saves as frames/frame_NNNN  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Script 2 — Create Template             │
│  Opens frame_0000, lets the user draw   │
│  a bounding box around the target       │
│  object → saves crop as template.png    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Script 3 — Evaluate Matching Methods   │
│  Runs all 6 OpenCV methods over every   │
│  frame; exports per-method CSVs and     │
│  min_val / max_val response plots       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Script 4 — Generate Tracking Video     │
│  User selects one method; the script    │
│  draws a bounding box on each frame and │
│  encodes the result as tracking_video   │
└─────────────────────────────────────────┘
                   │
                   ▼
        [tracking_video.mp4]
```

---

## Template Matching Methods

OpenCV's `cv2.matchTemplate(image, template, method)` slides the template over the input image and produces a response map `R(x, y)` where each value quantifies how well the template fits at that position. The six available methods differ in what they compute and in how the best match is interpreted.

The template is denoted **T(x', y')** and the image patch under evaluation is **I(x+x', y+y')**. For the _normed_ variants, T' and I' represent mean-subtracted (zero-mean) versions.

---

### 1. TM_CCOEFF

**Cross-Correlation with Mean Subtraction (Pearson-like)**

$$R(x,y) = \sum_{x',y'} \bigl[T'(x',y') \cdot I'(x+x',\, y+y')\bigr]$$

where:

$$T'(x',y') = T(x',y') - \frac{1}{w \cdot h}\sum_{x'',y''} T(x'',y'')$$

$$I'(x+x',y+y') = I(x+x',y+y') - \frac{1}{w \cdot h}\sum_{x'',y''} I(x+x'',y+y'')$$

By subtracting the local mean from both template and image patch, this method removes the effect of global brightness differences. **Higher R → better match.** Use `max_loc` as the detection result.

**Strengths:** Robust to additive illumination changes (e.g., uniform brightness shift).  
**Weaknesses:** Sensitive to scale; response magnitude depends on absolute pixel values, making thresholding across frames inconsistent.

---

### 2. TM_CCOEFF_NORMED

**Normalized Cross-Correlation Coefficient**

$$R(x,y) = \frac{\sum_{x',y'} T'(x',y') \cdot I'(x+x',y+y')}{\sqrt{\sum_{x',y'} T'(x',y')^2 \cdot \sum_{x',y'} I'(x+x',y+y')^2}}$$

This is the Pearson correlation coefficient between the template patch and the image patch. The result is always in **[-1, 1]**: a value of 1 means a perfect match, 0 means no correlation, and -1 means an inverted match.

**Higher R → better match.** Use `max_loc`.

**Strengths:** Invariant to both additive brightness and multiplicative contrast changes. Scores are directly comparable across frames and across different images, making thresholding reliable (e.g., reject match if `max_val < 0.8`).  
**Weaknesses:** Slightly more expensive to compute; still sensitive to rotation and scale.

> **This is generally the most recommended method for robust tracking.**

---

### 3. TM_CCORR

**Cross-Correlation (Raw)**

$$R(x,y) = \sum_{x',y'} T(x',y') \cdot I(x+x',\, y+y')$$

A simple dot product between the raw template and the image patch. **Higher R → better match.** Use `max_loc`.

**Strengths:** Computationally inexpensive.  
**Weaknesses:** Heavily biased toward bright regions. A patch with uniformly high pixel values will produce a large response regardless of structural similarity to the template. Unreliable in practice for object tracking.

---

### 4. TM_CCORR_NORMED

**Normalized Cross-Correlation**

$$R(x,y) = \frac{\sum_{x',y'} T(x',y') \cdot I(x+x',y+y')}{\sqrt{\sum_{x',y'} T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}$$

Normalizes the raw cross-correlation by the energy of both patches. Result is bounded in **[0, 1]** for non-negative images. **Higher R → better match.** Use `max_loc`.

**Strengths:** More stable than TM_CCORR; mitigates the brightness bias to some extent.  
**Weaknesses:** Without mean subtraction, it remains sensitive to additive brightness offsets. Generally outperformed by `TM_CCOEFF_NORMED`.

---

### 5. TM_SQDIFF

**Sum of Squared Differences**

$$R(x,y) = \sum_{x',y'} \bigl[T(x',y') - I(x+x',\, y+y')\bigr]^2$$

Measures the total squared pixel-wise error between the template and the image patch. A **perfect match produces R = 0**; larger values indicate poorer matches. Use `min_loc` as the detection result.

**Strengths:** Intuitive formulation; computationally simple.  
**Weaknesses:** Raw values are unbounded and depend heavily on image dimensions and absolute brightness, making cross-frame threshold comparison difficult.

---

### 6. TM_SQDIFF_NORMED

**Normalized Sum of Squared Differences**

$$R(x,y) = \frac{\sum_{x',y'} \bigl[T(x',y') - I(x+x',y+y')\bigr]^2}{\sqrt{\sum_{x',y'} T(x',y')^2 \cdot \sum_{x',y'} I(x+x',y+y')^2}}$$

Normalizes TM_SQDIFF by the product of patch energies. Result is bounded in **[0, 1]**: 0 = perfect match, 1 = maximum dissimilarity. Use `min_loc`.

**Strengths:** Normalized range makes thresholding feasible (e.g., discard if `min_val > 0.2`).  
**Weaknesses:** Like TM_SQDIFF, it is sensitive to brightness differences since it lacks mean subtraction.

---

## Method Comparison Summary

| Method | Score Direction | Best Match | Range | Mean-Subtracted | Normalized |
|---|---|---|---|---|---|
| TM_CCOEFF | ↑ Higher = better | `max_loc` | Unbounded | ✅ Yes | ❌ No |
| TM_CCOEFF_NORMED | ↑ Higher = better | `max_loc` | [-1, 1] | ✅ Yes | ✅ Yes |
| TM_CCORR | ↑ Higher = better | `max_loc` | Unbounded | ❌ No | ❌ No |
| TM_CCORR_NORMED | ↑ Higher = better | `max_loc` | [0, 1] | ❌ No | ✅ Yes |
| TM_SQDIFF | ↓ Lower = better | `min_loc` | Unbounded | ❌ No | ❌ No |
| TM_SQDIFF_NORMED | ↓ Lower = better | `min_loc` | [0, 1] | ❌ No | ✅ Yes |

---

## How to Choose the Best Method

Script 3 generates a response plot for each method, showing `min_val` and `max_val` across all frames. Use these plots to identify which method yields the cleanest, most stable signal:

**Look for these characteristics in a good method:**

1. **Stable signal line** — The best-match score (i.e., `max_val` for CCOEFF/CCORR methods, `min_val` for SQDIFF methods) should be relatively flat across frames, not erratic. Wild oscillations indicate that the method is sensitive to noise, lighting variation, or background clutter.

2. **High contrast between best and worst response** — A large gap between `min_val` and `max_val` means the method can clearly distinguish the target location from the rest of the image. A small gap suggests the match landscape is nearly uniform, which leads to unreliable localization.

3. **Bounded, interpretable values** — Normalized methods (`_NORMED`) produce values in [0, 1] or [-1, 1], which makes it straightforward to confirm whether a match is confident (e.g., `TM_CCOEFF_NORMED max_val > 0.85`). Unnormalized methods may inflate or deflate response values depending on image content, making threshold decisions harder.

4. **Monotonic behavior when object is present** — If the tracked object moves smoothly, the best-match score should remain consistently high (or consistently low for SQDIFF) throughout the sequence, without sudden drops. A sudden drop mid-sequence suggests the method lost the object, possibly because it is fooled by a visually similar background region.

**Practical decision guide:**

- If `TM_CCOEFF_NORMED max_val` stays consistently above ~0.8 and the `max_val` curve is smooth → **use TM_CCOEFF_NORMED**. This is the default recommendation.
- If the background is uniform and illumination is constant → `TM_CCORR_NORMED` may also work and is slightly faster.
- If you see the CCOEFF_NORMED curve dropping at specific frames while SQDIFF_NORMED `min_val` remains near 0 → consider **TM_SQDIFF_NORMED**.
- Avoid unnormalized methods (`TM_CCOEFF`, `TM_CCORR`, `TM_SQDIFF`) for quantitative comparison or thresholding; use them only when performance is critical and conditions are highly controlled.

---

## Requirements & Installation

**Python version:** 3.8 or higher

Install all dependencies with:

```bash
pip install opencv-python pandas matplotlib
```

Or using a `requirements.txt`:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python
pandas
matplotlib
numpy
```

---

## How to Run

All four scripts must be executed **in order** from the project root directory. Each script depends on the outputs of the previous one.

---

### Step 1 — Extract Frames

```bash
python 1_extract_frames.py
```

**What it does:** Prompts for the input video filename (without extension), reads the `.mp4` file, converts every frame to grayscale, and saves them sequentially to `frames/frame_0000.png`, `frames/frame_0001.png`, etc.

**Input:** A video file in the current directory (e.g., `myvideo.mp4`)  
**Output:** `frames/` directory containing all grayscale frames

```
Enter the video name (without extension): myvideo
```

> Ensure the video is at least 10 seconds long (≥ 300 frames at 30 fps) and has a minimum resolution of 640×480.

---

### Step 2 — Define the Template

```bash
python 2_create_template.py
```

**What it does:** Opens `frames/frame_0000.png` in an interactive window. Draw a bounding box around the object you want to track by clicking and dragging. Press **ENTER** or **SPACE** to confirm the selection. Press **C** to cancel.

**Input:** `frames/frame_0000.png`  
**Output:** `template.png` — the cropped target region

> Choose a region that is visually distinctive and well-separated from the background. Avoid selecting regions with repetitive textures or that resemble other parts of the scene.

---

### Step 3 — Evaluate All Matching Methods

```bash
python 3_match_methods.py
```

**What it does:** Loads all frames and the template, runs all six `matchTemplate` methods over every frame, exports one CSV per method, and generates one response plot per method.

**Input:** `frames/`, `template.png`  
**Output:**
- `matching_results/TM_CCOEFF.csv`, `matching_results/TM_CCOEFF_NORMED.csv`, ... (6 CSV files)
- `plots/TM_CCOEFF_plot.png`, `plots/TM_CCOEFF_NORMED_plot.png`, ... (6 PNG plots)

Each CSV contains three columns:

| frame | min_val | max_val |
|---|---|---|
| frame_0000.png | ... | ... |
| frame_0001.png | ... | ... |

**After running this script, inspect the plots in `plots/` to select the best method before proceeding to Step 4.**

---

### Step 4 — Generate the Tracking Video

```bash
python 4_generate_tracking_video.py
```

**What it does:** Prompts the user to select one of the six methods. Applies that method to every frame, draws a green bounding box around the detected object, overlays the method name, frame name, and match score, and encodes the result as an `.mp4` video.

**Input:** `frames/`, `template.png`  
**Output:** `tracking_video.mp4`

```
Available template matching methods:

1 - TM_CCOEFF
2 - TM_CCOEFF_NORMED
3 - TM_CCORR
4 - TM_CCORR_NORMED
5 - TM_SQDIFF
6 - TM_SQDIFF_NORMED

Select the method number: 2
```

---

## Output Structure

After running the full pipeline, the project directory will look like:

```
project/
├── myvideo.mp4                  ← Input video
├── template.png                 ← Cropped target region
├── tracking_video.mp4           ← Annotated output video
│
├── frames/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
│
├── matching_results/
│   ├── TM_CCOEFF.csv
│   ├── TM_CCOEFF_NORMED.csv
│   ├── TM_CCORR.csv
│   ├── TM_CCORR_NORMED.csv
│   ├── TM_SQDIFF.csv
│   └── TM_SQDIFF_NORMED.csv
│
├── plots/
│   ├── TM_CCOEFF_plot.png
│   ├── TM_CCOEFF_NORMED_plot.png
│   ├── TM_CCORR_plot.png
│   ├── TM_CCORR_NORMED_plot.png
│   ├── TM_SQDIFF_plot.png
│   └── TM_SQDIFF_NORMED_plot.png
│
├── 1_extract_frames.py
├── 2_create_template.py
├── 3_match_methods.py
└── 4_generate_tracking_video.py
```

---

## Limitations

- **Static camera required.** Template matching assumes the background is fixed. Camera movement will cause the method to produce false detections.
- **No scale or rotation invariance.** If the object changes size or orientation significantly, the match quality will degrade. This pipeline does not apply multi-scale or rotation-invariant search.
- **Single object.** The pipeline tracks one template at a time. Multiple objects would require separate templates and separate runs.
- **No temporal filtering.** Detection is performed independently per frame. There is no Kalman filter or smoothing applied to the trajectory.
- **Texture sensitivity.** Objects with low visual contrast against the background or repetitive textures (e.g., a plain white ball on a white surface) will be difficult to track reliably with any of these methods.

---

## References

- OpenCV Template Matching Tutorial: https://docs.opencv.org/4.5.2/d4/dc6/tutorial_py_template_matching.html
- OpenCV Template Matching (C++ reference): https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
- Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.
