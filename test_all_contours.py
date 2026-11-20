
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. LOAD IMAGE
# --------------------------------------------------

image_path = Path("backlight  (1).jpeg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"Cannot read image: {image_path}"
    )

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

# --------------------------------------------------
# 2. CROP IMAGE
# --------------------------------------------------

x1 = int(width * 0.10)
x2 = int(width * 0.90)
y1 = int(height * 0.02)
y2 = int(height * 0.95)

gray = gray[y1:y2, x1:x2]

original = cv2.cvtColor(
    gray, cv2.COLOR_GRAY2BGR
)

h, w = gray.shape

# --------------------------------------------------
# 3. EDGE DETECTION
# --------------------------------------------------

blur = cv2.GaussianBlur(
    gray, (5, 5), 0
)

edges = cv2.Canny(
    blur,
    threshold1=35,
    threshold2=100
)

# Close small gaps in detected edges
kernel = np.ones((3, 3), np.uint8)

edges = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

# --------------------------------------------------
# 4. TRACK LEFT AND RIGHT BOUNDARIES
# --------------------------------------------------

left_boundary = np.full(h, np.nan)
right_boundary = np.full(h, np.nan)

# Search regions on either side of flask
left_min = int(w * 0.10)
left_max = int(w * 0.49)

right_min = int(w * 0.51)
right_max = int(w * 0.90)

# Initial expected boundary positions
previous_left = int(w * 0.32)
previous_right = int(w * 0.68)

# Maximum movement allowed between rows
max_jump = max(8, int(w * 0.025))

for y in range(h):

    row = edges[y]

    left_candidates = np.where(
        row[left_min:left_max] > 0
    )[0] + left_min

    right_candidates = np.where(
        row[right_min:right_max] > 0
    )[0] + right_min

    # Track left boundary
    valid_left = left_candidates[
        np.abs(left_candidates - previous_left)
        <= max_jump
    ]

    if len(valid_left) > 0:

        selected_left = valid_left[
            np.argmin(
                np.abs(valid_left - previous_left)
            )
        ]

        left_boundary[y] = selected_left
        previous_left = selected_left

    # Track right boundary
    valid_right = right_candidates[
        np.abs(right_candidates - previous_right)
        <= max_jump
    ]

    if len(valid_right) > 0:

        selected_right = valid_right[
            np.argmin(
                np.abs(valid_right - previous_right)
            )
        ]

        right_boundary[y] = selected_right
        previous_right = selected_right

# --------------------------------------------------
# 5. FILL MISSING BOUNDARY POINTS
# --------------------------------------------------

valid_left = np.where(
    np.isfinite(left_boundary)
)[0]

valid_right = np.where(
    np.isfinite(right_boundary)
)[0]

if len(valid_left) < 10 or len(valid_right) < 10:

    raise RuntimeError(
        "Not enough boundary points detected. "
        "Try adjusting Canny thresholds or search regions."
    )

left_filled = np.interp(
    np.arange(h),
    valid_left,
    left_boundary[valid_left]
)

right_filled = np.interp(
    np.arange(h),
    valid_right,
    right_boundary[valid_right]
)

# --------------------------------------------------
# 6. SMOOTH BOUNDARIES
# --------------------------------------------------

def moving_average(values, window=9):

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="same"
    )


left_smooth = moving_average(
    left_filled,
    window=9
)

right_smooth = moving_average(
    right_filled,
    window=9
)

# --------------------------------------------------
# 7. DRAW DETECTED PROFILE
# --------------------------------------------------

result = original.copy()

for y in range(h):

    left_x = int(left_smooth[y])
    right_x = int(right_smooth[y])

    # Green = left boundary
    cv2.circle(
        result,
        (left_x, y),
        1,
        (0, 255, 0),
        -1
    )

    # Red = right boundary
    cv2.circle(
        result,
        (right_x, y),
        1,
        (0, 0, 255),
        -1
    )

# --------------------------------------------------
# 8. CALCULATE PIXEL DIMENSIONS
# --------------------------------------------------

flask_height_px = h - 1

maximum_width_px = int(
    np.max(right_smooth - left_smooth)
)

minimum_width_px = int(
    np.min(right_smooth - left_smooth)
)

aspect_ratio = (
    flask_height_px / maximum_width_px
)

print("\n========== FLASK PROFILE RESULTS ==========")

print(
    "Height (pixels):",
    flask_height_px
)

print(
    "Maximum width (pixels):",
    maximum_width_px
)

print(
    "Minimum width (pixels):",
    minimum_width_px
)

print(
    "Aspect ratio:",
    round(aspect_ratio, 3)
)

print("===========================================\n")

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

output_dir = Path("profile_results")

output_dir.mkdir(
    exist_ok=True
)

cv2.imwrite(
    str(output_dir / "edge_map.png"),
    edges
)

cv2.imwrite(
    str(output_dir / "tracked_profile.png"),
    result
)

# --------------------------------------------------
# 10. DISPLAY COMPARISON
# --------------------------------------------------

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)

plt.imshow(
    gray,
    cmap="gray"
)

plt.title("Original Cropped Image")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    edges,
    cmap="gray"
)

plt.title("Canny Edge Detection")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Tracked Flask Profile")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    str(output_dir / "profile_comparison.png"),
    dpi=200
)

plt.show()

print("Images saved successfully!")
print("Output folder:", output_dir)
