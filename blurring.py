import cv2
import numpy as np
import os

# =============================================================================
# 參數設定區（可自行調整）
# =============================================================================
#image_path = r'Z:/Image/Image blurring/20251230/001F4EF5_T550QVN10_TGT_G_Src/IP24/IP24_Origin000003.tif'
image_path = r'Z:/Image/Image blurring/20260119/001F7C6B_T550QVN10_TGT_G_Src/IP24/IP24_Origin000003.tif'


# ROI 設定
center_x       = 2500
center_y       = 3500
roi_width      = 1200
roi_height     = 800

blur_threshold = 1000.0        # 建議根據你的清晰樣本調整，例如 800~1200
# =============================================================================

if not os.path.exists(image_path):
    print(f"檔案不存在: {image_path}")
    exit()

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print("讀取影像失敗，請檢查路徑或格式")
    exit()

print(f"原始影像 shape: {img.shape}, dtype: {img.dtype}")

# 轉灰階
if len(img.shape) == 3 and img.shape[2] in (3, 4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img

height, width = gray.shape[:2]
print(f"原始寬高: {width} x {height}")

# 轉 float32
gray_float = gray.astype(np.float32) if gray.dtype != np.uint8 else gray.astype(np.float32)

# =============================================================================
# 決定 ROI 位置與大小
# =============================================================================
if center_x is None:
    center_x = width // 2
if center_y is None:
    center_y = height // 2

roi_w = roi_width
roi_h = roi_height

roi_x = center_x - (roi_w // 2)
roi_y = center_y - (roi_h // 2)

# 邊界檢查
if roi_x < 0: roi_x = 0
if roi_y < 0: roi_y = 0
if roi_x + roi_w > width: roi_w = width - roi_x
if roi_y + roi_h > height: roi_h = height - roi_y

print(f"最終 ROI 區域: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h} (中心約在 {roi_x + roi_w//2}, {roi_y + roi_h//2})")

gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

# =============================================================================
# 只計算 Laplacian variance (ROI)
# =============================================================================
lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
lap_abs = np.abs(lap)
_, lap_stddev = cv2.meanStdDev(lap_abs)
lap_variance = lap_stddev[0][0] ** 2
print(f"Laplacian Variance (ROI): {lap_variance:.4f}")

# 判斷結果（只用 Laplacian）
if lap_variance < blur_threshold:
    status = "Image blur"
    color = (0, 0, 255)   # 紅色
else:
    status = "Image Normal"
    color = (0, 255, 0)   # 綠色
print(f"判斷結果: {status} (閾值={blur_threshold})")

# =============================================================================
# 顯示結果（原尺寸，兩個視窗）
# =============================================================================

# 準備顯示用的灰階圖（uint8）
if gray.dtype != np.uint8:
    gray_display = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
else:
    gray_display = gray.copy()

# 畫綠色 ROI 框 + 文字（改用 Laplacian variance）
cv2.rectangle(gray_display, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), thickness=16)
text = f"{status}  |  Laplacian Var: {lap_variance:.1f}"
cv2.putText(gray_display, text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, thickness=5)

# Laplacian 絕對值顯示圖
lap_display = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 兩個視窗
cv2.namedWindow("Original Gray with ROI", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
cv2.imshow("Original Gray with ROI", gray_display)

cv2.namedWindow("Laplacian Absolute (ROI)", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
cv2.imshow("Laplacian Absolute (ROI)", lap_display)

cv2.waitKey(0)
cv2.destroyAllWindows()