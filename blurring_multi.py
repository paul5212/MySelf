import cv2
import numpy as np
import os
from collections import defaultdict

# =============================================================================
# 參數設定區（請自行調整）
# =============================================================================
date = '20260120'                  # 要處理的日期（可自行修改）

drives = ['O', 'P', 'Q', 'R', 'S', 'T']   # 6台電腦對應的網路磁碟機代號

# 每個磁碟機對應的 IP 子資料夾（共 IP1 ~ IP24）
subfolders_per_drive = {
    'O': ['IP1', 'IP2', 'IP3', 'IP4'],
    'P': ['IP5', 'IP6', 'IP7', 'IP8'],
    'Q': ['IP9', 'IP10', 'IP11', 'IP12'],
    'R': ['IP13', 'IP14', 'IP15', 'IP16'],
    'S': ['IP17', 'IP18', 'IP19', 'IP20'],
    'T': ['IP21', 'IP22', 'IP23', 'IP24'],
}

image_filename_pattern = '{sub}_Origin000003.tif'   # 圖像檔名格式

blur_threshold = 1000.0                     # 模糊判斷閾值（建議根據清晰樣本調整 800~1200）

# ROI 設定（假設所有圖像尺寸一致）
center_x   = 2500
center_y   = 3500
roi_width  = 1200
roi_height = 800

# 輸出設定
output_base_dir = './blur_detection_results'   # 結果儲存目錄（會自動建立）

# 若只想處理特定 IP，可取消註釋並填入
# ips_to_process = ['IP1', 'IP24']
ips_to_process = None   # None = 處理所有 24 個 IP

# =============================================================================

output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

# 儲存每個 drive 的最新資料夾名稱
latest_names_per_drive = {}

# 儲存 IP1~IP24 的狀態：0=Normal, 1=Blur, 'missing'=未找到或處理失敗
ip_status = {f'IP{i}': 'missing' for i in range(1, 25)}

processed_count = 0
blur_count = 0

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    
    if not os.path.exists(base_path):
        print(f"[{drive}] 路徑不存在，跳過: {base_path}")
        continue
    
    print(f"\n[{drive}] 開始處理: {base_path}")
    
    # 取得所有 _Src 資料夾並找出修改時間最新的
    try:
        entries = os.listdir(base_path)
        src_folders = [f for f in entries if '_Src' in f and os.path.isdir(os.path.join(base_path, f))]
        
        if not src_folders:
            print(f"[{drive}] 無符合 _Src 的子資料夾")
            continue
        
        folder_mtimes = []
        for f in src_folders:
            fp = os.path.join(base_path, f)
            folder_mtimes.append((os.path.getmtime(fp), f))
        
        folder_mtimes.sort(reverse=True)  # 降冪，取最新
        latest_name = folder_mtimes[0][1]
        latest_path = os.path.join(base_path, latest_name)
        
        latest_names_per_drive[drive] = latest_name
        
        print(f"[{drive}] 選取最新資料夾: {latest_name}")
        
    except Exception as e:
        print(f"[{drive}] 讀取資料夾失敗: {e}")
        continue
    
    # 處理該 drive 負責的 IP
    target_ips = subfolders_per_drive.get(drive, [])
    
    for subfolder in target_ips:
        if ips_to_process is not None and subfolder not in ips_to_process:
            continue
        
        image_filename = image_filename_pattern.format(sub=subfolder)
        image_path = os.path.join(latest_path, subfolder, image_filename)
        
        if not os.path.exists(image_path):
            print(f"  [{drive}-{subfolder}] 圖像不存在: {image_path}")
            continue
        
        # 讀取圖像
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [{drive}-{subfolder}] 讀取圖像失敗")
            continue
        
        # 轉灰階
        if len(img.shape) == 3 and img.shape[2] in (3, 4):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        gray_float = gray.astype(np.float32)
        height, width = gray.shape
        
        # ROI 邊界檢查
        roi_x = center_x - (roi_width // 2)
        roi_y = center_y - (roi_height // 2)
        if roi_x < 0: roi_x = 0
        if roi_y < 0: roi_y = 0
        roi_w = min(roi_width, width - roi_x)
        roi_h = min(roi_height, height - roi_y)
        
        gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Laplacian variance
        lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        _, lap_stddev = cv2.meanStdDev(lap_abs)
        lap_variance = lap_stddev[0][0] ** 2
        
        # 判斷
        if lap_variance < blur_threshold:
            status = "Image blur"
            status_num = 1
            blur_count += 1
        else:
            status = "Image Normal"
            status_num = 0
        
        ip_status[subfolder] = status_num
        processed_count += 1
        
        print(f"  [{drive}-{subfolder}] Laplacian Var: {lap_variance:.1f} → {status} ({status_num})")

# =============================================================================
# 產生 TXT 結果檔案
# =============================================================================
unique_latest = set(latest_names_per_drive.values())

if len(unique_latest) == 1:
    base_folder_name = list(unique_latest)[0]
    txt_filename = f"{base_folder_name}.txt"
    print(f"\n所有 drive 使用相同最新資料夾名稱，產生檔案: {txt_filename}")
else:
    base_folder_name = None
    txt_filename = f"{date}_multi_drives.txt"
    print(f"\n警告: 不同 drive 的最新資料夾名稱不一致，產生替代檔案: {txt_filename}")

txt_path = os.path.join(output_dir, txt_filename)

with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f"Blur Detection Results - Date: {date}\n")
    f.write(f"Threshold: {blur_threshold}\n")
    f.write("="*50 + "\n")
    
    if base_folder_name:
        f.write(f"Based on latest folder: {base_folder_name}\n\n")
    else:
        f.write("Latest folders per drive:\n")
        for d in drives:
            name = latest_names_per_drive.get(d, 'Not found')
            f.write(f"{d}: {name}\n")
        f.write("\n")
    
    f.write("IP Status (0 = Normal, 1 = Blur, missing = Not processed)\n")
    f.write("="*50 + "\n")
    
    for i in range(1, 25):
        ip = f"IP{i}"
        status = ip_status[ip]
        if status == 'missing':
            f.write(f"{ip} missing\n")
        else:
            f.write(f"{ip} {status}\n")

print(f"\nTXT 結果已儲存至: {txt_path}")

# =============================================================================
# 最終摘要（Console）
# =============================================================================
missing_count = sum(1 for v in ip_status.values() if v == 'missing')

print("\n" + "="*80)
print(f"檢測完成摘要（日期: {date}）")
print("="*80)
print(f"總 IP 數: 24 | 已處理: {processed_count} | 模糊: {blur_count} | 缺失: {missing_count}")
print("\n詳細狀態:")
for i in range(1, 25):
    ip = f"IP{i}"
    status = ip_status[ip]
    if status == 'missing':
        print(f"{ip}: missing")
    elif status == 0:
        print(f"{ip}: 0 (Normal)")
    else:
        print(f"{ip}: 1 (Blur)")

print(f"\n結果 TXT 檔案位於: {txt_path}")