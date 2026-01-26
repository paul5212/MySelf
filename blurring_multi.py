import cv2
import numpy as np
import os
from collections import defaultdict

# =============================================================================
# 參數設定區（請自行調整）
# =============================================================================
date = '20260120'                  # 要處理的日期（可修改）

drives = ['O', 'P', 'Q', 'R', 'S', 'T']

# 每個磁碟機對應的 IP 子資料夾（IP1 ~ IP24）
subfolders_per_drive = {
    'O': ['IP1', 'IP2', 'IP3', 'IP4'],
    'P': ['IP5', 'IP6', 'IP7', 'IP8'],
    'Q': ['IP9', 'IP10', 'IP11', 'IP12'],
    'R': ['IP13', 'IP14', 'IP15', 'IP16'],
    'S': ['IP17', 'IP18', 'IP19', 'IP20'],
    'T': ['IP21', 'IP22', 'IP23', 'IP24'],
}

image_filename_pattern = '{sub}_Origin000003.tif'   # 圖像檔名格式

blur_threshold = 1000.0                     # 模糊閾值（可依實際樣本調整）

# ROI 設定
center_x   = 2500
center_y   = 3500
roi_width  = 1200
roi_height = 800

# 輸出設定
output_base_dir = './blur_detection_results'
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

# =============================================================================

results = {}   # key: IPxx, value: (status_code, variance, folder_name)

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    
    if not os.path.exists(base_path):
        print(f"[{drive}] 路徑不存在，跳過: {base_path}")
        continue
    
    print(f"\n[{drive}] 開始處理: {base_path}")
    
    target_ips = subfolders_per_drive.get(drive, [])
    
    # 取得所有 _Src 資料夾
    try:
        all_src_dirs = [f for f in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
    except Exception as e:
        print(f"[{drive}] 讀取失敗: {e}")
        continue
    
    if not all_src_dirs:
        print(f"[{drive}] 無 _Src 資料夾")
        continue
    
    # 找出整體最新的 _Src 資料夾（所有 IP 共用）
    latest_mtime = 0
    latest_src_path = None
    latest_src_name = None
    
    for src in all_src_dirs:
        full = os.path.join(base_path, src)
        mtime = os.path.getmtime(full)
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_src_path = full
            latest_src_name = src
    
    if latest_src_path is None:
        print(f"[{drive}] 無法找到最新 _Src 資料夾")
        continue
    
    print(f"[{drive}] 使用最新資料夾: {latest_src_name}")
    
    # 處理該 drive 下的每個 IP
    for ip_sub in target_ips:
        image_filename = image_filename_pattern.format(sub=ip_sub)
        image_path = os.path.join(latest_src_path, ip_sub, image_filename)
        
        if not os.path.exists(image_path):
            print(f"  {ip_sub} 圖像不存在: {image_path}")
            results[ip_sub] = (-1, None, latest_src_name)
            continue
        
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  {ip_sub} 讀取失敗: {image_path}")
            results[ip_sub] = (-1, None, latest_src_name)
            continue
        
        # 轉灰階
        if len(img.shape) == 3 and img.shape[2] in (3, 4):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        gray_float = gray.astype(np.float32)
        h, w = gray.shape
        
        # ROI
        roi_x = max(0, center_x - roi_width // 2)
        roi_y = max(0, center_y - roi_height // 2)
        roi_w = min(roi_width, w - roi_x)
        roi_h = min(roi_height, h - roi_y)
        
        gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Laplacian variance
        lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        _, stddev = cv2.meanStdDev(lap_abs)
        variance = stddev[0][0] ** 2
        
        if variance < blur_threshold:
            status_code = 1   # 模糊
        else:
            status_code = 0   # 正常
        
        print(f"  {ip_sub}: variance = {variance:.1f} → {'模糊' if status_code==1 else '正常'}")
        
        results[ip_sub] = (status_code, variance, latest_src_name)

# =============================================================================
# 產生 txt 檔（以最新找到的 prefix 命名）
# =============================================================================
if results:
    # 取第一個有資料的 folder name 作為檔名（假設所有 drive 用同一個最新）
    sample_folder = next(iter(results.values()))[2]
    txt_filename = f"{sample_folder}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"檢測日期: {date}\n")
        f.write(f"使用的最新資料夾: {sample_folder}\n")
        f.write(f"模糊閾值: {blur_threshold}\n")
        f.write("-" * 40 + "\n")
        
        for i in range(1, 25):
            ip = f"IP{i}"
            if ip in results:
                code, var, _ = results[ip]
                if code == -1:
                    line = f"{ip}: -1 (無圖像或讀取失敗)"
                else:
                    line = f"{ip}: {code} ({'模糊' if code==1 else '正常'})"
                    if var is not None:
                        line += f"  Var: {var:.1f}"
            else:
                line = f"{ip}: -1 (未處理)"
            
            f.write(line + "\n")
    
    print(f"\n已產生結果檔案: {txt_path}")
    
    # 同時在 console 顯示摘要
    print("\n摘要（0=正常, 1=模糊, -1=無圖像/失敗）：")
    for i in range(1, 25):
        ip = f"IP{i}"
        if ip in results:
            code, var, _ = results[ip]
            print(f"{ip}: {code}" + (f" (Var:{var:.1f})" if var else ""))
        else:
            print(f"{ip}: -1")
else:
    print("\n無任何有效結果，沒有產生 txt 檔")

print(f"\n輸出目錄: {output_dir}")