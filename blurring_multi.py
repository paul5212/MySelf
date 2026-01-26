import cv2
import numpy as np
import os
from collections import defaultdict

# =============================================================================
# 參數設定區
# =============================================================================
date = '20260120'                  # ← 請改成你要的日期

drives = ['O', 'P', 'Q', 'R', 'S', 'T']

subfolders_per_drive = {
    'O': ['IP1', 'IP2', 'IP3', 'IP4'],
    'P': ['IP5', 'IP6', 'IP7', 'IP8'],
    'Q': ['IP9', 'IP10', 'IP11', 'IP12'],
    'R': ['IP13', 'IP14', 'IP15', 'IP16'],
    'S': ['IP17', 'IP18', 'IP19', 'IP20'],
    'T': ['IP21', 'IP22', 'IP23', 'IP24'],
}

image_filename_pattern = '{sub}_Origin000003.tif'

blur_threshold = 1000.0

center_x   = 2500
center_y   = 3500
roi_width  = 1200
roi_height = 800

output_base_dir = './blur_detection_results'
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

# =============================================================================

results = {}  # ip -> (status_code, variance or None, folder_name or None)

latest_src_name_overall = None
latest_mtime_overall = 0

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    
    if not os.path.exists(base_path):
        print(f"[{drive}] 路徑不存在，跳過")
        continue
    
    print(f"[{drive}] 掃描中...")
    
    try:
        all_src_dirs = [d for d in os.listdir(base_path) 
                        if os.path.isdir(os.path.join(base_path, d)) and '_Src' in d]
    except Exception as e:
        print(f"[{drive}] 讀取目錄失敗: {e}")
        continue
    
    if not all_src_dirs:
        continue
    
    # 找出這個 drive 最新的 _Src 資料夾
    for src in all_src_dirs:
        full = os.path.join(base_path, src)
        try:
            mtime = os.path.getmtime(full)
            if mtime > latest_mtime_overall:
                latest_mtime_overall = mtime
                latest_src_name_overall = src
        except:
            pass
    
    # 處理這個 drive 的每個 IP
    for ip_sub in subfolders_per_drive.get(drive, []):
        image_filename = image_filename_pattern.format(sub=ip_sub)
        image_path = os.path.join(base_path, latest_src_name_overall or "", ip_sub, image_filename)
        
        status_code = -1
        variance = None
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError("檔案不存在")
            
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2.imread 失敗")
            
            # 轉灰階
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            gray_float = gray.astype(np.float32)
            h, w = gray.shape
            
            # ROI 安全切片
            roi_x = max(0, center_x - roi_width // 2)
            roi_y = max(0, center_y - roi_height // 2)
            roi_w = min(roi_width, w - roi_x)
            roi_h = min(roi_height, h - roi_y)
            
            if roi_w <= 0 or roi_h <= 0:
                raise ValueError("ROI 尺寸無效")
            
            gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
            lap_abs = np.abs(lap)
            
            mean, stddev = cv2.meanStdDev(lap_abs)
            variance = float(stddev[0][0] ** 2)
            
            status_code = 1 if variance < blur_threshold else 0
            
            print(f"{ip_sub:5} | Var = {variance:8.1f} | {'模糊' if status_code==1 else '正常'}")
        
        except Exception as e:
            print(f"{ip_sub:5} | 處理失敗: {str(e)}")
            status_code = -1
            variance = None
        
        results[ip_sub] = (status_code, variance, latest_src_name_overall)

# =============================================================================
# 產生 txt 檔
# =============================================================================
if results:
    folder_name = latest_src_name_overall or "未知資料夾"
    txt_path = os.path.join(output_dir, f"{folder_name}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"檢測日期     : {date}\n")
        f.write(f"最新資料夾    : {folder_name}\n")
        f.write(f"模糊閾值      : {blur_threshold}\n")
        f.write(f"生成時間      : {os.path.getctime(txt_path):.0f}\n")
        f.write("-" * 50 + "\n")
        
        for i in range(1, 25):
            ip = f"IP{i}"
            if ip in results:
                code, var, _ = results[ip]
                if code == -1:
                    f.write(f"{ip:6} : -1  (處理失敗或無圖像)\n")
                elif code == 0:
                    f.write(f"{ip:6} : 0   (正常)    Var = {var:8.1f}\n")
                else:
                    f.write(f"{ip:6} : 1   (模糊)    Var = {var:8.1f}\n")
            else:
                f.write(f"{ip:6} : -1  (未掃描)\n")
    
    print(f"\n已產生結果檔案：\n{txt_path}\n")
    
    # 同時印在螢幕上方便快速查看
    print("快速摘要：")
    for i in range(1, 25):
        ip = f"IP{i}"
        if ip in results:
            code, var, _ = results[ip]
            print(f"{ip:6} : {code}" + (f" ({var:.1f})" if var is not None else ""))
        else:
            print(f"{ip:6} : -1")
else:
    print("完全沒有處理到任何 IP，請檢查磁碟機映射、路徑、日期是否正確")