import cv2
import numpy as np
import os
from collections import defaultdict
import sys
import traceback
from datetime import datetime

# =============================================================================
# 參數設定區（請自行調整）
# =============================================================================
date = '20260120'                  # 要處理的日期（可自行修改）

drives = ['O', 'P', 'Q', 'R', 'S', 'T']

subfolders_per_drive = {
    'O': ['IP01', 'IP02', 'IP03', 'IP04'],
    'P': ['IP05', 'IP06', 'IP07', 'IP08'],
    'Q': ['IP09', 'IP10', 'IP11', 'IP12'],
    'R': ['IP13', 'IP14', 'IP15', 'IP16'],
    'S': ['IP17', 'IP18', 'IP19', 'IP20'],
    'T': ['IP21', 'IP22', 'IP23', 'IP24'],
}

all_ips = [f'IP{i:02d}' for i in range(1, 25)]

image_filename_pattern = '{ip}_Origin000003.tif'

blur_threshold = 1000.0

center_x   = 2500
center_y   = 3500
roi_width  = 1200
roi_height = 800

output_base_dir = './blur_detection_results'
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, 'detection_log.txt')

# =============================================================================
log_file = open(log_path, 'w', encoding='utf-8')
def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("=== 程式開始執行 ===")
log(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Python 版本: {sys.version}")
log(f"日期: {date}")
log(f"輸出目錄: {output_dir}")
log(f"模糊閾值: {blur_threshold}")

try:
    from PIL import Image
    log("Pillow 已載入，將優先使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    log("Pillow 未安裝，使用 cv2.imread（建議 pip install pillow）")
    use_pillow = False

# =============================================================================
# 收集所有 prefix 與每個 drive 的最新資料夾
# =============================================================================
prefix_to_latest = defaultdict(lambda: (0, None, None, None))  # prefix -> (mtime, drive, folder_path, folder_name)

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    log(f"\n[{drive}] 掃描: {base_path}")
    
    if not os.path.exists(base_path):
        log(f"[{drive}] 路徑不存在，跳過")
        continue
    
    try:
        subfolders = [f for f in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
    except Exception as e:
        log(f"[{drive}] 讀取失敗: {type(e).__name__}: {e}")
        continue
    
    if not subfolders:
        log(f"[{drive}] 無 _Src 資料夾")
        continue
    
    for sub in subfolders:
        if '_' not in sub:
            continue
        prefix = sub.split('_', 1)[0]
        full_path = os.path.join(base_path, sub)
        try:
            mtime = os.path.getmtime(full_path)
        except Exception as e:
            log(f"[{drive}] mtime 失敗 ({sub}): {e}")
            continue
        
        # 比較是否更晚
        current_mtime, _, _, _ = prefix_to_latest[prefix]
        if mtime > current_mtime:
            prefix_to_latest[prefix] = (mtime, drive, full_path, sub)
            log(f"  更新 prefix {prefix} 最新資料夾: [{drive}] {sub} (mtime: {datetime.fromtimestamp(mtime)})")

# =============================================================================
results_summary = []

for prefix, (latest_mtime, drive, folder_path, folder_name) in sorted(prefix_to_latest.items()):
    if folder_path is None:
        log(f"prefix {prefix} 無有效資料夾，跳過")
        continue
    
    log(f"\n=== 處理最新資料夾: {folder_name} (prefix: {prefix}, drive: {drive}) ===")
    
    status_dict = {ip: -1 for ip in all_ips}
    variance_dict = {ip: 'N/A' for ip in all_ips}
    
    # 只處理該 drive 的 IP
    target_ips = subfolders_per_drive.get(drive, [])
    
    for ip in target_ips:
        image_filename = image_filename_pattern.format(ip=ip)
        image_path = os.path.join(folder_path, ip, image_filename)
        
        if not os.path.exists(image_path):
            log(f"[{folder_name}] [{drive}-{ip}] 圖像不存在: {image_path}")
            continue
        
        log(f"[{folder_name}] [{drive}-{ip}] 讀取: {image_path}")
        
        try:
            if use_pillow:
                pil_img = Image.open(image_path)
                img_array = np.array(pil_img)
            else:
                img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img_array is None:
                    raise Exception("cv2.imread 失敗")
            
            # 轉灰階
            if len(img_array.shape) == 3 and img_array.shape[2] in (3, 4):
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY if use_pillow else cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            log(f"[{folder_name}] [{drive}-{ip}] shape: {gray.shape}, dtype: {gray.dtype}")
            
            gray_float = gray.astype(np.float32)
            height, width = gray.shape
            
            roi_x = max(0, center_x - (roi_width // 2))
            roi_y = max(0, center_y - (roi_height // 2))
            roi_w = min(roi_width, width - roi_x)
            roi_h = min(roi_height, height - roi_y)
            
            if roi_w <= 0 or roi_h <= 0:
                raise Exception(f"ROI 無效: w={roi_w}, h={roi_h}")
            
            gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
            lap_abs = np.abs(lap)
            _, lap_stddev = cv2.meanStdDev(lap_abs)
            lap_variance = lap_stddev[0][0] ** 2
            
            status = 1 if lap_variance < blur_threshold else 0
            status_dict[ip] = status
            variance_dict[ip] = f"{lap_variance:.1f}"
            
            log(f"[{folder_name}] [{drive}-{ip}] Variance: {lap_variance:.1f} → {'Blur' if status else 'Normal'}")
            
        except Exception as e:
            log(f"[{folder_name}] [{drive}-{ip}] 失敗: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            variance_dict[ip] = 'Error'
    
    # 產生 txt
    txt_filename = f"{folder_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Prefix: {prefix}\n")
            f.write(f"最新資料夾: {folder_name} (Drive: {drive})\n")
            f.write(f"日期: {date}\n")
            f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模糊閾值: {blur_threshold}\n")
            f.write("="*50 + "\n")
            
            for ip in all_ips:
                status = status_dict[ip]
                var = variance_dict[ip]
                status_str = {0: '0 (Normal)', 1: '1 (Blur)', -1: '-1 (無圖像或錯誤)'}[status]
                f.write(f"{ip}: {status_str} | Variance: {var}\n")
            
            f.write("\n狀態數字一行 (0=Normal, 1=Blur, -1=無/錯誤):\n")
            f.write(' '.join(str(status_dict[ip]) for ip in all_ips) + "\n")
            
            f.write("\nVariance 值一行:\n")
            f.write(' '.join(str(v) for v in variance_dict.values()) + "\n")
        
        log(f"已產生: {txt_path}")
        results_summary.append((prefix, folder_name, txt_path))
        
    except Exception as e:
        log(f"寫入 {txt_filename} 失敗: {e}")

# =============================================================================
log("\n=== 執行結束 ===")
if results_summary:
    log(f"共產生 {len(results_summary)} 個 txt（每個 prefix 只取最新）")
    for prefix, fname, path in results_summary:
        log(f"{prefix} → {fname}")
else:
    log("無產生 txt（無任何 prefix 有有效資料夾）")

log_file.close()
print(f"\n完成，結果與 log 在: {output_dir}")