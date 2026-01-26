import cv2
import numpy as np
import os
from collections import defaultdict
import sys
import traceback

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

# 所有 IP1~IP24（用於排序輸出）
all_ips = [f'IP{i}' for i in range(1, 25)]

image_filename_pattern = '{ip}_Origin000003.tif'   # 圖像檔名格式

blur_threshold = 1000.0                     # 模糊判斷閾值

# ROI 設定（假設所有圖像尺寸一致）
center_x   = 2500
center_y   = 3500
roi_width  = 1200
roi_height = 800

# 輸出設定
output_base_dir = './blur_detection_results'   # 結果儲存目錄
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, 'detection_log.txt')

# =============================================================================
# 立即開啟 log 檔案（每條訊息即時寫入，避免閃退時完全無 log）
# =============================================================================
log_file = open(log_path, 'w', encoding='utf-8')
def log(msg):
    print(msg)                          # 同時輸出到 console（方便除錯）
    log_file.write(msg + '\n')
    log_file.flush()                    # 強制立即寫入磁碟

log("=== 程式開始執行 ===")
log(f"Python 版本: {sys.version}")
log(f"執行日期設定: {date}")
log(f"輸出目錄: {output_dir}")
log(f"模糊閾值: {blur_threshold}")

# =============================================================================
# 強烈建議安裝 Pillow（讀取 .tif 更穩定，避免 OpenCV 常見的閃退）
# =============================================================================
try:
    from PIL import Image
    log("成功匯入 Pillow，將使用 Pillow 讀取圖像（更穩定）")
    use_pillow = True
except ImportError as e:
    log("Pillow 未安裝，將 fallback 到 cv2.imread（可能閃退）")
    log("建議執行: pip install pillow")
    use_pillow = False

# =============================================================================
all_prefixes = set()
prefix_to_drive_data = defaultdict(dict)  # prefix -> drive -> (latest_mtime, latest_path, latest_name)

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    log(f"\n[{drive}] 掃描路徑: {base_path}")
    
    if not os.path.exists(base_path):
        log(f"[{drive}] 路徑不存在，跳過")
        continue
    
    try:
        subfolders = [f for f in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
    except Exception as e:
        log(f"[{drive}] 讀取資料夾失敗: {type(e).__name__}: {e}")
        continue
    
    if not subfolders:
        log(f"[{drive}] 無 _Src 資料夾")
        continue
    
    prefix_groups = defaultdict(list)
    for sub in subfolders:
        if '_' in sub:
            prefix = sub.split('_', 1)[0]
            full_path = os.path.join(base_path, sub)
            try:
                mtime = os.path.getmtime(full_path)
            except Exception as e:
                log(f"[{drive}] 取得 mtime 失敗 ({sub}): {e}")
                continue
            prefix_groups[prefix].append((mtime, full_path, sub))
            all_prefixes.add(prefix)
    
    for prefix, folder_list in prefix_groups.items():
        folder_list.sort(reverse=True, key=lambda x: x[0])
        latest_mtime, latest_path, latest_name = folder_list[0]
        prefix_to_drive_data[prefix][drive] = (latest_mtime, latest_path, latest_name)
        log(f"  [{drive}] prefix {prefix} → 最新資料夾 {latest_name}")

# =============================================================================
results_summary = []

for prefix in sorted(all_prefixes):
    log(f"\n=== 處理 prefix: {prefix} ===")
    
    status_dict = {ip: -1 for ip in all_ips}
    latest_candidates = []
    
    for drive, ips in subfolders_per_drive.items():
        drive_data = prefix_to_drive_data.get(prefix, {}).get(drive)
        if drive_data is None:
            log(f"[{prefix}] [{drive}] 無對應資料夾 → IP{'/'.join(ips)} 設為 -1")
            continue
        
        latest_mtime, latest_path, latest_name = drive_data
        latest_candidates.append((latest_mtime, latest_name))
        log(f"  [{prefix}] [{drive}] 使用最新資料夾: {latest_name}")
        
        for ip in ips:
            image_filename = image_filename_pattern.format(ip=ip)
            image_path = os.path.join(latest_path, ip, image_filename)
            
            if not os.path.exists(image_path):
                log(f"[{prefix}] [{drive}-{ip}] 圖像不存在: {image_path}")
                continue
            
            log(f"  [{prefix}] [{drive}-{ip}] 開始讀取圖像: {image_path}")
            
            try:
                if use_pillow:
                    # 使用 Pillow 讀取（更穩定）
                    pil_img = Image.open(image_path)
                    img_array = np.array(pil_img)
                else:
                    # fallback cv2
                    img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if img_array is None:
                        raise Exception("cv2.imread 返回 None")
                
                # 轉灰階
                if len(img_array.shape) == 3:
                    if img_array.shape[2] in (3, 4):
                        if use_pillow:
                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    else:
                        raise Exception(f"意外 channel 數: {img_array.shape[2]}")
                else:
                    gray = img_array
                
                log(f"  [{prefix}] [{drive}-{ip}] 圖像 shape: {gray.shape}, dtype: {gray.dtype}")
                
                gray_float = gray.astype(np.float32)
                height, width = gray.shape
                
                # ROI
                roi_x = max(0, center_x - (roi_width // 2))
                roi_y = max(0, center_y - (roi_height // 2))
                roi_w = min(roi_width, width - roi_x)
                roi_h = min(roi_height, height - roi_y)
                
                if roi_w <= 0 or roi_h <= 0:
                    raise Exception(f"ROI 無效: w={roi_w}, h={roi_h}")
                
                gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                
                # Laplacian variance
                lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
                lap_abs = np.abs(lap)
                _, lap_stddev = cv2.meanStdDev(lap_abs)
                lap_variance = lap_stddev[0][0] ** 2
                
                log(f"  [{prefix}] [{drive}-{ip}] Laplacian Variance: {lap_variance:.1f}")
                
                status = 1 if lap_variance < blur_threshold else 0
                status_dict[ip] = status
                log(f"  [{prefix}] [{drive}-{ip}] 判斷結果: {'Blur (1)' if status else 'Normal (0)'}")
                
            except Exception as e:
                log(f"[{prefix}] [{drive}-{ip}] 處理失敗: {type(e).__name__}: {e}")
                log(f"Traceback: {traceback.format_exc()}")
                status_dict[ip] = -1
    
    if not latest_candidates:
        log(f"[{prefix}] 無任何有效資料夾，跳過產生 txt")
        continue
    
    latest_candidates.sort(reverse=True, key=lambda x: x[0])
    representative_name = latest_candidates[0][1]
    txt_filename = f"{representative_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Prefix: {prefix} (代表資料夾: {representative_name})\n")
            f.write(f"日期: {date}\n")
            f.write(f"模糊閾值: {blur_threshold}\n")
            f.write("="*40 + "\n")
            for ip in all_ips:
                status = status_dict[ip]
                status_str = {0: '0 (Normal)', 1: '1 (Blur)', -1: '-1 (無圖像或錯誤)'}[status]
                f.write(f"{ip}: {status_str}\n")
            f.write("\n一行數字表示 (0=Normal, 1=Blur, -1=無/錯誤):\n")
            f.write(' '.join(str(status_dict[ip]) for ip in all_ips) + "\n")
        log(f"[{prefix}] 已成功產生 txt: {txt_path}")
    except Exception as e:
        log(f"[{prefix}] 寫入 txt 失敗: {e}")
    
    results_summary.append((prefix, representative_name, txt_path))

# =============================================================================
log("\n=== 程式執行結束 ===")
if results_summary:
    log(f"共產生 {len(results_summary)} 個 txt 檔案")
    for prefix, rep_name, txt_p in results_summary:
        log(f"{prefix} → {rep_name}.txt")
else:
    log("無產生任何 txt（可能所有 prefix 都有問題）")

log(f"完整 log 已儲存至: {log_path}")
log_file.close()
print(f"\n程式結束，log 位置: {log_path}")