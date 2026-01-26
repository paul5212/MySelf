import cv2
import numpy as np
import os
from collections import defaultdict
from datetime import datetime

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
log_lines = []

# =============================================================================

# 先收集所有 prefix（跨所有 drive）
all_prefixes = set()

prefix_to_drive_data = defaultdict(dict)  # prefix -> drive -> (latest_mtime, latest_path, latest_name)

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    
    if not os.path.exists(base_path):
        log_lines.append(f"[{drive}] 路徑不存在，跳過: {base_path}")
        continue
    
    print(f"\n[{drive}] 掃描資料夾: {base_path}")
    
    try:
        subfolders = [f for f in os.listdir(base_path) 
                      if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
    except Exception as e:
        log_lines.append(f"[{drive}] 讀取資料夾失敗: {e}")
        continue
    
    if not subfolders:
        log_lines.append(f"[{drive}] 無 _Src 資料夾")
        continue
    
    # 群組 per prefix
    prefix_groups = defaultdict(list)
    for sub in subfolders:
        if '_' in sub:
            prefix = sub.split('_', 1)[0]
            full_path = os.path.join(base_path, sub)
            mtime = os.path.getmtime(full_path)
            prefix_groups[prefix].append((mtime, full_path, sub))
            all_prefixes.add(prefix)
    
    # 每個 prefix 在此 drive 的最新資料夾
    for prefix, folder_list in prefix_groups.items():
        folder_list.sort(reverse=True, key=lambda x: x[0])  # 最新先
        latest_mtime, latest_path, latest_name = folder_list[0]
        prefix_to_drive_data[prefix][drive] = (latest_mtime, latest_path, latest_name)
        print(f"  [{drive}] prefix {prefix} → 最新資料夾 {latest_name}")

# 處理每個 prefix
results_summary = []

for prefix in sorted(all_prefixes):
    print(f"\n處理 prefix: {prefix}")
    
    # 初始化所有 IP 為 -1
    status_dict = {ip: -1 for ip in all_ips}
    
    # 收集該 prefix 跨 drive 的所有 latest（用於找代表性檔名）
    latest_candidates = []
    
    for drive, ips in subfolders_per_drive.items():
        drive_data = prefix_to_drive_data.get(prefix, {}).get(drive)
        if drive_data is None:
            # 此 drive 無此 prefix，所有該 drive 的 IP 保持 -1
            log_lines.append(f"[{prefix}] [{drive}] 無對應資料夾 → IP{'/'.join(ips)} 設為 -1")
            continue
        
        latest_mtime, latest_path, latest_name = drive_data
        latest_candidates.append((latest_mtime, latest_name))
        
        print(f"  [{prefix}] [{drive}] 使用最新資料夾: {latest_name}")
        
        for ip in ips:
            image_filename = image_filename_pattern.format(ip=ip)
            image_path = os.path.join(latest_path, ip, image_filename)
            
            if not os.path.exists(image_path):
                status_dict[ip] = -1
                log_lines.append(f"[{prefix}] [{drive}] [{ip}] 圖像不存在: {image_path}")
                continue
            
            try:
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise Exception("cv2.imread 返回 None")
                
                # 轉灰階
                if len(img.shape) == 3 and img.shape[2] in (3, 4):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img.copy()
                
                gray_float = gray.astype(np.float32)
                height, width = gray.shape
                
                # ROI
                roi_x = max(0, center_x - (roi_width // 2))
                roi_y = max(0, center_y - (roi_height // 2))
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
                    status = 1   # blur
                else:
                    status = 0   # normal
                
                status_dict[ip] = status
                print(f"  [{prefix}] [{drive}] [{ip}] Variance: {lap_variance:.1f} → {'blur' if status else 'normal'}")
                
            except Exception as e:
                status_dict[ip] = -1
                log_lines.append(f"[{prefix}] [{drive}] [{ip}] 處理失敗 ({e}): {image_path}")
    
    if not latest_candidates:
        log_lines.append(f"[{prefix}] 無任何最新資料夾，跳過產生 txt")
        continue
    
    # 取跨 drive 時間最新的資料夾名稱作為 txt 檔名
    latest_candidates.sort(reverse=True, key=lambda x: x[0])
    representative_name = latest_candidates[0][1]  # e.g. 001F7CB0_XXXX_Src
    txt_filename = f"{representative_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    # 寫入 txt
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Prefix: {prefix} (代表資料夾: {representative_name})\n")
        f.write(f"日期: {date}\n")
        f.write(f"模糊閾值: {blur_threshold}\n")
        f.write("="*40 + "\n")
        for ip in all_ips:
            status = status_dict[ip]
            status_str = {0: '0 (Normal)', 1: '1 (Blur)', -1: '-1 (無圖像或錯誤)'}[status]
            f.write(f"{ip}: {status_str}\n")
        # 簡易一行數字（方便解析）
        f.write("\n一行數字表示 (0=Normal, 1=Blur, -1=無/錯誤):\n")
        f.write(' '.join(str(status_dict[ip]) for ip in all_ips) + "\n")
    
    print(f"已產生: {txt_path}")
    results_summary.append((prefix, representative_name, txt_path))

# 寫入 log
with open(log_path, 'w', encoding='utf-8') as f:
    f.write(f"模糊檢測 Log (日期: {date})\n")
    f.write(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n")
    if log_lines:
        f.write('\n'.join(log_lines))
    else:
        f.write("無錯誤記錄\n")

# =============================================================================
# 最終摘要
# =============================================================================
print("\n" + "="*80)
print(f"檢測完成摘要（日期: {date}）")
print("="*80)
if results_summary:
    print(f"共產生 {len(results_summary)} 個 txt 檔案")
    print("")
    for prefix, rep_name, txt_p in results_summary:
        print(f"{prefix} → {rep_name}.txt （儲存於 {txt_p}）")
    blur_count_total = 0  # 可擴展統計
else:
    print("無任何 prefix 被處理（請檢查路徑與資料夾）")

print(f"\n所有 txt 檔案已儲存至: {output_dir}")
print(f"錯誤日誌: {log_path}")