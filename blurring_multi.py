import cv2
import numpy as np
import os
from collections import defaultdict
import sys
import traceback
from datetime import datetime
import concurrent.futures
import threading

# =============================================================================
# 參數設定區（請自行調整）
# =============================================================================
date = '20260120'
mode = 'LATEST'                    # 'ALL' 或 'LATEST'

# 當 mode = 'LATEST' 時，選擇要分析的資料夾索引 (0=最新, 1=第二新, ...)
latest_index = 0

# 當 mode = 'ALL' 時，是否要限制只處理前 N 個最新資料夾（設為 None 則處理全部）
all_max_count = None               # 例如設為 5 則只處理最新 5 個資料夾

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

# 基準中心點（所有 IP 共用）
base_center_x = 2500
base_center_y = 3500

roi_width  = 1200
roi_height = 800

# =============================================================================
# 【ROI 偏移設定區】每個 IP 可獨立調整偏移量（像素）
# =============================================================================
offsetX_dict = {
    'IP01': 0, 'IP02': 0, 'IP03': 0, 'IP04': 0,
    'IP05': 0, 'IP06': 0, 'IP07': 0, 'IP08': 0,
    'IP09': 0, 'IP10': 0, 'IP11': 0, 'IP12': 0,
    'IP13': 0, 'IP14': 0, 'IP15': 0, 'IP16': 0,
    'IP17': 0, 'IP18': 0, 'IP19': 0, 'IP20': 0,
    'IP21': 0, 'IP22': 0, 'IP23': 0, 'IP24': 0,
}

offsetY_dict = {
    'IP01': 0, 'IP02': 0, 'IP03': 0, 'IP04': 0,
    'IP05': 0, 'IP06': 0, 'IP07': 0, 'IP08': 0,
    'IP09': 0, 'IP10': 0, 'IP11': 0, 'IP12': 0,
    'IP13': 0, 'IP14': 0, 'IP15': 0, 'IP16': 0,
    'IP17': 0, 'IP18': 0, 'IP19': 0, 'IP20': 0,
    'IP21': 0, 'IP22': 0, 'IP23': 0, 'IP24': 0,
}

output_base_dir = './blur_detection_results'
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

MAX_WORKERS = min(6, len(drives))

# =============================================================================
def print_status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

print_status("=== 程式開始執行 ===")
print_status(f"模式: {mode}")
print_status(f"基準中心點: ({base_center_x}, {base_center_y})")
print_status(f"ROI 大小: {roi_width} x {roi_height}")
print_status(f"日期: {date}")
print_status(f"輸出目錄: {output_dir}")
print_status(f"模糊閾值: {blur_threshold}")
print_status(f"最大並行執行緒數: {MAX_WORKERS}")

use_pillow = False
try:
    from PIL import Image
    print_status("Pillow 已載入，將優先使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread")

# =============================================================================
results = []
lock = threading.Lock()

def process_drive(drive, folder_name, status_dict, variance_dict, median_dict):
    folder_path = f"{drive}:\\Image\\{date}\\{folder_name}"
    if not os.path.exists(folder_path):
        print_status(f"[{drive}] 無 {folder_name}，跳過")
        return

    print_status(f"[{drive}] 開始處理 {folder_name}")
    
    target_ips = subfolders_per_drive.get(drive, [])
    
    for ip in target_ips:
        image_filename = image_filename_pattern.format(ip=ip)
        image_path = os.path.join(folder_path, ip, image_filename)
        
        if not os.path.exists(image_path):
            print_status(f"  [{drive}-{ip}] 圖像不存在: {image_path}")
            continue
        
        print_status(f"  [{drive}-{ip}] 讀取: {image_path}")
        
        try:
            if use_pillow:
                pil_img = Image.open(image_path)
                img_array = np.array(pil_img)
            else:
                img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if img_array is None:
                    raise Exception("cv2.imread 失敗")
            
            if len(img_array.shape) == 3 and img_array.shape[2] in (3, 4):
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY if use_pillow else cv2.COLOR_BGR2GRAY)
            else:
                gray = img_array
            
            gray_float = gray.astype(np.float32)
            height, width = gray.shape
            
            offset_x = offsetX_dict.get(ip, 0)
            offset_y = offsetY_dict.get(ip, 0)
            center_x_actual = base_center_x + offset_x
            center_y_actual = base_center_y + offset_y
            
            roi_x = max(0, center_x_actual - (roi_width // 2))
            roi_y = max(0, center_y_actual - (roi_height // 2))
            roi_w = min(roi_width, width - roi_x)
            roi_h = min(roi_height, height - roi_y)
            
            if roi_w <= 0 or roi_h <= 0:
                raise Exception(f"ROI 無效 (IP={ip})")
            
            gray_roi = gray_float[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            
            lap = cv2.Laplacian(gray_roi, cv2.CV_32F, ksize=3)
            lap_abs = np.abs(lap)
            _, lap_stddev = cv2.meanStdDev(lap_abs)
            lap_variance = lap_stddev[0][0] ** 2
            
            median_gray = np.median(gray_roi)
            
            status = 1 if lap_variance < blur_threshold else 0
            
            with lock:
                status_dict[ip] = status
                variance_dict[ip] = f"{lap_variance:.1f}"
                median_dict[ip] = f"{median_gray:.1f}"
            
            print_status(f"  [{drive}-{ip}] Center: ({center_x_actual},{center_y_actual}) | Var: {lap_variance:.1f} | Med: {median_gray:.1f} → {'Blur' if status else 'Normal'}")
            
        except Exception as e:
            print_status(f"  [{drive}-{ip}] 失敗: {type(e).__name__}: {e}")
            with lock:
                variance_dict[ip] = 'Error'
                median_dict[ip] = 'Error'

# =============================================================================
start_time = datetime.now()

# =============================================================================
# 統一從 O: 取得排序後的資料夾列表
# =============================================================================
o_base_path = f"O:\\Image\\{date}"
print_status(f"[O] 掃描用於產生資料夾列表: {o_base_path}")

if not os.path.exists(o_base_path):
    print_status("[O] 路徑不存在，無法繼續")
    sys.exit(1)

try:
    o_subfolders = [f for f in os.listdir(o_base_path)
                    if os.path.isdir(os.path.join(o_base_path, f)) and '_Src' in f]
except Exception as e:
    print_status(f"[O] 讀取失敗: {type(e).__name__}: {e}")
    sys.exit(1)

if not o_subfolders:
    print_status("[O] 無任何 _Src 資料夾")
    sys.exit(1)

o_src_folders = []
for sub in o_subfolders:
    full_path = os.path.join(o_base_path, sub)
    try:
        mtime = os.path.getmtime(full_path)
        o_src_folders.append((sub, mtime))
    except:
        continue

if not o_src_folders:
    print_status("[O] 無有效資料夾")
    sys.exit(1)

# 按修改時間降序排序（最新在前）
o_src_folders.sort(key=lambda x: x[1], reverse=True)

print_status("\nO drive 資料夾列表（按時間降序）：")
for idx, (name, mtime) in enumerate(o_src_folders):
    print_status(f"  Index {idx}: {name} (mtime: {datetime.fromtimestamp(mtime)})")

# =============================================================================
if mode == 'LATEST':
    try:
        selected_folders = [o_src_folders[latest_index]]
        print_status(f"\n選擇 Index {latest_index} 的資料夾進行分析")
    except IndexError:
        print_status(f"錯誤: Index {latest_index} 超出範圍 (可用 0 ~ {len(o_src_folders)-1})")
        sys.exit(1)

elif mode == 'ALL':
    if all_max_count is not None and all_max_count > 0:
        selected_folders = o_src_folders[:all_max_count]
        print_status(f"\nALL 模式：僅處理前 {all_max_count} 個最新資料夾")
    else:
        selected_folders = o_src_folders
        print_status(f"\nALL 模式：處理全部 {len(selected_folders)} 個資料夾")
else:
    print_status("無效模式，請設為 'ALL' 或 'LATEST'")
    sys.exit(1)

# =============================================================================
# 逐一處理選定的資料夾
# =============================================================================
for idx, (folder_name, mtime) in enumerate(selected_folders):
    print_status(f"\n=== 開始處理資料夾 ({idx+1}/{len(selected_folders)}): {folder_name} ===")
    
    status_dict  = {ip: -1 for ip in all_ips}
    variance_dict = {ip: 'N/A' for ip in all_ips}
    median_dict  = {ip: 'N/A' for ip in all_ips}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_drive, drive, folder_name, status_dict, variance_dict, median_dict)
            for drive in drives
        ]
        concurrent.futures.wait(futures)
    
    # 產生 txt
    txt_filename = f"{folder_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"資料夾: {folder_name}\n")
            f.write(f"處理順序: Index {idx} (基於 O drive 排序)\n")
            f.write(f"日期: {date}\n")
            f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"基準中心: ({base_center_x}, {base_center_y})\n")
            f.write(f"模糊閾值: {blur_threshold}\n")
            f.write("="*70 + "\n\n")
            
            # 標題行 - 固定寬度對齊
            f.write(f"{'IP':<6} {'Variance':>12}  {'Median Gray':>12}  {'狀態':<15}\n")
            f.write("-" * 70 + "\n")
            
            for ip in all_ips:
                status = status_dict[ip]
                var_str = variance_dict[ip]
                med_str = median_dict[ip]
                
                if var_str in ('N/A', 'Error') or med_str in ('N/A', 'Error'):
                    status_str = '-1 (無/錯誤)'
                    var_display = var_str.rjust(12)
                    med_display = med_str.rjust(12)
                else:
                    try:
                        v = float(var_str)
                        m = float(med_str)
                        status_str = '0 (Normal)' if status == 0 else '1 (Blur)'
                        var_display = f"{v:12.1f}"
                        med_display = f"{m:12.1f}"
                    except:
                        var_display = var_str.rjust(12)
                        med_display = med_str.rjust(12)
                        status_str = '解析錯誤'
                
                f.write(f"{ip:<6} {var_display}  {med_display}  {status_str:<15}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("狀態總結一行 (0=Normal, 1=Blur, -1=無/錯誤):\n")
            f.write(' '.join(str(status_dict[ip]) for ip in all_ips) + "\n\n")
            
            f.write(f"所有結果儲存於: {output_dir}\n")
        
        print_status(f"已完成並產生: {txt_path}")
        
        results.append({
            'folder': folder_name,
            'txt': txt_filename,
            'statuses': status_dict,
            'variances': variance_dict,
            'medians': median_dict
        })
        
    except Exception as e:
        print_status(f"寫入 {txt_filename} 失敗: {e}")

# =============================================================================
# 結束總覽
# =============================================================================
print("\n" + "="*80)
print(f"執行完成（模式: {mode}，處理 {len(results)} 個資料夾）")
print("="*80)

if results:
    print(f"已產生 {len(results)} 個 txt\n")
    for res in results:
        print(f"資料夾: {res['folder']}")
        print(f"txt 檔: {res['txt']}")
        print("  狀態總覽 (前 8 個 IP 顯示):")
        print("  IP    | 狀態   | Variance   | Median Gray")
        print("  ------|--------|------------|------------")
        for ip in all_ips[:8]:
            s = res['statuses'][ip]
            v = res['variances'][ip]
            m = res['medians'][ip]
            status_text = {0: 'Normal', 1: 'Blur', -1: '無/錯誤'}[s]
            print(f"  {ip} | {status_text:<6} | {v:>10} | {m:>11}")
        print("  ... (共 24 個 IP)\n")
else:
    print("無任何 txt 產生")

print(f"所有 txt 儲存位置: {output_dir}")
print("程式結束")