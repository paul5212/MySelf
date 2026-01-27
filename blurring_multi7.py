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

MAX_WORKERS = min(6, len(drives))   # 最多同時跑 6 個 drive（可依 CPU 調整）

# =============================================================================
def print_status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

print_status("=== 程式開始執行 ===")
print_status(f"模式: {mode}")
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

lock = threading.Lock()  # 用來保護共用字典（雖然本例可不加，但保留安全）

def process_drive(drive, folder_name, status_dict, variance_dict):
    """單一 drive 的處理函式，可被多執行緒呼叫"""
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
            
            with lock:
                status_dict[ip] = status
                variance_dict[ip] = f"{lap_variance:.1f}"
            
            print_status(f"  [{drive}-{ip}] Variance: {lap_variance:.1f} → {'Blur' if status else 'Normal'}")
            
        except Exception as e:
            print_status(f"  [{drive}-{ip}] 失敗: {type(e).__name__}: {e}")
            with lock:
                variance_dict[ip] = 'Error'

# =============================================================================
start_time = datetime.now()

if mode == 'ALL':
    # ALL 模式保持原樣（不平行化，因為每個資料夾獨立且數量可能很多）
    # ... （保持你原本的 ALL 模式程式碼，這裡省略以節省空間）
    print_status("ALL 模式執行中（未平行化）")
    # 你可以自行貼上原本的 ALL 部分

elif mode == 'LATEST':
    # LATEST 模式：先找 O 的最新資料夾名稱，然後平行處理所有 drive
    o_base_path = f"O:\\Image\\{date}"
    print_status(f"[O] 掃描用於決定最新資料夾: {o_base_path}")
    
    if not os.path.exists(o_base_path):
        print_status("[O] 路徑不存在，無法決定最新資料夾")
        sys.exit(1)
    
    try:
        o_subfolders = [f for f in os.listdir(o_base_path)
                        if os.path.isdir(os.path.join(o_base_path, f)) and '_Src' in f]
    except Exception as e:
        print_status(f"[O] 讀取失敗: {e}")
        sys.exit(1)
    
    if not o_subfolders:
        print_status("[O] 無任何 _Src 資料夾")
        sys.exit(1)
    
    o_src_list = []
    for sub in o_subfolders:
        full_path = os.path.join(o_base_path, sub)
        try:
            mtime = os.path.getmtime(full_path)
            o_src_list.append((sub, mtime))
        except:
            continue
    
    if not o_src_list:
        print_status("[O] 無有效資料夾")
        sys.exit(1)
    
    latest_folder_name = max(o_src_list, key=lambda x: x[1])[0]
    print_status(f"最新資料夾名稱（來自 O）: {latest_folder_name}")
    
    # 初始化結果字典
    status_dict = {ip: -1 for ip in all_ips}
    variance_dict = {ip: 'N/A' for ip in all_ips}
    
    # 使用 ThreadPoolExecutor 平行處理每個 drive
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_drive = {}
        for drive in drives:
            future = executor.submit(process_drive, drive, latest_folder_name, status_dict, variance_dict)
            future_to_drive[future] = drive
        
        for future in concurrent.futures.as_completed(future_to_drive):
            drive = future_to_drive[future]
            try:
                future.result()  # 取得結果（若有例外會拋出）
            except Exception as e:
                print_status(f"[{drive}] 執行緒發生未捕捉例外: {e}")
    
    # 產生 txt
    txt_filename = f"{latest_folder_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"最新資料夾 (基於 O drive): {latest_folder_name}\n")
            f.write(f"日期: {date}\n")
            f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模糊閾值: {blur_threshold}\n")
            f.write("="*50 + "\n")
            
            for ip in all_ips:
                status = status_dict[ip]
                var = variance_dict[ip]
                status_str = {0: '0 (Normal)', 1: '1 (Blur)', -1: '-1 (無圖像或錯誤)'}[status]
                f.write(f"{ip}: {status_str} | Variance: {var}\n")
            
            f.write("\n狀態數字一行:\n")
            f.write(' '.join(str(status_dict[ip]) for ip in all_ips) + "\n")
            
            f.write("\nVariance 值一行:\n")
            f.write(' '.join(str(v) for v in variance_dict.values()) + "\n")
        
        print_status(f"已產生單一 txt: {txt_path}")
        
        results.append({
            'folder': latest_folder_name,
            'txt': txt_filename,
            'statuses': status_dict,
            'variances': variance_dict
        })
        
    except Exception as e:
        print_status(f"寫入 txt 失敗: {e}")

# =============================================================================
end_time = datetime.now()
print("\n" + "="*80)
print(f"執行完成（模式: {mode}，耗時: {end_time - start_time}）")
print("="*80)

if results:
    print(f"共產生 {len(results)} 個 txt\n")
    for res in results:
        print(f"資料夾: {res['folder']}")
        print(f"txt 檔: {res['txt']}")
        print("狀態總覽:")
        print("  IP    | 狀態   | Variance")
        print("  ------|--------|---------")
        for ip in all_ips:
            s = res['statuses'][ip]
            v = res['variances'][ip]
            status_text = {0: 'Normal', 1: 'Blur', -1: '無/錯誤'}[s]
            print(f"  {ip} | {status_text:<6} | {v}")
        print()
else:
    print("無 txt 產生")

print(f"儲存位置: {output_dir}")
print("程式結束")