import cv2
import numpy as np
import os
from collections import defaultdict
import sys
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 避免 numpy 警告干擾

# =============================================================================
# 參數設定區
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

# 多進程設定
NUM_PROCESSES = max(1, cpu_count() - 1)  # 保留一個核心給系統

# =============================================================================
def print_status(msg):
    print(msg, flush=True)

print_status("=== 程式開始執行 ===")
print_status(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_status(f"Python 版本: {sys.version}")
print_status(f"日期: {date}")
print_status(f"模式: {mode}")
print_status(f"輸出目錄: {output_dir}")
print_status(f"模糊閾值: {blur_threshold}")
print_status(f"使用進程數: {NUM_PROCESSES}")

try:
    from PIL import Image
    print_status("Pillow 已載入，將優先使用 Pillow 讀取 .tif")
    USE_PILLOW = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread（建議 pip install pillow）")
    USE_PILLOW = False

# =============================================================================
# 單一 IP 影像處理函數（供多進程呼叫）
# =============================================================================
def process_single_ip(args):
    drive, folder_path, ip, image_filename_pattern, blur_threshold, center_x, center_y, roi_width, roi_height = args
    
    image_filename = image_filename_pattern.format(ip=ip)
    image_path = os.path.join(folder_path, ip, image_filename)
    
    result = {'ip': ip, 'status': -1, 'variance': 'N/A', 'error': None}
    
    if not os.path.exists(image_path):
        result['error'] = f"圖像不存在: {image_path}"
        return result
    
    try:
        if USE_PILLOW:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
        else:
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise Exception("cv2.imread 失敗")
        
        if len(img_array.shape) == 3 and img_array.shape[2] in (3, 4):
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY if USE_PILLOW else cv2.COLOR_BGR2GRAY)
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
        
        result['status'] = status
        result['variance'] = f"{lap_variance:.1f}"
        
        return result
    
    except Exception as e:
        result['error'] = str(e)
        result['variance'] = 'Error'
        return result

# =============================================================================
# 主邏輯
# =============================================================================
results = []

if mode == 'ALL':
    # ALL 模式：每個 drive 的每個資料夾獨立處理
    tasks = []
    
    for drive in drives:
        base_path = f"{drive}:\\Image\\{date}"
        print_status(f"\n[{drive}] 掃描路徑: {base_path}")
        
        if not os.path.exists(base_path):
            continue
        
        try:
            subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
        except:
            continue
        
        for sub in subfolders:
            folder_path = os.path.join(base_path, sub)
            target_ips = subfolders_per_drive.get(drive, [])
            
            for ip in target_ips:
                tasks.append((drive, folder_path, ip, image_filename_pattern, blur_threshold, center_x, center_y, roi_width, roi_height))
    
    if tasks:
        print_status(f"\n開始多進程處理 {len(tasks)} 個影像...")
        with Pool(processes=NUM_PROCESSES) as pool:
            results_all = list(pool.imap(process_single_ip, tasks))
        
        # 這裡可以進一步整理 results_all 到 txt，但因 ALL 模式檔案多，建議自行擴充
        print_status("ALL 模式多進程處理完成（可自行整理成多個 txt）")
    
    else:
        print_status("ALL 模式：無任何任務")

elif mode == 'LATEST':
    # LATEST 模式：先找 O 的最新資料夾名稱
    o_base_path = f"O:\\Image\\{date}"
    print_status(f"\n[O] 掃描路徑 (決定最新資料夾名稱): {o_base_path}")
    
    if not os.path.exists(o_base_path):
        print_status("[O] 路徑不存在，無法繼續")
        sys.exit(1)
    
    try:
        o_subfolders = [f for f in os.listdir(o_base_path) if os.path.isdir(os.path.join(o_base_path, f)) and '_Src' in f]
    except Exception as e:
        print_status(f"[O] 讀取失敗: {e}")
        sys.exit(1)
    
    if not o_subfolders:
        print_status("[O] 無 _Src 資料夾")
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
    
    latest_folder_name = max(o_src_folders, key=lambda x: x[1])[0]
    print_status(f"\n=== 使用最新資料夾名稱 (O drive): {latest_folder_name} ===")
    
    # 對每個 drive 收集任務
    tasks = []
    drive_folder_paths = {}
    
    for drive in drives:
        folder_path = f"{drive}:\\Image\\{date}\\{latest_folder_name}"
        if not os.path.exists(folder_path):
            print_status(f"[{drive}] 無 {latest_folder_name}，IP 維持 -1")
            continue
        
        drive_folder_paths[drive] = folder_path
        target_ips = subfolders_per_drive.get(drive, [])
        
        for ip in target_ips:
            tasks.append((drive, folder_path, ip, image_filename_pattern, blur_threshold, center_x, center_y, roi_width, roi_height))
    
    # 多進程執行
    if tasks:
        print_status(f"\n開始多進程處理 {len(tasks)} 個影像...")
        with Pool(processes=NUM_PROCESSES) as pool:
            ip_results = list(pool.imap(process_single_ip, tasks))
        
        # 彙整結果
        status_dict = {ip: -1 for ip in all_ips}
        variance_dict = {ip: 'N/A' for ip in all_ips}
        
        for res in ip_results:
            if res['error']:
                print_status(f"  {res['ip']} 錯誤: {res['error']}")
            status_dict[res['ip']] = res['status']
            variance_dict[res['ip']] = res['variance']
        
        # 產生 txt
        txt_filename = f"{latest_folder_name}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
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
        
        print_status(f"已產生: {txt_path}")
        
        results.append({
            'folder': latest_folder_name,
            'txt': txt_filename,
            'statuses': status_dict,
            'variances': variance_dict
        })
    
    else:
        print_status("LATEST 模式：無任何有效資料夾")

# =============================================================================
# 結束總覽
# =============================================================================
print("\n" + "="*80)
print(f"總覽（日期: {date}，模式: {mode}）")
print("="*80)

if results:
    print(f"共產生 {len(results)} 個 txt\n")
    for res in results:
        print(f"資料夾: {res['folder']}")
        print(f"txt: {res['txt']}")
        print("狀態總覽:")
        print("IP    | 狀態   | Variance")
        print("------|--------|---------")
        for ip in all_ips:
            s = res['statuses'][ip]
            v = res['variances'][ip]
            status_text = {0: 'Normal', 1: 'Blur', -1: '無/錯誤'}[s]
            print(f"{ip} | {status_text:<6} | {v}")
        print()
else:
    print("無 txt 產生")

print(f"儲存位置: {output_dir}")
print("程式結束")