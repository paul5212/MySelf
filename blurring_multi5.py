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
date = '20260120'                  # 要處理的日期

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

mode = 'all'   # 切換模式: 'all' = 處理所有 _Src 資料夾; 'latest' = 每個 prefix 只處理最新時間資料夾

output_base_dir = './blur_detection_results'
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
def print_status(msg):
    print(msg)

print_status("=== 程式開始執行 ===")
print_status(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_status(f"Python 版本: {sys.version}")
print_status(f"日期: {date}")
print_status(f"模式: {mode} (all=所有資料夾; latest=每個 prefix 只取最新)")
print_status(f"輸出目錄: {output_dir}")
print_status(f"模糊閾值: {blur_threshold}")

try:
    from PIL import Image
    print_status("Pillow 已載入，將優先使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread（建議 pip install pillow）")
    use_pillow = False

# =============================================================================
# 收集所有 _Src 資料夾（用於 'all' 模式）或最新（用於 'latest' 模式）
# =============================================================================
all_src_folders = []  # (drive, folder_name, folder_path, mtime, prefix)
prefix_to_latest = defaultdict(lambda: (0, None, None, None, None))  # prefix -> (mtime, drive, folder_path, folder_name, prefix)

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    print_status(f"\n[{drive}] 掃描: {base_path}")
    
    if not os.path.exists(base_path):
        print_status(f"[{drive}] 路徑不存在，跳過")
        continue
    
    try:
        subfolders = [f for f in os.listdir(base_path)
                      if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
    except Exception as e:
        print_status(f"[{drive}] 讀取失敗: {type(e).__name__}: {e}")
        continue
    
    if not subfolders:
        print_status(f"[{drive}] 無 _Src 資料夾")
        continue
    
    for sub in subfolders:
        if '_' not in sub:
            continue
        prefix = sub.split('_', 1)[0]
        full_path = os.path.join(base_path, sub)
        try:
            mtime = os.path.getmtime(full_path)
        except Exception as e:
            print_status(f"[{drive}] mtime 失敗 ({sub}): {e}")
            continue
        
        # 收集所有，用於 'all'
        all_src_folders.append((drive, sub, full_path, mtime, prefix))
        print_status(f"  發現資料夾: [{drive}] {sub} (mtime: {datetime.fromtimestamp(mtime)})")
        
        # 更新最新，用於 'latest'
        current_mtime, _, _, _, _ = prefix_to_latest[prefix]
        if mtime > current_mtime:
            prefix_to_latest[prefix] = (mtime, drive, full_path, sub, prefix)
            print_status(f"  更新 prefix {prefix} 最新: [{drive}] {sub}")

# 根據模式選擇要處理的資料夾列表
if mode == 'all':
    folders_to_process = sorted(all_src_folders, key=lambda x: x[3])  # 所有，按時間排序
    print_status("\n模式: all - 將處理所有資料夾")
elif mode == 'latest':
    folders_to_process = [(d, fn, fp, mt, p) for p, (mt, d, fp, fn, _) in prefix_to_latest.items()]
    print_status("\n模式: latest - 將處理每個 prefix 的最新資料夾")
else:
    raise ValueError("無效模式: 請設定 mode 為 'all' 或 'latest'")

# =============================================================================
results = []

for drive, folder_name, folder_path, mtime, prefix in folders_to_process:
    print_status(f"\n=== 處理資料夾: {folder_name} (prefix: {prefix}, drive: {drive}, mtime: {datetime.fromtimestamp(mtime)}) ===")
    
    status_dict = {ip: -1 for ip in all_ips}
    variance_dict = {ip: 'N/A' for ip in all_ips}
    
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
            
            print_status(f"  [{drive}-{ip}] shape: {gray.shape}, dtype: {gray.dtype}")
            
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
            
            print_status(f"  [{drive}-{ip}] Variance: {lap_variance:.1f} → {'Blur' if status else 'Normal'}")
            
        except Exception as e:
            print_status(f"  [{drive}-{ip}] 失敗: {type(e).__name__}: {e}")
            variance_dict[ip] = 'Error'
    
    # 產生 txt
    txt_filename = f"{folder_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Prefix: {prefix}\n")
            f.write(f"資料夾: {folder_name} (Drive: {drive})\n")
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
        
        print_status(f"已產生: {txt_path}")
        
        results.append({
            'prefix': prefix,
            'folder': folder_name,
            'txt': txt_filename,
            'statuses': status_dict,
            'variances': variance_dict
        })
        
    except Exception as e:
        print_status(f"寫入 {txt_filename} 失敗: {e}")

# =============================================================================
# 結束時可視化總覽
# =============================================================================
print("\n" + "="*80)
print(f"所有產生的 txt 總覽（日期: {date}, 模式: {mode}）")
print("="*80)

if results:
    print(f"共 {len(results)} 個 txt 產生\n")
    
    for res in results:
        print(f"Prefix: {res['prefix']}")
        print(f"  資料夾: {res['folder']}")
        print(f"  txt 檔: {res['txt']}")
        print("  狀態總覽:")
        
        print("  IP    | 狀態   | Variance")
        print("  ------|--------|---------")
        for ip in all_ips:
            s = res['statuses'][ip]
            v = res['variances'][ip]
            status_text = {0: 'Normal', 1: 'Blur', -1: '無/錯誤'}[s]
            print(f"  {ip} | {status_text:<6} | {v}")
        print()
else:
    print("無 txt 產生（無有效資料夾）")

print(f"所有 txt 儲存位置: {output_dir}")
print("程式結束")