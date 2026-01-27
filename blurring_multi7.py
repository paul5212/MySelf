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
mode = 'LATEST'                    # 'ALL' 或 'LATEST' 模式切換
                                   # ALL: 處理所有 _Src 資料夾，每個產生一個 txt (檔名加 drive 前綴避免覆蓋)
                                   # LATEST: 只處理 O drive 計算出的最新資料夾名稱，對所有 drive 沿用此名稱 (如果存在則處理，否則 -1)，產生1個 txt

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

# =============================================================================
def print_status(msg):
    print(msg)

print_status("=== 程式開始執行 ===")
print_status(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_status(f"Python 版本: {sys.version}")
print_status(f"日期: {date}")
print_status(f"模式: {mode}")
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
results = []  # 收集結果，用於結束時總覽

if mode == 'ALL':
    # ALL 模式: 處理所有 drive 的所有 _Src 資料夾，每個產生一個 txt
    all_src_folders = []  # (drive, folder_name, folder_path, mtime)
    
    for drive in drives:
        base_path = f"{drive}:\\Image\\{date}"
        print_status(f"\n[{drive}] 掃描路徑: {base_path}")
        
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
            full_path = os.path.join(base_path, sub)
            try:
                mtime = os.path.getmtime(full_path)
            except Exception as e:
                print_status(f"[{drive}] mtime 失敗 ({sub}): {e}")
                continue
            all_src_folders.append((drive, sub, full_path, mtime))
            print_status(f"  [{drive}] 發現資料夾: {sub} (mtime: {datetime.fromtimestamp(mtime)})")
    
    for drive, folder_name, folder_path, mtime in all_src_folders:
        print_status(f"\n=== 處理資料夾: [{drive}] {folder_name} ===")
        
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
                print_status(traceback.format_exc())
                variance_dict[ip] = 'Error'
        
        # 產生 txt (加 drive 前綴避免同名覆蓋)
        txt_filename = f"{drive}_{folder_name}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
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
                'drive': drive,
                'folder': folder_name,
                'txt': txt_filename,
                'statuses': status_dict,
                'variances': variance_dict
            })
            
        except Exception as e:
            print_status(f"寫入 {txt_filename} 失敗: {e}")

elif mode == 'LATEST':
    # LATEST 模式: 只處理 O drive 計算出的最新資料夾名稱，對所有 drive 沿用此名稱，產生1個 txt
    o_base_path = f"O:\\Image\\{date}"
    print_status(f"\n[O] 掃描路徑 (用於計算最新資料夾): {o_base_path}")
    
    if not os.path.exists(o_base_path):
        print_status("[O] 路徑不存在，無法計算最新資料夾")
        sys.exit(1)  # 或 continue，但既然是 LATEST，無 O 則停止
    
    try:
        o_subfolders = [f for f in os.listdir(o_base_path)
                        if os.path.isdir(os.path.join(o_base_path, f)) and '_Src' in f]
    except Exception as e:
        print_status(f"[O] 讀取失敗: {type(e).__name__}: {e}")
        sys.exit(1)
    
    if not o_subfolders:
        print_status("[O] 無 _Src 資料夾，無法計算最新")
        sys.exit(1)
    
    o_src_folders = []
    for sub in o_subfolders:
        parts = sub.split('_')
        if len(parts) < 2:
            print_status(f"[O] 檔名格式無效 ({sub})，跳過")
            continue
        time_str = parts[1]  # 假設 _ 後第一部分為時間如 202601201500
        try:
            time_value = int(time_str)  # 轉 int 以比較大小（格式固定 YYYYMMDDHHMM）
        except ValueError:
            print_status(f"[O] 時間解析失敗 ({sub}): {time_str}")
            continue
        full_path = os.path.join(o_base_path, sub)
        o_src_folders.append((sub, full_path, time_value))
        print_status(f"  [O] 發現資料夾: {sub} (時間: {time_str})")
    
    if not o_src_folders:
        print_status("[O] 無有效資料夾，無法繼續")
        sys.exit(1)
    
    # 找時間最大的那個 (最新產出)
    latest_o = max(o_src_folders, key=lambda x: x[2])
    latest_folder_name = latest_o[0]
    print_status(f"\n=== O drive 計算出的最新資料夾 (基於檔名時間): {latest_folder_name} ===")
    
    # 初始化全域 status / variance
    status_dict = {ip: -1 for ip in all_ips}
    variance_dict = {ip: 'N/A' for ip in all_ips}
    
    # 對每個 drive，檢查是否存在該資料夾名稱，並處理
    for drive in drives:
        folder_path = f"{drive}:\\Image\\{date}\\{latest_folder_name}"
        if not os.path.exists(folder_path):
            print_status(f"[{drive}] 無 {latest_folder_name}，其 IP 設為 -1")
            continue
        
        print_status(f"=== 處理 [{drive}] {latest_folder_name} ===")
        
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
                print_status(traceback.format_exc())
                variance_dict[ip] = 'Error'
    
    # 產生單一 txt
    txt_filename = f"{latest_folder_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"最新資料夾 (基於 O drive 檔名時間): {latest_folder_name}\n")
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
            'folder': latest_folder_name,
            'txt': txt_filename,
            'statuses': status_dict,
            'variances': variance_dict
        })
        
    except Exception as e:
        print_status(f"寫入 {txt_filename} 失敗: {e}")

else:
    print_status("無效模式，請設為 'ALL' 或 'LATEST'")
    sys.exit(1)

# =============================================================================
# 結束時可視化總覽
# =============================================================================
print("\n" + "="*80)
print(f"所有將產生的 txt 總覽（日期: {date}，模式: {mode}）")
print("="*80)

if results:
    print(f"共 {len(results)} 個 txt 將產生\n")
    
    for res in results:
        if mode == 'ALL':
            print(f"Drive: {res['drive']}")
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
    print("無 txt 產生")

print(f"所有 txt 儲存位置: {output_dir}")
print("程式結束")