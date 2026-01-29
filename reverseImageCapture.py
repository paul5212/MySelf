import cv2
import numpy as np
import os
import shutil
import sys
from datetime import datetime
import concurrent.futures
import threading

# =============================================================================
# 參數設定區（請自行調整）
# =============================================================================
date = '20260120'

# 只處理最新資料夾，選擇要分析的資料夾索引 (0=最新, 1=第二新, ...)
latest_index = 0

drives = ['O', 'P', 'Q', 'R', 'S', 'T']

subfolders_per_drive = {
    'O': ['IP01', 'IP02', 'IP03', 'IP04'],
    'P': ['IP05', 'IP06', 'IP07', 'IP08'],
    'Q': ['IP09', 'IP10', 'IP11', 'IP12'],
    'R': ['IP13', 'IP14', 'IP15', 'IP16'],
    'S': ['IP17', 'IP18', 'IP19', 'IP20'],
    'T': ['IP21', 'IP22', 'IP23', 'IP24'],
}

drive_for_ip = {}
for drive, ips in subfolders_per_drive.items():
    for ip in ips:
        drive_for_ip[ip] = drive

all_ips = [f'IP{i:02d}' for i in range(1, 25)]

image_filename_pattern = '{ip}_Origin{slice_num:06d}.tif'  # 修改為動態 slice_num

blur_threshold = 1000.0

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

# 新增參數：輸入的解析 txt 檔案（如果存在則啟用解析模式）
parse_txt_path = './20260120_001F148A_T550QVN10_TGT_G_Src.txt'  # 請調整為實際路徑

# ROI 可視化顏色（紅色，BGR 格式）
roi_color = (0, 0, 255)  # 紅色
roi_thickness = 2

# =============================================================================
def print_status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

print_status("=== 程式開始執行 ===")
print_status(f"模式: LATEST (只處理最新/指定索引資料夾)")
print_status(f"選擇索引: {latest_index}")
print_status(f"ROI 大小: {roi_width} x {roi_height}")
print_status(f"日期: {date}")
print_status(f"輸出目錄: {output_dir}")
print_status(f"模糊閾值: {blur_threshold}")
print_status(f"最大並行執行緒數: {MAX_WORKERS}")

if os.path.exists(parse_txt_path):
    print_status(f"偵測到解析檔案: {parse_txt_path}，啟用解析與複製模式")
else:
    print_status("無解析檔案，執行標準模糊檢測模式")

use_pillow = False
try:
    from PIL import Image
    print_status("Pillow 已載入，將優先使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread")

# =============================================================================
lock = threading.Lock()

def compute_blur_and_median(image_path, center_x, center_y, ip, visualize_roi=False):
    try:
        if use_pillow:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
        else:
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise Exception("cv2.imread 失敗")
        
        # 確保是 RGB 或 BGR 以畫彩色矩形
        if len(img_array.shape) == 2:  # 灰階
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:  # RGBA → BGR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32)
        height, width = gray.shape
        
        offset_x = offsetX_dict.get(ip, 0)
        offset_y = offsetY_dict.get(ip, 0)
        center_x_actual = center_x + offset_x
        center_y_actual = center_y + offset_y
        
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
        status_text = 'Blur' if status else 'Normal'
        
        if visualize_roi:
            # 畫 ROI 矩形到 img_array
            cv2.rectangle(img_array, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), roi_color, roi_thickness)
            # 儲存回原檔案（覆蓋）
            cv2.imwrite(image_path, img_array)
            print_status(f"  已畫 ROI 矩形並儲存: {image_path}")
        
        return status, lap_variance, median_gray, status_text
    
    except Exception as e:
        print_status(f"  計算失敗: {type(e).__name__}: {e}")
        return -1, 'Error', 'Error', 'Error'

def process_drive(drive, folder_name, status_dict, variance_dict, median_dict, copy_success_dict, parsed_entries=None):
    folder_path = f"{drive}:\\Image\\{date}\\{folder_name}"
    if not os.path.exists(folder_path):
        print_status(f"[{drive}] 無 {folder_name}，跳過")
        return

    print_status(f"[{drive}] 開始處理 {folder_name}")
    
    target_ips = subfolders_per_drive.get(drive, [])
    
    if parsed_entries:  # 解析模式
        for entry in parsed_entries:
            if entry['ip'] not in target_ips:
                continue
            
            ip = entry['ip']
            slice_num = entry['slice_num']
            center_x = entry['center_x']
            center_y = entry['center_y']
            
            image_filename = image_filename_pattern.format(ip=ip, slice_num=slice_num)
            image_path = os.path.join(folder_path, ip, image_filename)
            
            if not os.path.exists(image_path):
                print_status(f"  [{drive}-{ip}] 圖像不存在: {image_path}")
                with lock:
                    copy_success_dict[ip] = 'Failed (No Image)'
                continue
            
            # 複製圖片到新資料夾
            new_folder = os.path.join(output_dir, folder_name)
            os.makedirs(new_folder, exist_ok=True)
            new_image_path = os.path.join(new_folder, image_filename)
            
            try:
                shutil.copy(image_path, new_image_path)
                print_status(f"  [{drive}-{ip}] 複製成功: {new_image_path}")
                copy_success = 'Success'
                
                # 計算模糊與灰階（使用新中心點），並畫 ROI 矩形
                status, variance, median, status_text = compute_blur_and_median(new_image_path, center_x, center_y, ip, visualize_roi=True)
                
                with lock:
                    status_dict[ip] = status
                    variance_dict[ip] = f"{variance:.1f}" if isinstance(variance, float) else variance
                    median_dict[ip] = f"{median:.1f}" if isinstance(median, float) else median
                    copy_success_dict[ip] = copy_success
                
                print_status(f"  [{drive}-{ip}] Center: ({center_x},{center_y}) | Var: {variance_dict[ip]} | Med: {median_dict[ip]} → {status_text}")
            
            except Exception as e:
                print_status(f"  [{drive}-{ip}] 複製失敗: {e}")
                with lock:
                    copy_success_dict[ip] = f'Failed ({str(e)})'
    
    else:  # 標準模式
        for ip in target_ips:
            image_filename = image_filename_pattern.format(ip=ip, slice_num=3)  # 標準模式固定使用 000003
            image_path = os.path.join(folder_path, ip, image_filename)
            
            if not os.path.exists(image_path):
                print_status(f"  [{drive}-{ip}] 圖像不存在: {image_path}")
                continue
            
            print_status(f"  [{drive}-{ip}] 讀取: {image_path}")
            
            # 標準模式使用固定中心點
            base_center_x = 2500
            base_center_y = 3500
            status, variance, median, status_text = compute_blur_and_median(image_path, base_center_x, base_center_y, ip)
            
            with lock:
                status_dict[ip] = status
                variance_dict[ip] = f"{variance:.1f}" if isinstance(variance, float) else variance
                median_dict[ip] = f"{median:.1f}" if isinstance(median, float) else median
            
            print_status(f"  [{drive}-{ip}] Center: ({base_center_x},{base_center_y}) | Var: {variance_dict[ip]} | Med: {median_dict[ip]} → {status_text}")

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
# 只處理指定的單一資料夾（LATEST 模式）
# =============================================================================
try:
    selected_folder = o_src_folders[latest_index]
    folder_name, mtime = selected_folder
    print_status(f"\n開始處理 Index {latest_index} 的資料夾：{folder_name}")
except IndexError:
    print_status(f"錯誤: Index {latest_index} 超出範圍 (可用 0 ~ {len(o_src_folders)-1})")
    sys.exit(1)

# =============================================================================
# 新增：解析輸入 txt 檔案（如果存在）
# =============================================================================
parsed_entries = []
if os.path.exists(parse_txt_path):
    try:
        with open(parse_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                print_status(f"無效行: {line}")
                continue
            
            # 解析: ignore1, ignore2, ip_str, slice_str, center_x, center_y
            _, _, ip_str, slice_str, center_x_str, center_y_str = parts
            ip = ip_str.strip().upper().replace('IP', 'IP')  # 確保 IPXX 格式
            if len(ip) == 3:  # IP8 → IP08
                ip = f"IP0{ip[2:]}"
            elif len(ip) != 4:
                print_status(f"無效 IP: {ip_str}")
                continue
            
            slice_num_str = slice_str.strip().replace('Slice', '')
            try:
                slice_num = int(slice_num_str)
            except ValueError:
                print_status(f"無效 Slice: {slice_str}")
                continue
            
            try:
                center_x = int(center_x_str)
                center_y = int(center_y_str)
            except ValueError:
                print_status(f"無效中心點: {center_x_str},{center_y_str}")
                continue
            
            if ip not in drive_for_ip:
                print_status(f"未知 IP: {ip}")
                continue
            
            parsed_entries.append({
                'ip': ip,
                'slice_num': slice_num,
                'center_x': center_x,
                'center_y': center_y,
                'drive': drive_for_ip[ip]
            })
        
        print_status(f"解析完成，共 {len(parsed_entries)} 筆條目")
    
    except Exception as e:
        print_status(f"解析 {parse_txt_path} 失敗: {e}")
        sys.exit(1)

# =============================================================================
# 開始處理
# =============================================================================
status_dict = {ip: -1 for ip in all_ips}
variance_dict = {ip: 'N/A' for ip in all_ips}
median_dict = {ip: 'N/A' for ip in all_ips}
copy_success_dict = {ip: 'N/A' for ip in all_ips}  # 新增複製成功資訊

# 平行處理 6 個 drive
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    if parsed_entries:
        # 解析模式：按 drive 分組 parsed_entries
        entries_by_drive = {drive: [] for drive in drives}
        for entry in parsed_entries:
            entries_by_drive[entry['drive']].append(entry)
        
        futures = [
            executor.submit(process_drive, drive, folder_name, status_dict, variance_dict, median_dict, copy_success_dict, entries_by_drive[drive])
            for drive in drives if entries_by_drive[drive]
        ]
    else:
        # 標準模式
        futures = [
            executor.submit(process_drive, drive, folder_name, status_dict, variance_dict, median_dict, copy_success_dict)
            for drive in drives
        ]
    concurrent.futures.wait(futures)

# 產生 txt
txt_filename = f"{folder_name}.txt"
txt_path = os.path.join(output_dir, txt_filename)

try:
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"資料夾: {folder_name}\n")
        f.write(f"處理順序: Index {latest_index} (基於 O drive 排序)\n")
        f.write(f"日期: {date}\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if parsed_entries:
            f.write(f"解析輸入: {parse_txt_path}\n")
        else:
            f.write(f"基準中心: (2500, 3500)\n")  # 標準模式顯示固定中心
        f.write(f"模糊閾值: {blur_threshold}\n")
        f.write("="*60 + "\n")
        
        for ip in all_ips:
            status = status_dict[ip]
            var = variance_dict[ip]
            med = median_dict[ip]
            copy_success = copy_success_dict[ip]
            status_str = {0: '0 (Normal)', 1: '1 (Blur)', -1: '-1 (無圖像或錯誤)'}[status]
            f.write(f"{ip}: {status_str} | Variance: {var} | Median Gray: {med} | Copy Success: {copy_success}\n")
        
        f.write("\n狀態數字一行 (0=Normal, 1=Blur, -1=無/錯誤):\n")
        f.write(' '.join(str(status_dict[ip]) for ip in all_ips) + "\n")
        
        f.write("\nVariance 值一行:\n")
        f.write(' '.join(str(v) for v in variance_dict.values()) + "\n")
        
        f.write("\nMedian Gray 值一行:\n")
        f.write(' '.join(str(m) for m in median_dict.values()) + "\n")
        
        f.write("\nCopy Success 一行:\n")
        f.write(' '.join(copy_success_dict[ip] for ip in all_ips) + "\n")
    
    print_status(f"已完成並產生: {txt_path}")

except Exception as e:
    print_status(f"寫入 {txt_filename} 失敗: {e}")

# =============================================================================
# 結束總覽
# =============================================================================
print("\n" + "="*80)
print(f"執行完成（只處理單一資料夾）")
print("="*80)

print(f"已產生 1 個 txt\n")
print(f"資料夾: {folder_name}")
print(f"txt 檔: {txt_filename}")
print("  狀態總覽 (部分顯示):")
print("  IP    | 狀態   | Variance | Median Gray | Copy Success")
print("  ------|--------|----------|-------------|-------------")
for ip in all_ips[:8]:  # 只顯示前 8 個避免太長
    s = status_dict[ip]
    v = variance_dict[ip]
    m = median_dict[ip]
    c = copy_success_dict[ip]
    status_text = {0: 'Normal', 1: 'Blur', -1: '無/錯誤'}[s]
    print(f"  {ip} | {status_text:<6} | {v:<8} | {m:<11} | {c}")
print("  ... (共 24 個 IP)")
print()

print(f"輸出儲存位置: {output_dir}")
print("程式結束")