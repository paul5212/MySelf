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

image_filename_pattern = '{ip}_Origin{slice_num:06d}.tif'

blur_threshold = 1000.0
roi_width  = 1200
roi_height = 800

offsetX_dict = {f'IP{i:02d}': 0 for i in range(1, 25)}
offsetY_dict = {f'IP{i:02d}': 0 for i in range(1, 25)}

output_base_dir = './blur_detection_results'
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

MAX_WORKERS = min(6, len(drives))

parse_txt_path = './20260120_001F148A_T550QVN10_TGT_G_Src.txt'  # 請調整為實際路徑

roi_color = (0, 0, 255)     # 紅色
roi_thickness = 2

# =============================================================================
def print_status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

print_status("=== 程式開始執行 ===")
print_status(f"日期: {date}")
print_status(f"選擇索引: {latest_index}")
print_status(f"ROI: {roi_width}×{roi_height} @ threshold {blur_threshold}")

if os.path.exists(parse_txt_path):
    print_status(f"解析模式啟用: {parse_txt_path}")
else:
    print_status("無解析檔案，執行標準模式（僅最新 _Src 的固定 slice 3）")

use_pillow = False
try:
    from PIL import Image
    print_status("使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread")

# =============================================================================
lock = threading.Lock()

def compute_blur_and_draw_roi(image_path, center_x, center_y, ip, slice_num):
    try:
        if use_pillow:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
        else:
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise Exception("讀取失敗")

        # 轉成 BGR 以畫彩色框
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        cx = center_x + offsetX_dict.get(ip, 0)
        cy = center_y + offsetY_dict.get(ip, 0)

        rx = max(0, cx - roi_width // 2)
        ry = max(0, cy - roi_height // 2)
        rw = min(roi_width, w - rx)
        rh = min(roi_height, h - ry)

        if rw <= 0 or rh <= 0:
            raise Exception("ROI 無效")

        roi = gray[ry:ry+rh, rx:rx+rw]
        lap = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
        variance = np.var(np.abs(lap))   # 更常見的寫法，也可以用 stddev**2

        median_gray = float(np.median(roi))

        status = 1 if variance < blur_threshold else 0

        # 畫框
        cv2.rectangle(img_array, (rx, ry), (rx + rw, ry + rh), roi_color, roi_thickness)
        cv2.imwrite(image_path, img_array)

        return status, variance, median_gray, 'Success'

    except Exception as e:
        print_status(f"處理 {os.path.basename(image_path)} 失敗: {e}")
        return -1, 'Error', 'Error', str(e)

def process_entries(entries, folder_name, folder_path_base):
    results = []

    for entry in entries:
        entry_id = entry['entry_id']  # 001, 002, ...
        ip = entry['ip']
        slice_num = entry['slice_num']
        cx = entry['center_x']
        cy = entry['center_y']
        drive = entry['drive']

        folder_path = os.path.join(folder_path_base, folder_name)
        orig_filename = image_filename_pattern.format(ip=ip, slice_num=slice_num)
        orig_path = os.path.join(folder_path, ip, orig_filename)

        if not os.path.exists(orig_path):
            results.append({
                'entry_id': entry_id,
                'ip': ip,
                'slice': slice_num,
                'center': f"({cx},{cy})",
                'status': -1,
                'variance': 'N/A',
                'median': 'N/A',
                'copy': 'Failed (No Image)',
                'output_file': 'N/A'
            })
            print_status(f"{ip} Slice{slice_num} 圖檔不存在")
            continue

        # 產生唯一檔名：001_IP08_Origin000006.tif
        base_name, ext = os.path.splitext(orig_filename)
        output_filename = f"{entry_id}_{base_name}{ext}"

        new_folder = os.path.join(output_dir, folder_name)
        os.makedirs(new_folder, exist_ok=True)
        new_path = os.path.join(new_folder, output_filename)

        try:
            shutil.copy(orig_path, new_path)
            print_status(f"複製成功: {output_filename}")

            status, var, med, copy_msg = compute_blur_and_draw_roi(new_path, cx, cy, ip, slice_num)

            results.append({
                'entry_id': entry_id,
                'ip': ip,
                'slice': slice_num,
                'center': f"({cx},{cy})",
                'status': status,
                'variance': str(int(var)) if isinstance(var, (int, float)) else var,
                'median': f"{med:.1f}" if isinstance(med, (int, float)) else med,
                'copy': 'Success' if copy_msg == 'Success' else copy_msg,
                'output_file': output_filename
            })

            print_status(f"{ip} Slice{slice_num} → Var={results[-1]['variance']}, Med={results[-1]['median']}, {'Blur' if status==1 else 'Normal'}")

        except Exception as e:
            results.append({
                'entry_id': entry_id,
                'ip': ip,
                'slice': slice_num,
                'center': f"({cx},{cy})",
                'status': -1,
                'variance': 'Error',
                'median': 'Error',
                'copy': f'Failed ({str(e)})',
                'output_file': 'N/A'
            })

    return results

# =============================================================================
# 主流程
# =============================================================================

o_base_path = f"O:\\Image\\{date}"
if not os.path.exists(o_base_path):
    print_status("O: 路徑不存在")
    sys.exit(1)

o_subfolders = [f for f in os.listdir(o_base_path) if os.path.isdir(os.path.join(o_base_path, f)) and '_Src' in f]
o_src_folders = []
for sub in o_subfolders:
    try:
        mtime = os.path.getmtime(os.path.join(o_base_path, sub))
        o_src_folders.append((sub, mtime))
    except:
        pass

if not o_src_folders:
    print_status("無有效 _Src 資料夾")
    sys.exit(1)

o_src_folders.sort(key=lambda x: x[1], reverse=True)
folder_name = o_src_folders[latest_index][0]
print_status(f"處理資料夾: {folder_name}")

# 解析輸入 txt 並加入行號
parsed_entries = []
if os.path.exists(parse_txt_path):
    try:
        with open(parse_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                print_status(f"無效行 (行 {line_idx}): {line}")
                continue

            _, _, ip_str, slice_str, center_x_str, center_y_str = [p.strip() for p in parts]

            # IP 格式修正
            ip = ip_str.upper()
            if ip.startswith('IP') and len(ip) == 3:  # IP8 → IP08
                ip = f"IP0{ip[2:]}"
            elif not ip.startswith('IP') or len(ip) != 4:
                print_status(f"無效 IP 格式 (行 {line_idx}): {ip_str}")
                continue

            # Slice
            slice_str_clean = slice_str.replace('Slice', '').strip()
            try:
                slice_num = int(slice_str_clean)
            except ValueError:
                print_status(f"無效 Slice (行 {line_idx}): {slice_str}")
                continue

            # 中心點
            try:
                center_x = int(center_x_str)
                center_y = int(center_y_str)
            except ValueError:
                print_status(f"無效中心點 (行 {line_idx}): {center_x_str},{center_y_str}")
                continue

            if ip not in drive_for_ip:
                print_status(f"未知 IP (行 {line_idx}): {ip}")
                continue

            parsed_entries.append({
                'line_num': line_idx,
                'entry_id': f"{line_idx:03d}",
                'ip': ip,
                'slice_num': slice_num,
                'center_x': center_x,
                'center_y': center_y,
                'drive': drive_for_ip[ip]
            })

        print_status(f"解析完成，共 {len(parsed_entries)} 筆有效條目")

    except Exception as e:
        print_status(f"解析失敗: {e}")
        sys.exit(1)

    # 排序