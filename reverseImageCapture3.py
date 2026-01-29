import cv2
import numpy as np
import os
import shutil
import sys
from datetime import datetime
import concurrent.futures

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
    sys.exit(0)

use_pillow = False
try:
    from PIL import Image
    print_status("使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread")

# =============================================================================
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
        variance = np.var(np.abs(lap))

        median_gray = float(np.median(roi))

        status = 1 if variance < blur_threshold else 0

        # 畫框
        cv2.rectangle(img_array, (rx, ry), (rx + rw, ry + rh), roi_color, roi_thickness)
        cv2.imwrite(image_path, img_array)

        return status, int(variance), median_gray, 'Success'

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
                'copy': 'Failed (No Image)'
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
                'variance': str(var) if var != 'Error' else var,
                'median': f"{med:.1f}" if med != 'Error' else med,
                'copy': 'Success' if copy_msg == 'Success' else copy_msg
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
                'copy': f'Failed ({str(e)})'
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

# 解析輸入 txt 並分析行數
parsed_entries = []
total_lines = 0
valid_lines = 0

if os.path.exists(parse_txt_path):
    try:
        with open(parse_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)

        temp_entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            valid_lines += 1

            parts = line.split(',')
            if len(parts) != 6:
                print_status(f"無效行: {line}")
                continue

            _, _, ip_str, slice_str, center_x_str, center_y_str = [p.strip() for p in parts]

            ip = ip_str.upper()
            if ip.startswith('IP') and len(ip) == 3:
                ip = f"IP0{ip[2:]}"
            elif not ip.startswith('IP') or len(ip) != 4:
                print_status(f"無效 IP 格式: {ip_str}")
                continue

            slice_str_clean = slice_str.replace('Slice', '').strip()
            try:
                slice_num = int(slice_str_clean)
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

            temp_entries.append({
                'ip': ip,
                'slice_num': slice_num,
                'center_x': center_x,
                'center_y': center_y,
                'drive': drive_for_ip[ip]
            })

        print_status(f"輸入檔案總行數：{total_lines}")
        print_status(f"去除空白行後有效行數：{valid_lines}")
        print_status(f"有效解析筆數：{len(temp_entries)}")

        # 排序 temp_entries 按 IP → Slice → Y軸
        temp_entries.sort(key=lambda e: (
            int(e['ip'][2:]),  # IP01~24 數字
            e['slice_num'],    # 同 IP Slice 小→大
            e['center_y']      # 同 Slice Y 小→大 (上→下)
        ))
        print_status("已按 IP → Slice → Y軸 (上→下) 排序完成")

        # 排序後給 entry_id 001, 002, ...
        for idx, e in enumerate(temp_entries, start=1):
            e['entry_id'] = f"{idx:03d}"
            parsed_entries.append(e)

    except Exception as e:
        print_status(f"解析失敗: {e}")
        sys.exit(1)

    # 按 drive 分組處理
    entries_by_drive = {}
    for e in parsed_entries:
        d = e['drive']
        entries_by_drive.setdefault(d, []).append(e)

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for drive, entries in entries_by_drive.items():
            folder_path_base = f"{drive}:\\Image\\{date}"
            futures.append(executor.submit(process_entries, entries, folder_name, folder_path_base))

        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

else:
    print_status("無解析檔案，程式結束")
    sys.exit(0)

# =============================================================================
# 輸出報告 txt（按 entry_id 由小到大，即排序後順序）
# =============================================================================
all_results.sort(key=lambda r: int(r['entry_id']))

report_filename = f"{folder_name}_report.txt"
report_path = os.path.join(output_dir, folder_name, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"資料夾: {folder_name}\n")
    f.write(f"解析來源: {os.path.basename(parse_txt_path)}\n")
    f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"總行數: {total_lines}   有效行數: {valid_lines}   解析筆數: {len(parsed_entries)}\n")
    f.write(f"排序規則: IP01→IP24 → 同IP Slice小→大 → 同Slice Y小→大(上→下)\n")
    f.write(f"ROI: {roi_width}×{roi_height}  Threshold: {blur_threshold}\n")
    f.write("=" * 90 + "\n")
    
    f.write(f"{'#':<4} {'IP':<6} {'Slice':<6} {'Center':<15} {'Status':<8} {'Variance':<10} {'Median':<8} {'Copy':<12}\n")
    f.write("-" * 80 + "\n")

    blur_count = 0
    for r in all_results:
        status_str = {0: 'Normal', 1: 'Blur', -1: 'Error'}.get(r['status'], 'Unknown')
        if r['status'] == 1:
            blur_count += 1
        f.write(f"{r['entry_id']:<4} {r['ip']:<6} {r['slice']:<6} {r['center']:<15} {status_str:<8} {r['variance']:<10} {r['median']:<8} {r['copy']:<12}\n")

    f.write("=" * 90 + "\n")
    f.write(f"總筆數: {len(all_results)}    模糊數: {blur_count}\n")

print_status(f"報告已產生: {report_path}")
print_status("程式結束")