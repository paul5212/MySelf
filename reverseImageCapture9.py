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

# === 新增輸出控制選項 ===
save_big_image = True      # 是否輸出大圖（原圖或加框圖）
draw_roi_box   = True      # 是否在大圖畫紅色 ROI 框（僅在 save_big_image=True 時有效）
save_roi       = True      # 是否輸出固定1200×800的ROI小圖（不含紅框）

# =============================================================================
def print_status(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")

print_status("=== 程式開始執行 ===")
print_status(f"日期: {date}")
print_status(f"選擇索引: {latest_index}")
print_status(f"ROI: {roi_width}×{roi_height} @ threshold {blur_threshold}")
print_status(f"輸出大圖: {'是' if save_big_image else '否'}")
print_status(f"　　└─ 大圖畫框: {'是' if draw_roi_box else '否'}")
print_status(f"輸出ROI小圖: {'是' if save_roi else '否'}")

if os.path.exists(parse_txt_path):
    print_status(f"解析模式啟用: {parse_txt_path}")
else:
    print_status("無解析檔案，程式結束")
    sys.exit(0)

use_pillow = False
try:
    from PIL import Image
    print_status("使用 Pillow 讀取 .tif")
    use_pillow = True
except ImportError:
    print_status("Pillow 未安裝，使用 cv2.imread")

# =============================================================================
def compute_blur_only(image_path, center_x, center_y, ip, slice_num):
    """只計算 blur（不畫框、不產生 ROI）"""
    try:
        if use_pillow:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
        else:
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise Exception("讀取失敗")

        if len(img_array.shape) == 3 and img_array.shape[2] > 1:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img_array.astype(np.float32)

        h, w = gray.shape

        cx = center_x + offsetX_dict.get(ip, 0)
        cy = center_y + offsetY_dict.get(ip, 0)

        ideal_x = cx - roi_width // 2
        ideal_y = cy - roi_height // 2

        start_x = max(0, ideal_x)
        start_y = max(0, ideal_y)
        end_x = min(w, ideal_x + roi_width)
        end_y = min(h, ideal_y + roi_height)

        crop_w = end_x - start_x
        crop_h = end_y - start_y

        if crop_w <= 0 or crop_h <= 0:
            raise Exception("ROI 完全超出邊界")

        roi_gray = gray[start_y:end_y, start_x:end_x]
        lap = cv2.Laplacian(roi_gray, cv2.CV_32F, ksize=3)
        variance = np.var(np.abs(lap))
        median_gray = float(np.median(roi_gray))
        status = 1 if variance < blur_threshold else 0

        return status, int(variance), median_gray, 'Success'

    except Exception as e:
        print_status(f"計算 blur 失敗 {os.path.basename(image_path)}: {e}")
        return -1, 'Error', 'Error', str(e)

def compute_blur_and_process(image_path, center_x, center_y, ip, slice_num, draw_box=False):
    """完整處理（可選畫框 + 產生 ROI 固定圖）"""
    try:
        if use_pillow:
            pil_img = Image.open(image_path)
            img_array = np.array(pil_img)
        else:
            img_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img_array is None:
                raise Exception("讀取失敗")

        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        cx = center_x + offsetX_dict.get(ip, 0)
        cy = center_y + offsetY_dict.get(ip, 0)

        ideal_x = cx - roi_width // 2
        ideal_y = cy - roi_height // 2

        start_x = max(0, ideal_x)
        start_y = max(0, ideal_y)
        end_x = min(w, ideal_x + roi_width)
        end_y = min(h, ideal_y + roi_height)

        crop_w = end_x - start_x
        crop_h = end_y - start_y

        if crop_w <= 0 or crop_h <= 0:
            raise Exception("ROI 完全超出邊界")

        roi_gray = gray[start_y:end_y, start_x:end_x]
        lap = cv2.Laplacian(roi_gray, cv2.CV_32F, ksize=3)
        variance = np.var(np.abs(lap))
        median_gray = float(np.median(roi_gray))
        status = 1 if variance < blur_threshold else 0

        roi_crop = img_array[start_y:end_y, start_x:end_x].copy()

        roi_fixed = np.zeros((roi_height, roi_width, img_array.shape[2]), dtype=img_array.dtype)

        paste_x = max(0, -ideal_x)
        paste_y = max(0, -ideal_y)

        roi_fixed[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = roi_crop

        if draw_box:
            cv2.rectangle(img_array, (start_x, start_y), (end_x, end_y), roi_color, roi_thickness)
            cv2.imwrite(image_path, img_array)

        return status, int(variance), median_gray, 'Success', roi_fixed

    except Exception as e:
        print_status(f"處理 {os.path.basename(image_path)} 失敗: {e}")
        return -1, 'Error', 'Error', str(e), None

# =============================================================================
def process_entries(entries, folder_name, folder_path_base):
    results = []

    big_output_folder = os.path.join(output_dir, folder_name)
    roi_output_folder = os.path.join(output_dir, folder_name, 'ROI')
    if save_big_image:
        os.makedirs(big_output_folder, exist_ok=True)
    if save_roi:
        os.makedirs(roi_output_folder, exist_ok=True)

    for entry in entries:
        entry_id = entry['entry_id']
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

        base_name_no_ext = os.path.splitext(orig_filename)[0]
        ext = os.path.splitext(orig_filename)[1]
        big_filename = f"{entry_id}_{base_name_no_ext}{ext}"
        roi_filename = f"{entry_id}_{base_name_no_ext}_ROI{ext}"

        big_path = None
        copy_status = 'No Copy'

        # 處理大圖輸出
        if save_big_image:
            big_path = os.path.join(big_output_folder, big_filename)
            shutil.copy(orig_path, big_path)
            copy_status = f"Success ({'Box' if draw_roi_box else 'No Box'})"
            print_status(f"複製大圖: {big_filename} ({'加框' if draw_roi_box else '原圖'})")

        # 決定處理路徑與是否畫框
        process_path = big_path if save_big_image else orig_path
        need_full_process = save_big_image or save_roi

        if need_full_process:
            status, var, med, msg, roi_fixed = compute_blur_and_process(
                process_path, cx, cy, ip, slice_num, draw_box=draw_roi_box and save_big_image
            )

            if save_roi and roi_fixed is not None:
                roi_path = os.path.join(roi_output_folder, roi_filename)
                cv2.imwrite(roi_path, roi_fixed)
                print_status(f"ROI 截圖儲存: {roi_filename}")

        else:
            status, var, med, msg = compute_blur_only(orig_path, cx, cy, ip, slice_num)

        results.append({
            'entry_id': entry_id,
            'ip': ip,
            'slice': slice_num,
            'center': f"({cx},{cy})",
            'status': status,
            'variance': str(var) if var != 'Error' else var,
            'median': f"{med:.1f}" if med != 'Error' else med,
            'copy': copy_status
        })

        print_status(f"{ip} Slice{slice_num} → Var={results[-1]['variance']}, Med={results[-1]['median']}, {'Blur' if status==1 else 'Normal'}")

    return results

# =============================================================================
# 主流程（其餘不變）
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

# 解析 txt ...
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

        temp_entries.sort(key=lambda e: (
            int(e['ip'][2:]),
            e['slice_num'],
            e['center_y']
        ))
        print_status("已按 IP → Slice → Y軸 (上→下) 排序完成")

        for idx, e in enumerate(temp_entries, start=1):
            e['entry_id'] = f"{idx:03d}"
            parsed_entries.append(e)

    except Exception as e:
        print_status(f"解析失敗: {e}")
        sys.exit(1)

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
# 輸出報告（永遠產生）
# =============================================================================
all_results.sort(key=lambda r: int(r['entry_id']))

os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
report_filename = f"{folder_name}_report.txt"
report_path = os.path.join(output_dir, folder_name, report_filename)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"資料夾: {folder_name}\n")
    f.write(f"解析來源: {os.path.basename(parse_txt_path)}\n")
    f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"總行數: {total_lines}   有效行數: {valid_lines}   解析筆數: {len(parsed_entries)}\n")
    f.write(f"排序規則: IP01→IP24 → 同IP Slice小→大 → 同Slice Y小→大(上→下)\n")
    f.write(f"ROI: {roi_width}×{roi_height}  Threshold: {blur_threshold}\n")
    f.write(f"輸出大圖: {'是' if save_big_image else '否'}\n")
    f.write(f"　　└─ 大圖畫框: {'是' if draw_roi_box else '否'}\n")
    f.write(f"輸出ROI小圖: {'是' if save_roi else '否'}\n")
    f.write(f"ROI小圖檔名規則: {{entry_id}}_{{原檔名無副檔名}}_ROI.tif\n")
    f.write("=" * 90 + "\n")
    
    f.write(f"{'#':<4} {'IP':<6} {'Slice':<6} {'Center':<15} {'Status':<8} {'Variance':<10} {'Median':<8} {'Copy':<20}\n")
    f.write("-" * 90 + "\n")

    blur_count = 0
    for r in all_results:
        status_str = {0: 'Normal', 1: 'Blur', -1: 'Error'}.get(r['status'], 'Unknown')
        if r['status'] == 1:
            blur_count += 1
        f.write(f"{r['entry_id']:<4} {r['ip']:<6} {r['slice']:<6} {r['center']:<15} {status_str:<8} {r['variance']:<10} {r['median']:<8} {r['copy']:<20}\n")

    f.write("=" * 90 + "\n")
    f.write(f"總筆數: {len(all_results)}    模糊數: {blur_count}\n")

print_status(f"報告已產生: {report_path}")
print_status("程式結束")