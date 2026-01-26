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

image_filename_pattern = '{sub}_Origin000003.tif'   # 圖像檔名格式（會自動替換 {sub}）

blur_threshold = 1000.0                     # 模糊判斷閾值（建議根據清晰樣本調整 800~1200）

# ROI 設定（假設所有圖像尺寸一致，若不同可再調整）
center_x   = 2500
center_y   = 3500
roi_width  = 1200
roi_height = 800

# 輸出設定
output_base_dir = './blur_detection_results'   # 結果儲存目錄（會自動建立）
show_windows = False                          # 是否即時顯示視窗（批量時建議 False）

# 若只想處理特定 IP（如 ['IP1', 'IP24']），可取消註釋並填入
# ips_to_process = ['IP1', 'IP24']
ips_to_process = None   # None = 處理所有 24 個 IP

# =============================================================================

# 建立輸出目錄（以日期為子資料夾）
output_dir = os.path.join(output_base_dir, date)
os.makedirs(output_dir, exist_ok=True)

results = []

for drive in drives:
    base_path = f"{drive}:\\Image\\{date}"
    
    if not os.path.exists(base_path):
        print(f"[{drive}] 路徑不存在，跳過: {base_path}")
        continue
    
    print(f"\n[{drive}] 開始處理: {base_path}")
    
    # 取得該 drive 對應的所有 IP 子資料夾
    target_subfolders = subfolders_per_drive.get(drive, [])
    
    for subfolder in target_subfolders:
        if ips_to_process is not None and subfolder not in ips_to_process:
            continue
        
        print(f"  [{drive}-{subfolder}] 處理中...")
        
        # 取得所有 _Src 資料夾（群組相同 prefix）
        try:
            all_subdirs = [f for f in os.listdir(base_path) 
                           if os.path.isdir(os.path.join(base_path, f)) and '_Src' in f]
        except Exception as e:
            print(f"  [{drive}-{subfolder}] 讀取資料夾失敗: {e}")
            continue
        
        if not all_subdirs:
            print(f"  [{drive}-{subfolder}] 無符合 _Src 的子資料夾")
            continue
        
        # 以 prefix 群組（例如 001F7CAF）
        prefix_to_folders = defaultdict(list)
        for sub in all_subdirs:
            full_path = os.path.join(base_path, sub)
            if '_' in sub:
                prefix = sub.split('_', 1)[0]
                mtime = os.path.getmtime(full_path)
                prefix_to_folders[prefix].append((mtime, full_path, sub))
        
        if not prefix_to_folders:
            print(f"  [{drive}-{subfolder}] 無有效 prefix 群組")
            continue
        
        # 取每個 prefix 的最新資料夾（最新時間）
        latest_path = None
        latest_name = None
        latest_mtime = 0
        for prefix, folder_list in prefix_to_folders.items():
            folder_list.sort(reverse=True, key=lambda x: x[0])
            mtime, path, name = folder_list[0]
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = path
                latest_name = name
        
        if latest_path is None:
            print(f"  [{drive}-{subfolder}] 無法取得最新資料夾")
            continue
        
        # 動態圖像路徑
        image_filename = image_filename_pattern.format(sub=subfolder)
        image_path = os.path.join(latest_path, subfolder, image_filename)
        
        if not os.path.exists(image_path):
            print(f"  [{drive}-{subfolder}] 圖像不存在（最新資料夾 {latest_name}）: {image_path}")
            continue
        
        print(f"  [{drive}-{subfolder}] 處理圖像: {latest_name}/{subfolder}/{image_filename}")
        
        # 讀取圖像
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [{drive}-{subfolder}] 讀取圖像失敗")
            continue
        
        # 轉灰階
        if len(img.shape) == 3 and img.shape[2] in (3, 4):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        gray_float = gray.astype(np.float32)
        height, width = gray.shape
        
        # ROI 邊界檢查
        roi_x = center_x - (roi_width // 2)
        roi_y = center_y - (roi_height // 2)
        if roi_x < 0: roi_x = 0
        if roi_y < 0: roi_y = 0
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
            status = "Image blur"
            color = (0, 0, 255)      # BGR 紅色
        else:
            status = "Image Normal"
            color = (0, 255, 0)      # BGR 綠色
        
        print(f"  [{drive}-{subfolder}] Laplacian Variance: {lap_variance:.1f} → {status}")
        
        # 準備顯示/儲存圖像
        if gray.dtype != np.uint8:
            gray_display = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            gray_display = gray.copy()
        
        # 畫 ROI 框與文字
        cv2.rectangle(gray_display, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 
                      (0, 255, 0), thickness=16)
        text = f"{drive}-{subfolder} | {status} | Var: {lap_variance:.1f}"
        cv2.putText(gray_display, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    2.5, color, thickness=6)
        
        # 儲存
        save_name = f"{drive}_{subfolder}_{status}_Var{lap_variance:.1f}.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, gray_display)
        print(f"  [{drive}-{subfolder}] 已儲存: {save_path}")
        
        if show_windows:
            cv2.namedWindow(f"Result - {drive}_{subfolder}", cv2.WINDOW_NORMAL)
            cv2.imshow(f"Result - {drive}_{subfolder}", gray_display)
            cv2.waitKey(500)
        
        results.append((drive, subfolder, latest_name, lap_variance, status, save_path))

# 若開啟顯示
if show_windows:
    print("\n按任意鍵關閉所有視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================================================================
# 最終摘要
# =============================================================================
print("\n" + "="*80)
print(f"檢測完成摘要（日期: {date}）")
print("="*80)
if results:
    blur_count = sum(1 for r in results if r[4] == "Image blur")
    print(f"總處理數: {len(results)} 個 IP | 模糊數: {blur_count} 個")
    print("")
    for r in results:
        print(f"{r[0]} - {r[1]} ({r[2]}): {r[4]} (Var: {r[3]:.1f}) → {r[5]}")
else:
    print("無任何圖像被處理（請檢查路徑、磁碟機映射與資料夾）")

print(f"\n所有標註圖像已儲存至: {output_dir}")