# MySelf
開發目的精簡 LIST監測當天批量生產的環境穩定度（同一日期內 IP01~IP24 的 variance 一致性高 → 環境穩定）
比較不同日期的環境穩定性（跨天 variance 變化大 → 環境不穩定）
找出特定 IPXX 在不同天 variance 差異過大的情況（例如 IP03、IP05 等異常波動）
預防因環境變化導致 AOI 參數失效（variance 過低觸發模糊警報）
提供量化數據支持 AOI 參數調整與長期趨勢追蹤

type %USERPROFILE%\.ssh\id_ed25519.pub | ssh pi@10.131.74.239 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

Main Line

-Station 1: Midplane - Bay Side

-Station 2: Midplane - Strata Side

-S6: Final Assembly

-AOI_BAY_INS

-------------------------------------------------
Strata Sub Line

-Station 3: Subassemblies - Strata

-------------------------------------------------
Bay C Sub Line

-Station 4: Subassemblies - Bay C

-------------------------------------------------
Bay B Sub Line

-Station 5: Subassemblies - Bay B

-------------------------------------------------
Midplane Sub Line

-Station: Subassemblies - Midplane

-------------------------------------------------
以上正確嗎?
 
 
Strata FA 和 Bay FA 這是哪條線的???

---

import requests, json, math, time
import pandas as pd

BASE = "http://10.28.81.92:8080/api/Business/business_apiLog/GetTaskLogs"
HEADERS = {
    "Authorization": "Bearer 你的token",  # 從 Request headers 複製
    "lang": "zh-TW",                      # 依實際值
}
PAGE_SIZE = 200  # 先試大一點，若被限制就改回 20

# 篩選參數名稱請依實際 URL 修改
FILTERS = {
    "ProductionLineCode": "L6-L1-2",
    "TrayNo": "B49E56DFAD0108E0",
    "Result": "true",
}

def fetch(page):
    params = {"pageindex": page, "pagesize": PAGE_SIZE, **FILTERS}
    r = requests.get(BASE, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["response"]

first = fetch(1)
total = first["dataCount"]
pages = math.ceil(total / PAGE_SIZE)
print(f"共 {total} 條，{pages} 頁")

rows = first["data"]
for p in range(2, pages + 1):
    rows += fetch(p)["data"]
    time.sleep(0.2)

# 展開 Request / Response 的 JSON 字串
def parse(s):
    try:
        return json.loads(s)
    except Exception:
        return {}

for row in rows:
    req, resp = parse(row.get("Request")), parse(row.get("Response"))
    row["Req_DeviceNo"] = req.get("DeviceNo")
    row["Req_StationNo"] = req.get("StationNo")
    row["Resp_status"] = resp.get("status")
    row["Resp_msg"] = resp.get("msg")

df = pd.DataFrame(rows)
df.to_excel("taskRecordLog.xlsx", index=False)
print("已輸出 taskRecordLog.xlsx")
