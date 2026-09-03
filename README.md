# MySelf
開發目的精簡 LIST監測當天批量生產的環境穩定度（同一日期內 IP01~IP24 的 variance 一致性高 → 環境穩定）
比較不同日期的環境穩定性（跨天 variance 變化大 → 環境不穩定）
找出特定 IPXX 在不同天 variance 差異過大的情況（例如 IP03、IP05 等異常波動）
預防因環境變化導致 AOI 參數失效（variance 過低觸發模糊警報）
提供量化數據支持 AOI 參數調整與長期趨勢追蹤


type %USERPROFILE%\.ssh\id_ed25519.pub | ssh pi@10.131.74.239 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
