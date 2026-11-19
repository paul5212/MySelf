# 查序號：只呼叫 GetUSNInfo（唯讀，不會改 MES 資料）
$url = "http://wimes-f60k.wih-sat-prd-c1.k8s.wistron.com/SoapCore/WebService.asmx"
$ns  = "http://localhost/Tester.WebService/WebService"

while ($true) {
    Write-Host ""
    $usn = Read-Host "輸入機器序號 USN（直接按 Enter 結束）"
    if ([string]::IsNullOrWhiteSpace($usn)) { break }
    $usnXml = [System.Security.SecurityElement]::Escape($usn.Trim())

    $body = @"
<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetUSNInfo xmlns="$ns">
      <UnitSerialNumber>$usnXml</UnitSerialNumber>
    </GetUSNInfo>
  </soap:Body>
</soap:Envelope>
"@

    try {
        $r = Invoke-WebRequest $url -Method Post -UseBasicParsing -TimeoutSec 30 `
             -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
             -ContentType "text/xml; charset=utf-8" `
             -Headers @{ SOAPAction = "`"$ns/GetUSNInfo`"" }
        $raw = $r.Content
    } catch {
        Write-Host "連線或伺服器錯誤：$($_.Exception.Message)" -ForegroundColor Red
        if ($_.Exception.Response) {
            $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
            Write-Host $sr.ReadToEnd()
        }
        continue
    }

    try {
        [xml]$x = $raw
        $res = $x.Envelope.Body.GetUSNInfoResponse.GetUSNInfoResult
    } catch { $res = $null }

    if ($res) {
        Write-Host "`n=== 查詢結果 ===" -ForegroundColor Cyan
        "Result      : {0}" -f $res.Result
        "USN         : {0}" -f $res.USN
        "ProductCode : {0}" -f $res.ProductCode
        "Model       : {0}" -f $res.Model
    }
    Write-Host "`n--- 伺服器原始回應 ---" -ForegroundColor DarkGray
    Write-Host $raw
}
