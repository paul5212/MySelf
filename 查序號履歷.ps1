# 查序號履歷：只呼叫 GetUSNGenealogyBasic（唯讀，不會改 MES 資料）
$url = "http://wimes-f60k.wih-sat-prd-c1.k8s.wistron.com/SoapCore/WebService.asmx"
$ns  = "http://localhost/Tester.WebService/WebService"

function Esc($s) { [System.Security.SecurityElement]::Escape($s) }

while ($true) {
    Write-Host ""
    $usn = Read-Host "輸入機器序號 USN（直接按 Enter 結束）"
    if ([string]::IsNullOrWhiteSpace($usn)) { break }
    $stage = Read-Host "站別代碼 StageCode（不知道就直接按 Enter）"

    $body = @"
<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetUSNGenealogyBasic xmlns="$ns">
      <UnitSerialNumber>$(Esc $usn.Trim())</UnitSerialNumber>
      <StageCode>$(Esc $stage.Trim())</StageCode>
    </GetUSNGenealogyBasic>
  </soap:Body>
</soap:Envelope>
"@

    try {
        $r = Invoke-WebRequest $url -Method Post -UseBasicParsing -TimeoutSec 30 `
             -Body ([Text.Encoding]::UTF8.GetBytes($body)) `
             -ContentType "text/xml; charset=utf-8" `
             -Headers @{ SOAPAction = "`"$ns/GetUSNGenealogyBasic`"" }
        $raw = $r.Content
    } catch {
        Write-Host "連線或伺服器錯誤：$($_.Exception.Message)" -ForegroundColor Red
        if ($_.Exception.Response) {
            $sr = New-Object IO.StreamReader($_.Exception.Response.GetResponseStream())
            Write-Host $sr.ReadToEnd()
        }
        continue
    }

    # 原始回應存檔，方便對照
    $logDir = Join-Path $PSScriptRoot "回應紀錄"
    New-Item -ItemType Directory -Force $logDir | Out-Null
    $file = Join-Path $logDir ("{0}_{1}.xml" -f ($usn -replace '[\\/:*?"<>|]','_'), (Get-Date -Format "yyyyMMdd_HHmmss"))
    [IO.File]::WriteAllText($file, $raw, [Text.Encoding]::UTF8)

    [xml]$x = $raw
    $res = $x.Envelope.Body.GetUSNGenealogyBasicResponse.GetUSNGenealogyBasicResult
    if (-not $res) {
        Write-Host "沒有拿到結果，原始回應如下：" -ForegroundColor Yellow
        Write-Host $raw
        continue
    }

    Write-Host "`n=== 基本資料 ===" -ForegroundColor Cyan
    $fields = "USN","UPN","MO","CATEGORY","Line","Stage","NextStage","Status","RouteName","Tour","OldUSN","LotNo","ErpPalletID","CreateDate","UpdateDate"
    foreach ($f in $fields) { "{0,-12}: {1}" -f $f, $res.$f }

    Write-Host "`n=== 綁定零件 ===" -ForegroundColor Cyan
    $items = $res.USNItems.clsUSNItem
    if ($items) {
        $items | Select-Object Position, CATEGORY, CPN, CSN, MO | Format-Table -AutoSize
    } else {
        "（沒有零件資料）"
    }
    Write-Host "原始回應已存：$file" -ForegroundColor DarkGray
}
