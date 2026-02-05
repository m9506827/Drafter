# Quota 滿時，用此腳本切換至 API mode 繼續工作
# 使用方式: .\claude-api.ps1

# ===== 請填入你的 API Key =====
$env:ANTHROPIC_API_KEY = "sk-ant-api03-你的KEY放這裡"
# ==============================

Write-Host ""
Write-Host "已切換至 API mode (使用個人 API Key)" -ForegroundColor Yellow
Write-Host "將繼續上次的對話..." -ForegroundColor Cyan
Write-Host ""

# 繼續上次對話
claude -c
