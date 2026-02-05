# run_tests.ps1 - 本地測試腳本
# 使用方式: .\run_tests.ps1

param(
    [switch]$Quick,      # 只執行快速測試
    [switch]$Verbose,    # 詳細輸出
    [switch]$Watch       # 監控模式
)

$ErrorActionPreference = "Continue"

# 啟動虛擬環境
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & .\.venv\Scripts\Activate.ps1
}

Write-Host "================================" -ForegroundColor Cyan
Write-Host "  AutoDrafter 測試執行器" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

if ($Watch) {
    Write-Host "[Watch] 監控模式 - 檔案變更時自動執行測試" -ForegroundColor Yellow
    Write-Host "[Watch] 按 Ctrl+C 停止" -ForegroundColor Yellow
    Write-Host ""
    
    $lastHash = ""
    while ($true) {
        $currentHash = (Get-FileHash -Algorithm MD5 -Path "auto_drafter_system.py").Hash
        
        if ($currentHash -ne $lastHash) {
            if ($lastHash -ne "") {
                Write-Host ""
                Write-Host "[Watch] 偵測到變更，執行測試..." -ForegroundColor Yellow
            }
            
            python -m unittest test.test_components -v
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "[PASS] 所有測試通過 ✓" -ForegroundColor Green
            } else {
                Write-Host ""
                Write-Host "[FAIL] 測試失敗 ✗" -ForegroundColor Red
            }
            
            $lastHash = $currentHash
            Write-Host ""
            Write-Host "[Watch] 等待檔案變更..." -ForegroundColor Gray
        }
        
        Start-Sleep -Seconds 2
    }
} else {
    $startTime = Get-Date
    
    if ($Quick) {
        Write-Host "[Quick] 執行快速測試..." -ForegroundColor Yellow
        if ($Verbose) {
            python -m unittest test.test_components.TestSubAssemblyDrawings.test_01_load_model test.test_components.TestSubAssemblyDrawings.test_02_generates_3_dxf -v
        } else {
            python -m unittest test.test_components.TestSubAssemblyDrawings.test_01_load_model test.test_components.TestSubAssemblyDrawings.test_02_generates_3_dxf
        }
    } else {
        Write-Host "[Full] 執行完整測試..." -ForegroundColor Yellow
        if ($Verbose) {
            python -m unittest discover test -v
        } else {
            python -m unittest discover test
        }
    }
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  結果: 通過 ✓" -ForegroundColor Green
    } else {
        Write-Host "  結果: 失敗 ✗" -ForegroundColor Red
    }
    Write-Host "  耗時: $([math]::Round($duration, 2)) 秒" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
}

exit $LASTEXITCODE
