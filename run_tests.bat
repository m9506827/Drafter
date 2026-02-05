@echo off
REM run_tests.bat - 簡易測試執行器
REM 雙擊執行或從命令列執行

echo ================================
echo   AutoDrafter 測試執行器
echo ================================
echo.

REM 啟動虛擬環境
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [Test] 執行測試中...
echo.

python -m unittest discover test -v

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================
    echo   結果: 通過
    echo ================================
) else (
    echo.
    echo ================================
    echo   結果: 失敗
    echo ================================
)

echo.
pause
