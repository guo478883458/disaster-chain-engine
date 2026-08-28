@echo off
chcp 65001 >nul
title 灾害链推理演示系统 v1

echo ============================================
echo  灾害链推理演示系统 v1
echo  正在启动 Dashboard...
echo ============================================
echo.

REM 检查 Python 是否可用
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未找到 Python，请确保 Python 3.10+ 已安装并加入 PATH
    echo.
    pause
    exit /b 1
)

REM 检查依赖是否安装
python -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [提示] 首次运行请先安装依赖：
    echo   pip install -r requirements.txt
    echo   pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo.
    pause
    exit /b 1
)

echo 正在启动 Streamlit...
echo 请在浏览器中打开 http://localhost:8501
echo.
start http://localhost:8501
python -m streamlit run dashboard.py --server.maxUploadSize 100

echo.
echo 按任意键退出...
pause