@echo off
setlocal EnableDelayedExpansion
title 星尘知识库 · 一键启动
cd /d "%~dp0"

echo ==================================================
echo   星尘知识库 ^| 一键启动（后端 8000 + 前端 3000）
echo ==================================================
echo.

REM ---- 0. 端口占用提醒 ----
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul && echo [提醒] 端口 8000 已被占用，后端可能已在运行
netstat -ano | findstr ":3000" | findstr "LISTENING" >nul && echo [提醒] 端口 3000 已被占用，前端可能已在运行

REM ---- 1. MySQL 连通性（仅提醒，不阻断）----
mysql -uroot -proot -e "SELECT 1" >nul 2>&1
if errorlevel 1 (
  echo [提醒] 无法连接 MySQL（127.0.0.1:3306 root/root^)，评论/点赞/浏览量将不可用
  echo         请确认 MySQL 服务已启动，或检查环境变量 DATABASE_URL
) else (
  mysql -uroot -proot -e "CREATE DATABASE IF NOT EXISTS llm_kb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci" >nul 2>&1
  echo [OK] MySQL 连接正常，数据库 llm_kb 已就绪
)

REM ---- 2. 后端 Python 环境（缺则自动初始化）----
if not exist ".cache\venv\Scripts\python.exe" (
  echo [初始化] 未找到虚拟环境，创建 .cache\venv 并安装后端依赖...
  python -m venv .cache\venv
  if errorlevel 1 ( echo [错误] 未找到 python，请先安装 Python 3.11+ 并加入 PATH & pause & exit /b 1 )
  .cache\venv\Scripts\python.exe -m pip install --upgrade pip >nul
  .cache\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
  if errorlevel 1 ( echo [错误] 后端依赖安装失败 & pause & exit /b 1 )
)
echo [OK] 后端依赖就绪（.cache\venv）

REM ---- 3. 前端依赖（缺则自动安装）----
if not exist "frontend\node_modules" (
  echo [初始化] 安装前端依赖（首次较慢，请耐心等待）...
  pushd frontend
  call npm install --registry=https://registry.npmmirror.com
  if errorlevel 1 call npm install
  if errorlevel 1 ( echo [错误] 前端依赖安装失败，可手动执行: cd frontend ^&^& npx npm@12.0.2 install & popd & pause & exit /b 1 )
  popd
)
echo [OK] 前端依赖就绪

REM ---- 4. 启动服务（各自独立窗口，关窗即停止；cmd /c 进程结束后窗口自动关闭）----
echo.
echo [启动] 后端 FastAPI  -^>  http://127.0.0.1:8000 （代码/笔记变更自动重载）
start "星尘-后端:8000" cmd /c "set PYTHONUTF8=1&& .cache\venv\Scripts\python.exe -m uvicorn --app-dir backend app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir backend/app --reload-dir content"

echo [启动] 前端 Nuxt     -^>  http://localhost:3000
start "星尘-前端:3000" cmd /c "cd /d %~dp0frontend && npm run dev -- --port 3000"

REM ---- 5. 等前端就绪后打开浏览器（前端编译需要时间，最多等约 60 秒）----
echo.
echo 等待前端就绪...
set /a RETRY=0
:wait_frontend
%SystemRoot%\System32\timeout.exe /t 2 /nobreak >nul
curl -s http://localhost:3000/ >nul 2>&1
if not errorlevel 1 goto frontend_ready
set /a RETRY+=1
if !RETRY! lss 30 goto wait_frontend
echo [提醒] 前端暂未就绪，请稍后手动访问 http://localhost:3000/
goto done

:frontend_ready
echo [完成] 服务已启动，正在打开浏览器...
start "" "http://localhost:3000/"

:done
echo.
echo 提示：本窗口可直接关闭。停止服务 = 关闭「星尘-后端」「星尘-前端」两个窗口。
echo 注意：前端请访问 http://localhost:3000/ （不要用 127.0.0.1:3000）
endlocal
