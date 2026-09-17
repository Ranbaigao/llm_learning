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

REM ---- 3. 检查 Nuxt 包和命令入口，避免把安装中断留下的空目录当成依赖就绪 ----
set FRONTEND_DEPS_MISSING=0
if not exist "frontend\node_modules\.bin\nuxt.cmd" set FRONTEND_DEPS_MISSING=1
if not exist "frontend\node_modules\nuxt\package.json" set FRONTEND_DEPS_MISSING=1
if !FRONTEND_DEPS_MISSING! equ 1 (
  echo [初始化] 前端依赖缺失或不完整，按 package-lock.json 重新安装...
  pushd frontend
  call npm ci --registry=https://registry.npmmirror.com
  if errorlevel 1 call npm ci
  if errorlevel 1 ( echo [错误] 前端依赖安装失败，请查看上方报错；若文件被占用，先关闭本项目的前端进程再重试 & popd & pause & exit /b 1 )
  popd
)
echo [OK] 前端依赖就绪

REM ---- 4. 启动服务（各自独立窗口；前端失败后保留窗口，便于查看报错）----
echo.
echo [启动] 后端 FastAPI  -^>  http://127.0.0.1:8000 （代码/笔记变更自动重载）
start "星尘-后端:8000" cmd /c "set PYTHONUTF8=1&& .cache\venv\Scripts\python.exe -m uvicorn --app-dir backend app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir backend/app --reload-dir content"

echo [启动] 前端 Nuxt     -^>  http://localhost:3000
start "星尘-前端:3000" /D "%~dp0frontend" cmd /k "npm run dev -- --port 3000"

REM ---- 5. 等前端就绪后打开浏览器（每次请求限时 2 秒，最多等约 120 秒）----
echo.
echo 等待前端就绪...
set /a RETRY=0
:wait_frontend
%SystemRoot%\System32\timeout.exe /t 2 /nobreak >nul
curl --noproxy "*" --connect-timeout 1 --max-time 2 --fail --silent http://localhost:3000/ >nul 2>&1
if not errorlevel 1 goto frontend_ready
set /a RETRY+=1
if !RETRY! lss 30 goto wait_frontend
echo [错误] 前端启动超时，请查看「星尘-前端:3000」窗口中的报错。
echo        可在 frontend 目录执行 npm run dev -- --port 3000 复现问题。
pause
exit /b 1

:frontend_ready
echo [完成] 服务已启动，正在打开浏览器...
start "" "http://localhost:3000/"

:done
echo.
echo 提示：本窗口可直接关闭。停止服务 = 关闭「星尘-后端」「星尘-前端」两个窗口。
echo 注意：前端请访问 http://localhost:3000/ （不要用 127.0.0.1:3000）
endlocal
