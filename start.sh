#!/usr/bin/env bash
# 星尘知识库 一键启动（开发模式）：后端 FastAPI:8000 + 前端 Nuxt:3000
# 用法: bash start.sh    （Ctrl+C 停止全部服务）
set -euo pipefail
cd "$(dirname "$0")"

info() { echo -e "\033[36m$*\033[0m"; }
warn() { echo -e "\033[33m$*\033[0m"; }
fail() { echo -e "\033[31m$*\033[0m" >&2; exit 1; }

echo "=================================================="
echo "  星尘知识库 | 一键启动（后端 8000 + 前端 3000）"
echo "=================================================="
echo

# ---- 0. 端口占用提醒 ----
for port in 8000 3000; do
  if ss -tln 2>/dev/null | grep -q ":${port} "; then
    warn "[提醒] 端口 ${port} 已被占用，对应服务可能已在运行"
  fi
done

# ---- 1. MySQL 连通性（仅提醒，不阻断）----
if command -v mysql >/dev/null 2>&1; then
  if mysql -uroot -proot -e "SELECT 1" >/dev/null 2>&1; then
    mysql -uroot -proot -e "CREATE DATABASE IF NOT EXISTS llm_kb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci" >/dev/null 2>&1
    info "[OK] MySQL 连接正常，数据库 llm_kb 已就绪"
  else
    warn "[提醒] 无法连接 MySQL（127.0.0.1:3306 root/root），评论/点赞/浏览量将不可用"
    warn "       请确认 MySQL 已启动，或检查环境变量 DATABASE_URL"
  fi
else
  warn "[提示] 未找到 mysql 客户端，跳过数据库连通性检查"
fi

# ---- 2. 后端 Python 环境（缺则自动初始化）----
PY=.cache/venv/bin/python
if [ ! -x "$PY" ]; then
  info "[初始化] 未找到虚拟环境，创建 .cache/venv 并安装后端依赖..."
  command -v python3 >/dev/null 2>&1 || fail "[错误] 未找到 python3，请先安装 Python 3.11+"
  python3 -m venv .cache/venv
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -r backend/requirements.txt
fi
info "[OK] 后端依赖就绪（.cache/venv）"

# ---- 3. 前端依赖（缺则自动安装）----
if [ ! -d frontend/node_modules ]; then
  command -v npm >/dev/null 2>&1 || fail "[错误] 未找到 npm，请先安装 Node.js 18+"
  info "[初始化] 安装前端依赖（首次较慢，请耐心等待）..."
  (cd frontend && npm install)
fi
info "[OK] 前端依赖就绪"

# ---- 4. 启动服务（后台运行，日志写入 .cache/logs/）----
LOG_DIR="$(pwd)/.cache/logs"
mkdir -p "$LOG_DIR"

export PYTHONUTF8=1
info "[启动] 后端 FastAPI  ->  http://127.0.0.1:8000  （日志 $LOG_DIR/backend.log，代码/笔记变更自动重载）"
"$PY" -m uvicorn --app-dir backend app.main:app --host 127.0.0.1 --port 8000 \
  --reload --reload-dir backend/app --reload-dir content >"$LOG_DIR/backend.log" 2>&1 &
BACK_PID=$!

info "[启动] 前端 Nuxt     ->  http://localhost:3000  （日志 $LOG_DIR/frontend.log）"
(cd frontend && npm run dev -- --port 3000 >"$LOG_DIR/frontend.log" 2>&1) &
FRONT_PID=$!

cleanup() {
  echo
  info "正在停止服务..."
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
  pkill -P "$BACK_PID" 2>/dev/null || true
  pkill -P "$FRONT_PID" 2>/dev/null || true
  wait "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
  info "已全部停止"
}
trap cleanup INT TERM

# ---- 5. 等前端就绪后尝试打开浏览器（无图形环境自动跳过）----
for _ in $(seq 1 30); do
  if curl -fsS -o /dev/null http://localhost:3000/ 2>/dev/null; then
    info "[完成] 服务已启动: http://localhost:3000/"
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open http://localhost:3000/ >/dev/null 2>&1 || true
    fi
    break
  fi
  sleep 2
done

info "按 Ctrl+C 停止全部服务"
wait "$BACK_PID" "$FRONT_PID"
