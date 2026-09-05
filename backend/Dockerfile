# 后端镜像：FastAPI + uvicorn。content/ 不打进镜像，由 compose 只读挂载（./content:/app/content:ro）
FROM python:3.11-slim

# git：最新更新榜单按「最近内容提交时间」排序，需在容器内读挂载进来的 .git
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# pip 源：默认官方源（本机代理环境实测可用；国内无代理服务器建议
# 在 .env 设 PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple 覆盖为清华镜像）
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
