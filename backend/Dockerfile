# 后端镜像：FastAPI + uvicorn。content/ 不打进镜像，由 compose 只读挂载（./content:/app/content:ro）
FROM python:3.11-slim

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
