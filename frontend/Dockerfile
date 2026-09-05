# 前端镜像：多阶段构建 Nuxt SSR（nitro node-server 预设），产物仅为 .output/
# SSR 服务端请求后端的地址通过运行时环境变量 NUXT_API_SERVER 注入（compose 里指向 http://backend:8000）
FROM node:22 AS build
WORKDIR /app

# npm 源：默认 npmmirror（国内网络友好）；海外可 --build-arg NPM_REGISTRY=https://registry.npmjs.org 覆盖
ARG NPM_REGISTRY=https://registry.npmmirror.com
ENV npm_config_registry=${NPM_REGISTRY}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:22
WORKDIR /app

ENV NITRO_HOST=0.0.0.0 \
    NITRO_PORT=3000 \
    NODE_ENV=production

COPY --from=build /app/.output ./.output

EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
