# 星尘知识库 · 微信小程序端

个人知识库的微信小程序端，基于 **uni-app（Vue3 + TypeScript）**，编译目标 `mp-weixin`。
与 Web 端共用同一套 FastAPI 后端（`backend/`，默认 `http://127.0.0.1:8000/api`）。

## 页面结构

```
src/
├── pages/
│   ├── index/index.vue      # 首页：最新更新 + 浏览热度 双榜，顶部分类入口（tree 一级目录）+ 博客入口
│   ├── category/index.vue   # 目录浏览：/articles/tree 树形展开，点笔记进详情
│   ├── article/index.vue    # 文章详情：rich-text 渲染 html + 浏览上报 + 点赞（乐观更新）+ 评论区
│   └── blog/index.vue       # 博客列表：/articles?category=blog&n=50
├── components/
│   └── CommentList.vue      # 评论树（楼中楼一层缩进）+ 发表框（含微信头像昵称填写能力）
├── common/
│   ├── config.ts            # BASE_URL / SERVER_ORIGIN
│   ├── request.ts           # uni.request 封装：token 头、错误 toast、FastAPI detail 解析
│   └── auth.ts              # visitor_id（本地 UUID）+ wx.login 登录流程 + 昵称头像存储
├── App.vue                  # onLaunch：生成 visitor_id + 静默尝试微信登录
├── manifest.json            # mp-weixin appid 为占位 YOUR_APPID
└── pages.json               # 深色主题全局样式（无 tabBar，首页内导航）
```

主题与 Web 端星空暗色风一致：背景 `#0b1220`，文字 `#e2e8f0`，强调 `#38bdf8`。

## 开发

```bash
# 安装依赖（本机 npm 10.9.2 有 edgesOut 崩溃 bug，用 npm@12）
npx npm@12.0.2 install --registry=https://registry.npmmirror.com

# 开发编译（watch，产物在 dist/dev/mp-weixin）
npm run dev:mp-weixin

# 构建（产物在 dist/build/mp-weixin）
npm run build:mp-weixin
```

## 微信开发者工具导入步骤

1. **启动后端**：先启动 `backend/`（FastAPI，默认 `http://127.0.0.1:8000`），否则小程序所有接口都会失败。
2. **注册/准备 AppID**（可跳过）：
   - 到 <https://mp.weixin.qq.com> 注册「个人小程序」，在 开发管理 → 开发设置 拿到 AppID；
   - 替换 `src/manifest.json` 中 `mp-weixin.appid` 的 `YOUR_APPID`，重新编译；
   - 仅本地体验也可以不注册：导入时选择「测试号」。
   - **注意**：未在后端配置 `WX_MINI_APPID` / `WX_MINI_SECRET` 时，`POST /auth/wx-mini` 返回 503，小程序会自动降级为「手动填昵称」模式（与 Web 端匿名评论一致），不影响浏览、点赞、评论。
3. **导入项目**：
   - 微信开发者工具 → 导入项目 → 目录选择编译产物：
     - 开发：`miniprogram/dist/dev/mp-weixin`
     - 构建：`miniprogram/dist/build/mp-weixin`
   - 若使用 `dev:mp-weixin` 的 watch 模式，保存代码后开发者工具会自动刷新。
4. **关闭域名校验**：开发者工具右上角 详情 → 本地设置 → 勾选「不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书」。manifest 中 `urlCheck: false` 已内置，但工具内仍需确认勾选。
5. 编译预览即可。

## 真机调试

- `127.0.0.1` 在手机上指手机本身，必须把 `src/common/config.ts` 的 `BASE_URL` 改为电脑的局域网 IP（如 `http://192.168.1.100:8000/api`）；
- 手机与电脑连同一 Wi-Fi（同网段），电脑防火墙放行 8000 端口；
- 后端需监听 `0.0.0.0`（如 `uvicorn app.main:app --host 0.0.0.0 --port 8000`）；
- 图片等资源路径会由 `SERVER_ORIGIN` 自动拼接为同一来源，无需额外配置。

## 发布上线

- 小程序 request 合法域名**必须是已备案的 HTTPS 域名**，在 mp 后台「开发管理 → 服务器域名」中配置；
- `BASE_URL` 改为生产 HTTPS 地址；
- 后端需配置 `WX_MINI_APPID` / `WX_MINI_SECRET` 才能启用微信登录（否则一直走降级的手动昵称模式）。

## 已知限制

- **数学公式**：小程序 `rich-text` 无法运行 MathJax/KaTeX，公式只显示 TeX 源码（如 `$\alpha$`）。如需渲染需引入图片公式或 web-view 方案，本期未做。
- **ipynb 笔记**：Jupyter 渲染出的 HTML 较重（内联样式/脚本残留），小程序端展示属「能看但重」，加载稍慢属正常现象。
- **代码高亮**：Web 端依赖 CSS/JS 高亮，小程序端只有 rich-text 默认样式，代码块以等宽文本原样展示。
- **头像**：`chooseAvatar` 拿到的是微信临时文件路径，后端暂无头像上传接口，头像仅在本地展示，不落库（昵称会随登录同步到后端）。
- **点赞状态**：后端无「我是否已赞」查询接口，已赞状态存在本地存储（`liked_ids`），换设备/清缓存后状态会丢失，但重复点赞服务端幂等不会出错。
- **token**：后端 token 为进程内存存储，重启即失效；小程序端会在下次操作时静默重新登录，用户无感。
