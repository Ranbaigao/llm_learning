/**
 * 全局配置。
 *
 * 开发环境：微信开发者工具里勾选「不校验合法域名」后可直接访问本机后端。
 * 真机调试：127.0.0.1 指向手机本身，必须改成电脑的局域网 IP（如 http://192.168.1.100:8000/api），
 *           且手机与电脑处于同一网段、后端监听 0.0.0.0。
 * 生产环境：必须改为已备案的 HTTPS 域名（小程序要求 request 合法域名为 https）。
 */
export const BASE_URL = "http://127.0.0.1:8000/api";

/** 服务器源（去掉 /api 后缀），用于拼接 html 中的 /api/assets/... 相对资源路径 */
export const SERVER_ORIGIN = BASE_URL.replace(/\/api\/?$/, "");
