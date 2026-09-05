/** 请求封装：统一拼 BASE_URL、携带 token、错误 toast。 */
import { BASE_URL } from "./config";

export interface RequestOptions {
  url: string; // 以 / 开头的 API 路径，如 /articles/latest
  method?: "GET" | "POST" | "DELETE";
  data?: Record<string, any>;
  /** 出错时静默（不 toast），由调用方自行处理，如登录降级 */
  silent?: boolean;
}

export interface ApiError {
  statusCode: number;
  message: string;
}

function getToken(): string {
  return uni.getStorageSync("token") || "";
}

export function request<T = any>(options: RequestOptions): Promise<T> {
  const token = getToken();
  const header: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    header["Authorization"] = `Bearer ${token}`;
  }
  return new Promise((resolve, reject) => {
    uni.request({
      url: BASE_URL + options.url,
      method: options.method || "GET",
      data: options.data,
      header,
      timeout: 30000,
      success: (res) => {
        const status = res.statusCode;
        if (status >= 200 && status < 300) {
          resolve(res.data as T);
          return;
        }
        // FastAPI 的错误体为 {detail: "..."}（校验错误时 detail 是数组）
        const data = res.data as any;
        let message = `请求失败(${status})`;
        if (data && data.detail) {
          message =
            typeof data.detail === "string" ? data.detail : "参数校验失败";
        }
        if (!options.silent) {
          uni.showToast({ title: message, icon: "none", duration: 2500 });
        }
        reject({ statusCode: status, message } as ApiError);
      },
      fail: (err) => {
        const message = "网络异常，请确认后端已启动";
        if (!options.silent) {
          uni.showToast({ title: message, icon: "none", duration: 2500 });
        }
        reject({ statusCode: 0, message: err.errMsg || message } as ApiError);
      },
    });
  });
}
