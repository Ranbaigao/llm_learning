/**
 * 身份与登录：
 * - visitor_id：本地生成的 UUID，用于浏览量/点赞/评论的访客标识，持久化存储。
 * - 微信登录：uni.login 拿 code → POST /auth/wx-mini → 存 token 与 user。
 *   后端未配置 WX_MINI_APPID/SECRET 时返回 503，此时降级为「手动填昵称」模式，
 *   与 Web 端匿名评论一致，不阻断任何功能。
 */
import { request } from "./request";
import type { ApiError } from "./request";

const KEY_VISITOR = "visitor_id";
const KEY_TOKEN = "token";
const KEY_USER = "user";
/** 登录降级标记：后端 503（未配置小程序 AppID）时置 "1" */
const KEY_AUTH_FALLBACK = "auth_fallback";

export interface UserInfo {
  id: number;
  source: string;
  nickname: string;
  avatar: string | null;
}

function uuid(): string {
  // 小程序基础库无可靠 crypto.getRandomValues，用随机数拼 UUID v4 足够做访客标识
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function getVisitorId(): string {
  let id: string = uni.getStorageSync(KEY_VISITOR);
  if (!id) {
    id = uuid();
    uni.setStorageSync(KEY_VISITOR, id);
  }
  return id;
}

export function getToken(): string {
  return uni.getStorageSync(KEY_TOKEN) || "";
}

export function getUser(): UserInfo | null {
  return uni.getStorageSync(KEY_USER) || null;
}

export function isAuthFallback(): boolean {
  return uni.getStorageSync(KEY_AUTH_FALLBACK) === "1";
}

/** 本地保存昵称/头像（头像为 chooseAvatar 返回的临时路径，仅本地展示用） */
export function saveProfile(nickname: string, avatar?: string) {
  const user: UserInfo = Object.assign(
    { id: 0, source: "local", nickname: "", avatar: null },
    getUser() || {},
    { nickname }
  );
  if (avatar) user.avatar = avatar;
  uni.setStorageSync(KEY_USER, user);
}

let loginPromise: Promise<boolean> | null = null;

/**
 * 确保已尝试微信登录。成功返回 true；503（未配置 AppID）或失败返回 false，
 * 并标记降级模式。整个流程静默，不打断用户操作。
 */
export function ensureLogin(): Promise<boolean> {
  if (getToken()) return Promise.resolve(true);
  if (loginPromise) return loginPromise;

  loginPromise = new Promise<boolean>((resolve) => {
    uni.login({
      provider: "weixin",
      success: async (loginRes) => {
        if (!loginRes.code) {
          resolve(false);
          return;
        }
        try {
          const user = getUser();
          const res = await request<{ token: string; user: UserInfo }>({
            url: "/auth/wx-mini",
            method: "POST",
            data: {
              code: loginRes.code,
              nickname: user?.nickname || undefined,
            },
            silent: true,
          });
          uni.setStorageSync(KEY_TOKEN, res.token);
          uni.setStorageSync(KEY_USER, res.user);
          uni.removeStorageSync(KEY_AUTH_FALLBACK);
          resolve(true);
        } catch (e) {
          const err = e as ApiError;
          if (err.statusCode === 503) {
            // 后端未配置小程序登录：降级为手动填昵称模式
            uni.setStorageSync(KEY_AUTH_FALLBACK, "1");
          }
          resolve(false);
        }
      },
      fail: () => resolve(false),
    });
  }).finally(() => {
    loginPromise = null;
  });

  return loginPromise;
}

/** 已登录状态下更新昵称（同步到后端，失败静默） */
export async function syncProfileToServer(nickname: string) {
  if (!getToken()) return;
  uni.login({
    provider: "weixin",
    success: async (loginRes) => {
      if (!loginRes.code) return;
      try {
        const res = await request<{ token: string; user: UserInfo }>({
          url: "/auth/wx-mini",
          method: "POST",
          data: { code: loginRes.code, nickname },
          silent: true,
        });
        uni.setStorageSync(KEY_TOKEN, res.token);
        uni.setStorageSync(KEY_USER, res.user);
      } catch (e) {
        // 静默
      }
    },
  });
}
