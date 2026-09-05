"""微信小程序登录：jscode2session 换 openid，upsert user，发内存 token。

注意：token 是进程内存 dict 里的随机串，重启即失效、多实例不共享——
仅用于本地开发，生产环境请换成 JWT（或 Redis 共享会话）。
"""
from __future__ import annotations

import secrets

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import get_db
from app.models.user import User
from app.schemas import WxMiniLoginIn

router = APIRouter(tags=["auth"])

# token -> user_id，进程内存，开发期够用
_TOKENS: dict[str, int] = {}

_JSCODE2SESSION = "https://api.weixin.qq.com/sns/jscode2session"


@router.post("/auth/wx-mini")
def wx_mini_login(body: WxMiniLoginIn, db: Session = Depends(get_db)):
    if not settings.WX_MINI_APPID or not settings.WX_MINI_SECRET:
        raise HTTPException(
            status_code=503,
            detail="微信小程序登录未配置：请在环境变量或 backend/.env 中设置 "
            "WX_MINI_APPID 与 WX_MINI_SECRET",
        )

    try:
        resp = httpx.get(
            _JSCODE2SESSION,
            params={
                "appid": settings.WX_MINI_APPID,
                "secret": settings.WX_MINI_SECRET,
                "js_code": body.code,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        payload = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"微信登录服务请求失败: {exc}")

    openid = payload.get("openid")
    if not openid:
        errmsg = payload.get("errmsg") or "未知错误"
        raise HTTPException(
            status_code=502, detail=f"微信登录失败(errcode={payload.get('errcode')}): {errmsg}"
        )

    user = db.query(User).filter(User.openid == openid).first()
    if user is None:
        user = User(
            source="wx_mini",
            openid=openid,
            nickname=body.nickname or "",
            avatar=body.avatar,
        )
        db.add(user)
    else:
        if body.nickname:
            user.nickname = body.nickname
        if body.avatar:
            user.avatar = body.avatar
    db.commit()
    db.refresh(user)

    token = secrets.token_urlsafe(32)
    _TOKENS[token] = user.id
    return {
        "token": token,
        "user": {
            "id": user.id,
            "source": user.source,
            "nickname": user.nickname,
            "avatar": user.avatar,
        },
    }
