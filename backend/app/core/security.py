"""IP 脱敏等小工具。"""
from __future__ import annotations

import hashlib

from app.core.config import settings


def hash_ip(ip: str | None) -> str:
    """加盐哈希客户端 IP，库里不存明文。"""
    raw = f"{settings.IP_SALT}|{ip or ''}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
