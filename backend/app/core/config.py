"""全局配置：环境变量优先，全部有本地开发默认值。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ 目录（config.py 位于 backend/app/core/）
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    DATABASE_URL: str = (
        "mysql+pymysql://root:root@127.0.0.1:3306/llm_kb?charset=utf8mb4"
    )
    ADMIN_TOKEN: str = "dev-admin-token"
    WX_MINI_APPID: str = ""
    WX_MINI_SECRET: str = ""
    REDIS_URL: str = ""
    # content/ 目录，默认相对 backend/ 的上上级（即项目根下的 content/）
    CONTENT_DIR: str = str(PROJECT_ROOT / "content")
    # 评论 IP 脱敏盐值，生产环境务必覆盖
    IP_SALT: str = "dev-ip-salt"

    @property
    def content_dir(self) -> Path:
        return Path(self.CONTENT_DIR).resolve()

    @property
    def assets_dir(self) -> Path:
        return self.content_dir / ".assets"

    @property
    def nb_cache_dir(self) -> Path:
        return BACKEND_DIR / ".cache" / "nb_html"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
