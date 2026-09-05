"""后端 API 冒烟测试（开发用）。

用法：先启动 uvicorn，再运行
    export PYTHONUTF8=1
    ../.cache/venv/Scripts/python.exe scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from urllib.parse import quote

import httpx

BASE = "http://127.0.0.1:8000/api"
ADMIN = "dev-admin-token"

# 每次运行用独立 visitor 前缀，避免 page_view 按天去重/点赞唯一约束撞上历史脏数据
RUN = f"smoke-{int(time.time())}"

passed = failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name} {detail}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=60)

    print("== 1. 目录树 ==")
    r = client.get("/articles/tree")
    tree = r.json()
    top = {c["name"] for c in tree["children"]}
    check("tree 200", r.status_code == 200)
    check("tree 顶层分类", {"CV", "NLP", "代码实践", "性能优化"} <= top, str(top))
    check("tree 排除 blog/.assets", "blog" not in top and ".assets" not in top)
    nlp = next(c for c in tree["children"] if c["name"] == "NLP")
    check("tree 嵌套子目录", len(nlp["children"]) > 0 and nlp["note_count"] > 0,
          f"NLP note_count={nlp['note_count']}")

    print("== 2. 最新/热门 ==")
    latest = client.get("/articles/latest", params={"n": 5}).json()
    check("latest 返回 5 条", len(latest) == 5)
    check("latest 有 content_updated_at", all(a["content_updated_at"] for a in latest))
    hot = client.get("/articles/hot", params={"n": 5}).json()
    check("hot 返回 5 条", len(hot) == 5)

    print("== 3. Markdown 详情（数学公式 + 图片改写）==")
    slug_math = "NLP/LLM模型架构/残差连接/mHC/双随机矩阵"
    r = client.get(f"/articles/{quote(slug_math)}")
    art = r.json()
    html = art["html"]
    check("md 详情 200", r.status_code == 200)
    check("md format=markdown", art["format"] == "markdown")
    math_hits = html.count('class="math')
    check("math span/div 存在", 'class="math' in html, f"出现 {math_hits} 处")
    r2 = client.get("/articles/" + quote("NLP/LLM模型训练/SFT/SFT不完整学习现象ILP"))
    html2 = r2.json()["html"]
    check("图片改写为 /api/assets/", "/api/assets/ilp_fig1_schematic.jpg" in html2)
    check("无相对路径 ../.assets 残留", "../.assets" not in html2)

    print("== 4. frontmatter meta（博客）==")
    r = client.get("/articles/" + quote("blog/posts/2026-08-23-hello-blog"))
    blog = r.json()
    check("blog category=blog", blog["category"] == "blog")
    check("frontmatter 已剔除", "---" not in blog["html"][:100])
    check("meta.date 返回", "date" in blog["meta"], str(blog["meta"].get("date")))
    check("meta.categories 返回", blog["meta"].get("categories") == ["随笔"])

    print("== 5. Notebook 详情 ==")
    r = client.get("/articles/" + quote("代码实践/gpt"))
    nb = r.json()
    check("ipynb 200", r.status_code == 200)
    check("ipynb format=jupyter", nb["format"] == "jupyter")
    check("ipynb 返回 HTML", "<" in nb["html"] and len(nb["html"]) > 500,
          f"len={len(nb['html'])}")

    print("== 6. 星图 ==")
    r = client.get("/graph")
    g = r.json()
    check("graph 200", r.status_code == 200)
    stats = g["stats"]
    check("graph stats notes/jupyter", stats["notes"] == 64 and stats["jupyter"] == 7,
          str(stats))
    root = next(n for n in g["nodes"] if n["type"] == "root")
    check("graph root 节点", root["id"] == "__root__")
    note = next(n for n in g["nodes"] if n["type"] == "note")
    check("note 节点有 mtime/pv/url", "mtime" in note and "pv" in note
          and note["url"].startswith("/notes/"), f"url={note['url']}")
    check("有 reference 边", stats["references"] > 0, f"references={stats['references']}")
    check("无 blog 节点", not any(n["id"].startswith("blog") for n in g["nodes"]))

    print("== 7. 浏览量去重 ==")
    aid = latest[0]["id"]
    v1 = client.post(f"/articles/{aid}/view", json={"visitor_id": f"{RUN}-v1"}).json()
    v2 = client.post(f"/articles/{aid}/view", json={"visitor_id": f"{RUN}-v1"}).json()
    check("首次 view counted=True", v1["counted"] is True)
    check("重复 view counted=False", v2["counted"] is False)
    check("views 只加一次", v2["views"] == v1["views"], f"{v1['views']} -> {v2['views']}")
    v3 = client.post(f"/articles/{aid}/view", json={"visitor_id": f"{RUN}-v2"}).json()
    check("新访客 counted=True 且 views+1",
          v3["counted"] is True and v3["views"] == v1["views"] + 1)

    print("== 8. 点赞幂等 ==")
    base_count = client.post(f"/articles/{aid}/like", json={"visitor_id": f"{RUN}-setup"}).json()["like_count"]
    client.request("DELETE", f"/articles/{aid}/like", json={"visitor_id": f"{RUN}-setup"})
    l1 = client.post(f"/articles/{aid}/like", json={"visitor_id": f"{RUN}-v1"}).json()
    l2 = client.post(f"/articles/{aid}/like", json={"visitor_id": f"{RUN}-v1"}).json()
    check("首次 like liked=True", l1["liked"] is True)
    check("重复 like 不报错且计数不变",
          l2["liked"] is True and l2["like_count"] == l1["like_count"])
    l3 = client.request("DELETE", f"/articles/{aid}/like", json={"visitor_id": f"{RUN}-v1"}).json()
    check("取消 like", l3["liked"] is False and l3["like_count"] == l1["like_count"] - 1)
    l4 = client.request("DELETE", f"/articles/{aid}/like", json={"visitor_id": f"{RUN}-v1"}).json()
    check("重复取消幂等", l4["liked"] is False and l4["like_count"] == l3["like_count"])

    print("== 9. 评论全流程 ==")
    c1 = client.post(f"/articles/{aid}/comments",
                     json={"visitor_id": f"{RUN}-v1", "nickname": "路人甲", "content": "第一条评论"})
    check("发评论 201", c1.status_code == 201, str(c1.status_code))
    cid = c1.json()["id"]
    c2 = client.post(f"/articles/{aid}/comments",
                     json={"visitor_id": f"{RUN}-v2", "nickname": "路人乙",
                           "content": "楼中楼回复", "parent_id": cid})
    check("楼中楼 201", c2.status_code == 201)
    cl = client.get(f"/articles/{aid}/comments").json()
    mine = next(c for c in cl if c["id"] == cid)
    check("评论树嵌套",
          len(mine["children"]) == 1 and mine["children"][0]["content"] == "楼中楼回复")
    bad = client.post(f"/articles/{aid}/comments",
                      json={"visitor_id": "x", "nickname": "", "content": "x"})
    check("空昵称 422", bad.status_code == 422)
    # 限频：同 IP（127.0.0.1）已发 2 条，第 3 条成功，第 4 条 429
    c3 = client.post(f"/articles/{aid}/comments",
                     json={"visitor_id": f"{RUN}-v1", "nickname": "路人甲", "content": "第三条"})
    c4 = client.post(f"/articles/{aid}/comments",
                     json={"visitor_id": f"{RUN}-v1", "nickname": "路人甲", "content": "第四条应被限频"})
    check("同 IP 第 3 条成功", c3.status_code == 201)
    check("同 IP 第 4 条 429", c4.status_code == 429, str(c4.status_code))
    d1 = client.delete(f"/comments/{cid}")
    check("无 token 删评论 403", d1.status_code == 403)
    d2 = client.delete(f"/comments/{cid}", headers={"X-Admin-Token": "wrong"})
    check("错 token 删评论 403", d2.status_code == 403)
    d3 = client.delete(f"/comments/{cid}", headers={"X-Admin-Token": ADMIN})
    check("管理员软删成功", d3.status_code == 200 and d3.json()["status"] == "deleted")
    cl2 = client.get(f"/articles/{aid}/comments").json()
    check("软删后不可见（含其子评论）", all(c["id"] != cid for c in cl2))

    print("== 10. 搜索 ==")
    r = client.get("/search", params={"q": "agent"})
    res = r.json()
    check("search agent 有结果", len(res) > 0, f"{len(res)} 条")
    check("search 字段齐全",
          all({"slug", "title", "category", "snippet"} <= set(x) for x in res))
    check("search snippet 含关键词", any("agent" in x["snippet"].lower() for x in res))
    r = client.get("/search", params={"q": "100%_不存在的词%"})
    check("LIKE 通配符转义", r.json() == [])

    print("== 11. 静态资产 ==")
    r = client.get("/assets/1780588934023.png")
    check("图片 200 + image/png",
          r.status_code == 200 and r.headers["content-type"] == "image/png",
          f"{r.status_code} {r.headers.get('content-type')} {len(r.content)}B")
    r = client.get("/assets/" + quote("../app/main.py"))
    check("路径穿越被拒绝", r.status_code in (403, 404), str(r.status_code))
    r = client.get("/assets/nonexistent.png")
    check("不存在 404", r.status_code == 404)

    print("== 12. 微信登录未配置 ==")
    r = client.post("/auth/wx-mini", json={"code": "dummy"})
    check("未配置 503 + 中文提示",
          r.status_code == 503 and "微信小程序登录未配置" in r.json()["detail"])

    print(f"\n结果: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
