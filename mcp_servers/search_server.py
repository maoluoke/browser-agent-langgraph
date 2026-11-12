"""MCP Server：网页搜索工具（必应，轻量 HTTP 抓取）

不占用浏览器实例，直接请求必应搜索结果页并解析标题/链接/摘要，
适合「先搜再决定打开哪个页面」的场景；需要交互操作时再交给 browser server。

独立启动（由 MCP 客户端托管，一般不需手动执行）：
  /home/zkx456/other/bin/python3.13 mcp_servers/search_server.py
"""

import re

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search")

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


@mcp.tool()
def web_search(query: str, top_k: int = 5) -> str:
    """用必应搜索关键词，返回前 top_k 条结果（标题/链接/摘要），每条一段"""
    resp = httpx.get("https://www.bing.com/search",
                     params={"q": query, "setlang": "zh-CN"},
                     headers={"User-Agent": _UA}, timeout=15, follow_redirects=True)
    blocks = re.findall(r'<li class="b_algo".*?</li>', resp.text, re.S)
    results = []
    for b in blocks[:top_k]:
        m = re.search(r'<h2><a[^>]*href="([^"]+)"[^>]*>(.*?)</a></h2>', b, re.S)
        if not m:
            continue
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        snip_m = re.search(r'<p[^>]*>(.*?)</p>', b, re.S)
        snip = re.sub(r"<[^>]+>", "", snip_m.group(1)).strip()[:150] if snip_m else ""
        results.append(f"{len(results)+1}. {title}\n   {url}\n   {snip}")
    return "\n\n".join(results) if results else "未解析到搜索结果"


if __name__ == "__main__":
    mcp.run()  # stdio 传输
