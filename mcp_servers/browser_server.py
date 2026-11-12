"""MCP Server：浏览器工具（封装 01_browser_tools.py 的 BrowserTool）

通过 stdio 传输对外提供 open_url / click / type_text / scroll / read_page / page_elements
六个工具。浏览器实例为进程级单例，MCP 客户端（06_mcp_client.py）以子进程方式拉起本服务。

独立启动（一般不需要，由客户端托管）：
  /home/zkx456/other/bin/python3.13 mcp_servers/browser_server.py
"""

import importlib.util
import json
import os

from mcp.server.fastmcp import FastMCP

# 动态导入上级目录的 01_browser_tools.py（数字开头文件名不能直接 import）
_spec = importlib.util.spec_from_file_location(
    "browser_tools",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_browser_tools.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BrowserTool = _mod.BrowserTool

mcp = FastMCP("browser")

_bt: BrowserTool | None = None


def _get() -> BrowserTool:
    """惰性启动浏览器单例"""
    global _bt
    if _bt is None:
        _bt = BrowserTool().start()
    return _bt


@mcp.tool()
def open_url(url: str) -> str:
    """打开指定 URL 的网页，返回页面标题"""
    return _get().open(url)


@mcp.tool()
def page_elements() -> str:
    """提取当前页面可交互元素列表（eid/标签/文本/坐标），JSON 字符串"""
    return json.dumps(_get().elements(), ensure_ascii=False)


@mcp.tool()
def click(eid: str) -> str:
    """点击编号为 eid 的元素，返回 ok=true/false"""
    return f"ok={_get().click(eid)}"


@mcp.tool()
def type_text(eid: str, text: str, submit: bool = False) -> str:
    """在元素 eid 中填写文本，submit=true 时按回车提交"""
    return f"ok={_get().type_text(eid, text, submit)}"


@mcp.tool()
def scroll(direction: str = "down") -> str:
    """滚动页面，direction 为 up 或 down"""
    _get().scroll(direction)
    return "ok"


@mcp.tool()
def read_page(max_len: int = 2000) -> str:
    """阅读当前页面正文文本"""
    return _get().page_text(max_len)


if __name__ == "__main__":
    mcp.run()  # stdio 传输
