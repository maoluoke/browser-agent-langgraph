"""MCP Server：文件系统工具（限定根目录，防路径逃逸）

所有读写操作限制在项目目录（project3/）内，相对路径解析后
必须仍位于根目录之下，否则拒绝执行。

独立启动（由 MCP 客户端托管，一般不需手动执行）：
  /home/zkx456/other/bin/python3.13 mcp_servers/filesystem_server.py
"""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("filesystem")

# 文件操作根目录：project3/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _safe(path: str) -> str:
    """把相对路径解析为根目录内的绝对路径，越界则抛错"""
    full = os.path.realpath(os.path.join(ROOT, path))
    if not (full == ROOT or full.startswith(ROOT + os.sep)):
        raise ValueError(f"路径越界，已拒绝：{path}")
    return full


@mcp.tool()
def read_file(path: str) -> str:
    """读取项目目录内的文本文件，path 为相对路径"""
    with open(_safe(path), encoding="utf-8") as f:
        return f.read()[:5000]


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """把文本写入项目目录内的文件（覆盖写），path 为相对路径"""
    with open(_safe(path), "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path}（{len(content)} 字符）"


@mcp.tool()
def list_dir(path: str = ".") -> str:
    """列出项目目录内某个子目录的文件清单"""
    entries = sorted(os.listdir(_safe(path)))
    return "\n".join(entries[:100])


if __name__ == "__main__":
    mcp.run()  # stdio 传输
