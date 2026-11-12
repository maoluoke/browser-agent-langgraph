"""Step 6：MCP 客户端接入（LLM 通过 MCP 协议调用三个工具服务）

架构：
  06_mcp_client.py（本文件，MCP Host/Client）
    ├─ stdio ├─ mcp_servers/browser_server.py    （浏览器操作）
    ├─ stdio ├─ mcp_servers/filesystem_server.py （文件读写，限根目录）
    └─ stdio └─ mcp_servers/search_server.py     （必应搜索）

  与阶段2直接函数调用的区别：工具发现（list_tools）与调用（call_tool）
  全部走 MCP 协议，服务端是独立子进程，符合 OpenManus 的工具解耦设计。
  LLM 侧只需把 MCP tool schema 转成 OpenAI function calling 格式即可复用 ReAct 循环。

验证（直接运行本文件）：
  /home/zkx456/other/bin/python3.13 06_mcp_client.py
  -> 任务：搜索 OpenManus，把第一条结果标题写入 bing_result.txt
  -> 预期：LLM 依次调用 web_search、write_file，最终文件落盘
"""

import asyncio
import importlib.util
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

_spec = importlib.util.spec_from_file_location(
    "react_agent", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_react_agent.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

PY = sys.executable
SERVERS = ["browser_server.py", "filesystem_server.py", "search_server.py"]
MAX_STEPS = 10

SYSTEM_PROMPT = """你是一个智能体，通过 MCP 工具完成任务。
可用工具来自三个服务：browser（浏览器操作）、filesystem（项目目录内文件读写）、search（必应搜索）。
规则：先搜索拿到信息再写文件；完成后直接给出最终回答，不要调用多余工具。"""


class MCPHost:
    """托管多个 MCP Server 子进程，统一暴露工具发现与调用"""

    def __init__(self):
        self._stack = None
        self._route: dict[str, ClientSession] = {}  # 工具名 -> 所属 session

    async def __aenter__(self):
        from contextlib import AsyncExitStack
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_servers")
        for fname in SERVERS:
            params = StdioServerParameters(command=PY, args=[os.path.join(base, fname)])
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools = (await session.list_tools()).tools
            for t in tools:
                self._route[t.name] = session
            print(f"[MCP] {fname} 上线，工具: {[t.name for t in tools]}")
        return self

    async def __aexit__(self, *exc):
        await self._stack.__aexit__(*exc)

    async def call(self, name: str, args: dict) -> str:
        result = await self._route[name].call_tool(name, args)
        return "".join(c.text for c in result.content if hasattr(c, "text"))


async def build_openai_tools(host: MCPHost) -> list[dict]:
    """遍历所有 session 的工具定义，逐个转换（需要在连接后调用）"""
    tools = []
    for session in dict.fromkeys(host._route.values()):
        for t in (await session.list_tools()).tools:
            tools.append({"type": "function", "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            }})
    return tools


async def run_task(task: str) -> str:
    llm = OpenAI(api_key=_mod._load_env_from_bashrc("DASHSCOPE_API_KEY"),
                 base_url=_mod.DASHSCOPE_BASE_URL, timeout=120)
    async with MCPHost() as host:
        tools = await build_openai_tools(host)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"任务：{task}"}]
        for step in range(1, MAX_STEPS + 1):
            resp = llm.chat.completions.create(
                model=_mod.AGENT_MODEL, messages=messages, tools=tools, tool_choice="auto")
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                return msg.content or ""
            for tc in msg.tool_calls:
                name, args = tc.function.name, json.loads(tc.function.arguments)
                print(f"[step {step}] {name}({json.dumps(args, ensure_ascii=False)[:80]})")
                result = await host.call(name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:3000]})
        return "（达到步数上限）"


if __name__ == "__main__":
    answer = asyncio.run(run_task(
        "用必应搜索 OpenManus，把第一条结果的标题和链接写入项目目录下的 bing_result.txt"))
    print(f"\n[最终回答] {answer}")
    # 验证文件确实落盘
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bing_result.txt")
    assert os.path.exists(path), "bing_result.txt 未生成"
    print(f"[验证] bing_result.txt 内容:\n{open(path, encoding='utf-8').read()}")
    print("[完成] 阶段6验证通过")
