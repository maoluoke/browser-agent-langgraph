"""Step 2：ReAct 单智能体闭环（LLM 决策 + 浏览器工具）

架构：
  LLM（deepseek-v4-flash，原生 tool_calls）在循环中决策：
  观察（页面元素列表）-> 思考 -> 调用工具（click/type/scroll/...）-> 观察新状态
  直到调用 finish(answer) 或达到步数上限。

设计要点：
  - 每次动作成功后，工具返回值里直接附带最新元素列表（eid 会随 DOM 变化刷新）
  - 元素列表压缩为单行文本，控制 token 消耗
  - click 返回 ok=False 时提示 LLM 换元素或方式（阶段3将在此接入视觉兜底）

验证（直接运行本文件）：
  /home/zkx456/other/bin/python3.13 02_react_agent.py
"""

import importlib.util
import json
import os
import sys

from openai import OpenAI

# 动态导入 01_browser_tools.py（文件名以数字开头，不能直接 import）
_spec = importlib.util.spec_from_file_location(
    "browser_tools", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_browser_tools.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BrowserTool = _mod.BrowserTool

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.8-2.4t-a95b")  # 按 token 计费单价低；flash 免费档已耗尽
MAX_STEPS = 12


def _load_env_from_bashrc(name: str) -> str:
    """从 ~/.bashrc 兜底读取环境变量（IDE/调试器启动时不会加载 bashrc）。"""
    value = os.getenv(name, "")
    if value:
        return value
    bashrc = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"export {name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


# ---------- 工具定义（OpenAI function calling schema） ----------
TOOLS = [
    {"type": "function", "function": {
        "name": "open_url", "description": "打开指定 URL 的网页",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "完整网址，含 https://"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "click", "description": "点击编号为 eid 的元素。建议同时给出 desc 描述该元素",
        "parameters": {"type": "object", "properties": {
            "eid": {"type": "string", "description": "元素编号，如 e3"},
            "desc": {"type": "string", "description":
                     "该元素的自然语言描述（如「搜索输入框」「登录按钮」），"
                     "DOM 点击失败时供视觉定位兜底使用"}},
            "required": ["eid"]}}},
    {"type": "function", "function": {
        "name": "type_text", "description": "在输入框元素中填写文本",
        "parameters": {"type": "object", "properties": {
            "eid": {"type": "string"}, "text": {"type": "string"},
            "submit": {"type": "boolean", "description": "填完后是否按回车提交"}},
            "required": ["eid", "text"]}}},
    {"type": "function", "function": {
        "name": "click_text", "description": "按页面上的可见文字点击，适合日期、城市、车次等没有稳定编号的元素",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "要点击的文字，如「11月5日」「北京」"},
            "exact": {"type": "boolean", "description": "是否要求文字完全一致，默认 false"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "click_nth", "description":
            "点击第 index 个（从 0 开始）文字匹配的元素。结果页有多个同名按钮（多个「预订」）时按顺序选",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "要点击的文字，如「预订」"},
            "index": {"type": "integer", "description": "第几个匹配项，0 表示第一个"},
            "exact": {"type": "boolean", "description": "是否要求文字完全一致，默认 false"}},
            "required": ["text", "index"]}}},
    {"type": "function", "function": {
        "name": "select_suggestion", "description":
            "从输入框弹出的自动补全下拉中选中文字完全匹配的条目（城市/车站/酒店名）。"
            "比 click_text 准：携程会把「北京」高亮成「北京北」的前缀导致点错",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "候选条目的完整文字，如「北京」「上海虹桥」"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "select_option", "description": "在下拉框中选择选项",
        "parameters": {"type": "object", "properties": {
            "eid": {"type": "string", "description": "下拉框元素编号"},
            "value": {"type": "string", "description": "要选择的选项文字或值"}},
            "required": ["eid", "value"]}}},
    {"type": "function", "function": {
        "name": "check", "description": "勾选或取消勾选单选按钮/复选框（如车次类型、座位等级、保险）",
        "parameters": {"type": "object", "properties": {
            "eid": {"type": "string"},
            "checked": {"type": "boolean", "description": "true 勾选，false 取消，默认 true"}},
            "required": ["eid"]}}},
    {"type": "function", "function": {
        "name": "press_key", "description": "按下键盘按键，如 Enter 提交、Escape 关闭弹窗、Tab 切换焦点",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "按键名，如 Enter / Escape / Tab"}},
            "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "wait_for", "description": "等待页面出现指定文字或元素，用于等结果加载、页面跳转",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "等待出现的文字"},
            "timeout": {"type": "number", "description": "最长等待秒数，默认 10"}}}}},
    {"type": "function", "function": {
        "name": "switch_tab", "description": "切换到另一个标签页（支付页等常在新标签打开）",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer", "description": "标签页序号，-1 表示最后一个"}}}}},
    {"type": "function", "function": {
        "name": "pause_for_human", "description":
            "暂停并交由人工操作。登录、短信验证码、滑块验证、提交订单/支付前必须调用它",
        "parameters": {"type": "object", "properties": {
            "prompt": {"type": "string", "description": "提示用户要做什么"}}}}},
    {"type": "function", "function": {
        "name": "scroll", "description": "滚动页面以查看更多内容",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["up", "down"]}}, "required": ["direction"]}}},
    {"type": "function", "function": {
        "name": "read_page", "description": "阅读当前页面正文文本（找信息、提取内容时用）",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "任务完成，提交最终答案",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "给用户的最终回答"}}, "required": ["answer"]}}},
]

SYSTEM_PROMPT = """你是一个浏览器自动化智能体。通过调用工具操作浏览器完成用户任务。

页面上的可交互元素已编号，格式：eid <标签> 文本。
注意：
1. 每次动作后元素编号会刷新，必须使用最新列表中的编号。
2. 如果 click 返回 ok=false，说明该元素不可点，请换一个元素、改用 click_text 或先 scroll。
3. 需要阅读页面内容找信息时调用 read_page。
4. 拿到答案后立即调用 finish，不要多做无关操作。
5. 输入搜索词后记得用 submit=true 提交。
6. 调用 click 时尽量补上 desc（元素的中文描述），DOM 点击失败时会用它做视觉定位兜底。
7. 日期、城市、车次等没有稳定编号的元素，优先用 click_text 按文字点击；城市/车站下拉用 select_suggestion 选完整条目，避免点到「北京北」这类含「北京」前缀的项。
8. 结果页、支付页加载慢时，用 wait_for 等待关键词出现，不要盲目重复点击。
9. 遇到登录、短信验证码、滑块验证：立即调用 pause_for_human 让人工完成，不要自己尝试绕过。
10. 严禁自行提交订单或进行支付：到确认页面前必须调用 pause_for_human 交还给用户。
11. 结果页有多个同名按钮（如多个「预订」「抢票」）时，用 click_nth 按 index 选择；元素后面的 @数字 是 y 坐标，可用于判断第几行。"""


def _fmt_elements(els: list[dict], limit: int = 60) -> str:
    """压缩元素列表为单行文本；带 y 坐标便于按行（车次/房型）对应"""
    parts = [f"{e['eid']}<{e['tag']}>{e['text'][:30]}@{e['y']}" for e in els[:limit]]
    return " | ".join(parts) if parts else "(无可交互元素)"


def exec_tool(bt, name: str, args: dict, click_fallback=None) -> str:
    """执行单个工具，返回给 LLM 的观察文本。
    click_fallback: 可选回调 fn(args)->bool，DOM 点击失败时用于视觉兜底（阶段4 注入）。
    """
    if name == "open_url":
        title = bt.open(args["url"])
        return f"已打开，标题：{title}\n元素：{_fmt_elements(bt.elements())}"
    if name == "click":
        ok = bt.click(args["eid"])
        if not ok and click_fallback:
            ok = click_fallback(args)
        if not ok:
            return (f"click {args['eid']} 失败 ok=false（元素不可点或已失效，"
                    f"可改用 click_text）\n元素：{_fmt_elements(bt.elements())}")
        return f"已点击 {args['eid']}，当前页：{bt.page.title()}\n元素：{_fmt_elements(bt.elements())}"
    if name == "click_text":
        ok = bt.click_text(args["text"], args.get("exact", False))
        if not ok:
            return (f"点击文字「{args['text']}」失败，页面上没找到该文字\n"
                    f"元素：{_fmt_elements(bt.elements())}")
        return f"已点击「{args['text']}」，当前页：{bt.page.title()}\n元素：{_fmt_elements(bt.elements())}"
    if name == "click_nth":
        ok = bt.click_nth(args["text"], args.get("index", 0), args.get("exact", False))
        if not ok:
            return (f"点击第 {args.get('index', 0)} 个「{args['text']}」失败\n"
                    f"元素：{_fmt_elements(bt.elements())}")
        return f"已点击第 {args.get('index', 0)} 个「{args['text']}」，当前页：{bt.page.title()}\n元素：{_fmt_elements(bt.elements())}"
    if name == "select_suggestion":
        ok = bt.select_suggestion(args["text"])
        if not ok:
            return (f"下拉中没选中「{args['text']}」，可能候选未出现或文字不完全一致\n"
                    f"元素：{_fmt_elements(bt.elements())}")
        return f"已从下拉选中「{args['text']}」，当前页：{bt.page.title()}\n元素：{_fmt_elements(bt.elements())}"
    if name == "select_option":
        ok = bt.select_option(args["eid"], args["value"])
        if not ok:
            return f"select_option {args['eid']}={args['value']} 失败\n元素：{_fmt_elements(bt.elements())}"
        return f"已选择 {args['value']}\n元素：{_fmt_elements(bt.elements())}"
    if name == "check":
        ok = bt.check(args["eid"], args.get("checked", True))
        if not ok:
            return f"check {args['eid']} 失败\n元素：{_fmt_elements(bt.elements())}"
        return f"已切换勾选状态\n元素：{_fmt_elements(bt.elements())}"
    if name == "press_key":
        bt.press_key(args["key"])
        return f"已按下 {args['key']}\n元素：{_fmt_elements(bt.elements())}"
    if name == "wait_for":
        ok = bt.wait_for(args.get("keyword"), timeout=args.get("timeout", 10))
        if ok:
            return f"等待成功，页面已出现「{args.get('keyword')}」\n元素：{_fmt_elements(bt.elements())}"
        return f"等待超时（{args.get('timeout', 10)}s），页面未出现「{args.get('keyword')}」"
    if name == "switch_tab":
        title = bt.switch_tab(args.get("index", -1))
        return f"已切换标签页，当前页：{title}\n元素：{_fmt_elements(bt.elements())}"
    if name == "pause_for_human":
        return bt.pause_for_human(args.get("prompt", "请完成人工操作后按回车继续"))
    if name == "type_text":
        ok = bt.type_text(args["eid"], args["text"], args.get("submit", False))
        if not ok:
            return f"type_text {args['eid']} 失败 ok=false\n元素：{_fmt_elements(bt.elements())}"
        return f"已输入，当前页：{bt.page.title()}\n元素：{_fmt_elements(bt.elements())}"
    if name == "scroll":
        bt.scroll(args["direction"])
        return f"已滚动\n元素：{_fmt_elements(bt.elements())}"
    if name == "read_page":
        return f"页面正文：\n{bt.page_text()}"
    if name == "finish":
        return "__FINISH__"
    return f"未知工具: {name}"


class ReactAgent:
    def __init__(self, headless: bool | None = None, slow_mo: int | None = None,
                 profile_dir: str | None = None):
        self.bt = BrowserTool(headless=headless, slow_mo=slow_mo,
                              profile_dir=profile_dir).start()
        self.llm = OpenAI(api_key=_load_env_from_bashrc("DASHSCOPE_API_KEY"),
                          base_url=DASHSCOPE_BASE_URL, timeout=120)

    def close(self):
        self.bt.close()

    def _exec_tool(self, name: str, args: dict) -> str:
        return exec_tool(self.bt, name, args)

    def run(self, task: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"任务：{task}"}]
        for step in range(1, MAX_STEPS + 1):
            resp = self.llm.chat.completions.create(
                model=AGENT_MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                print(f"[step {step}] LLM 未调用工具，输出：{msg.content}")
                continue
            for tc in msg.tool_calls:
                name, args = tc.function.name, json.loads(tc.function.arguments)
                print(f"[step {step}] {name}({json.dumps(args, ensure_ascii=False)})")
                result = self._exec_tool(name, args)
                if result == "__FINISH__":
                    return args.get("answer", "")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        return "（达到步数上限，任务未完成）"


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "打开必应 www.bing.com，搜索 OpenManus，把第一条搜索结果的标题告诉我"
    agent = ReactAgent()
    try:
        answer = agent.run(task)
        print(f"\n[最终答案] {answer}")
    finally:
        agent.close()
