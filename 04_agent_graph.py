"""Step 4：LangGraph 三层状态机（Planner / Executor / Reviewer）

相对 02 单体 ReAct 的升级：
  - Planner：把复杂任务拆解为有序子任务列表；执行失败时带上下文重新规划
  - Executor：针对单个子任务运行受限步数的 ReAct 循环（共享浏览器会话）
    - 集成阶段3的 VisionLocator：click 返回 ok=false 时截图做视觉定位兜底
  - Reviewer：审查每个子任务的执行结果，pass 进入下一个，fail 触发重规划
  - 状态经 LangGraph StateGraph 显式流转，支持长链路任务（>10 步）

验证（直接运行本文件）：
  /home/zkx456/other/bin/python3.13 04_agent_graph.py
"""

import importlib.util
import json
import os
import re
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

# ---- 动态导入前序阶段的模块 ----
_dir = os.path.dirname(os.path.abspath(__file__))
def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_dir, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

_bt_mod = _load("browser_tools", "01_browser_tools.py")
_ra_mod = _load("react_agent", "02_react_agent.py")
BrowserTool = _bt_mod.BrowserTool
TOOLS = _ra_mod.TOOLS
_fmt_elements = _ra_mod._fmt_elements
_load_env_from_bashrc = _ra_mod._load_env_from_bashrc
DASHSCOPE_BASE_URL = _ra_mod.DASHSCOPE_BASE_URL
AGENT_MODEL = _ra_mod.AGENT_MODEL

try:
    _vl_mod = _load("vision_locator", "03_vision_locator.py")
    _HAS_VL = True
except Exception:
    _HAS_VL = False

MAX_SUBTASK_STEPS = 6   # 单个子任务的最大 ReAct 步数
MAX_REPLAN = 2          # 最大重规划次数

PLANNER_PROMPT = """你是任务规划器。把用户的浏览器任务拆解为 2-6 个有序、可独立执行的子任务。
要求：每个子任务一句话、具体可验证；第一个子任务通常是打开某个网站。
只输出 JSON 数组，如 ["打开必应", "搜索关键词xxx", "阅读结果页找出第一条标题"]。"""

REVIEWER_PROMPT = """你是执行结果审查器。判断子任务是否已达成。
子任务：{subtask}
执行记录：{record}
只输出 JSON：{{"pass": true/false, "reason": "一句话原因"}}"""

EXECUTOR_PROMPT = """你是浏览器执行器。只完成当前子任务，不要做子任务之外的事。
页面上可交互元素已编号（eid），每次动作后编号会刷新，必须用最新编号。
当前页面已就绪：除非子任务要求跳转到新页面，否则不要重复 open_url 打开当前页。
click 返回 ok=false 时系统会自动尝试视觉定位兜底，无需你重复点击同一元素。
点击时尽量补上 desc（元素的中文描述），视觉兜底会用它定位。
子任务达成后立即调用 finish，answer 里写出该子任务的结果要点。"""


class AgentState(TypedDict):
    task: str
    plan: list[str]
    current: int          # 当前子任务下标
    findings: list[str]   # 各子任务结果摘要
    verdict: bool         # reviewer 对当前子任务的结论
    replan_count: int
    answer: str


class HierarchicalAgent:
    def __init__(self, headless: bool | None = None,
                 vision: bool = True, slow_mo: int | None = None,
                 profile_dir: str | None = None,
                 planner_prompt: str | None = None,
                 executor_prompt: str | None = None):
        self.bt = BrowserTool(headless=headless, slow_mo=slow_mo,
                              profile_dir=profile_dir).start()
        self.llm = OpenAI(api_key=_load_env_from_bashrc("DASHSCOPE_API_KEY"),
                          base_url=DASHSCOPE_BASE_URL, timeout=120)
        self.vl = _vl_mod.VisionLocator() if (vision and _HAS_VL) else None
        # 允许子类/调用方注入领域专用提示词（如携程订票模板）
        self.planner_prompt = planner_prompt or PLANNER_PROMPT
        self.executor_prompt = executor_prompt or EXECUTOR_PROMPT
        self.graph = self._build_graph()

    def close(self):
        self.bt.close()

    # ---------- Planner ----------
    def planner(self, state: AgentState) -> dict:
        user = f"任务：{state['task']}"
        if state["replan_count"] > 0:
            user += f"\n（上次执行到第 {state['current']+1} 步失败，已完成：{state['findings']}，请重新规划剩余步骤）"
        resp = self.llm.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "system", "content": self.planner_prompt},
                      {"role": "user", "content": user}],
            temperature=0)
        text = resp.choices[0].message.content
        m = re.search(r"\[.*\]", text, re.S)
        plan = json.loads(m.group(0)) if m else [state["task"]]
        print(f"\n[Planner] 计划（{len(plan)} 步）: {plan}")
        return {"plan": plan, "current": 0, "verdict": False}

    # ---------- Executor ----------
    def _vision_click(self, args: dict) -> bool:
        """DOM 点击失败时的视觉兜底：截图 -> VL 定位 -> 坐标点击"""
        if not self.vl:
            return False
        desc = args.get("desc") or args["eid"]
        try:
            pos = self.vl.locate(self.bt.screenshot(), desc)
        except Exception as e:
            print(f"    [Vision] 调用失败（{type(e).__name__}），跳过兜底")
            return False
        if not pos:
            return False
        print(f"    [Vision] DOM 点击失败，视觉定位到 {pos}，坐标点击")
        self.bt.click_xy(*pos)
        return True

    def _exec_tool(self, name: str, args: dict) -> str:
        # 与 02 共用同一套工具执行，只是额外注入视觉兜底
        return _ra_mod.exec_tool(self.bt, name, args, click_fallback=self._vision_click)

    def executor(self, state: AgentState) -> dict:
        subtask = state["plan"][state["current"]]
        print(f"\n[Executor] 子任务 {state['current']+1}/{len(state['plan'])}: {subtask}")
        messages = [{"role": "system", "content": self.executor_prompt},
                    {"role": "user", "content":
                     f"子任务：{subtask}\n"
                     f"当前页面：{self.bt.page.title()}（页面已就绪，可直接在此页操作）"}]
        record = ""
        last_action, last_obs = "", ""
        for step in range(1, MAX_SUBTASK_STEPS + 1):
            resp = self.llm.chat.completions.create(
                model=AGENT_MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                record = msg.content or ""
                break
            done = False
            for tc in msg.tool_calls:
                name, args = tc.function.name, json.loads(tc.function.arguments)
                print(f"  [step {step}] {name}({json.dumps(args, ensure_ascii=False)[:80]})")
                result = self._exec_tool(name, args)
                if result == "__FINISH__":
                    record = args.get("answer", "")
                    done = True
                    break
                last_action = f"{name}({json.dumps(args, ensure_ascii=False)[:60]})"
                last_obs = result
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if done:
                break
        if not record:
            # 达到步数上限仍未 finish：把最后动作与观察交给 reviewer，避免"无输出"一律判失败
            record = (f"（达到单子任务 {MAX_SUBTASK_STEPS} 步上限，未调用 finish）"
                      f"最后动作：{last_action}；最后观察：{last_obs[:300]}")
        findings = state["findings"] + [f"{subtask} => {record or '（无输出）'}"]
        return {"findings": findings}

    # ---------- Reviewer ----------
    def reviewer(self, state: AgentState) -> dict:
        subtask = state["plan"][state["current"]]
        record = state["findings"][-1]
        resp = self.llm.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": REVIEWER_PROMPT.format(subtask=subtask, record=record)}],
            temperature=0)
        text = resp.choices[0].message.content
        m = re.search(r"\{.*\}", text, re.S)
        ok = False
        if m:
            try:
                ok = bool(json.loads(m.group(0)).get("pass"))
            except json.JSONDecodeError:
                pass
        print(f"[Reviewer] {'PASS' if ok else 'FAIL'}: {text[:120]}")
        updates: dict = {"verdict": ok}
        if ok:
            updates["current"] = state["current"] + 1
        else:
            updates["replan_count"] = state["replan_count"] + 1
        return updates

    # ---------- 路由 ----------
    @staticmethod
    def _after_reviewer(state: AgentState) -> str:
        if state["verdict"]:
            return "finish" if state["current"] >= len(state["plan"]) else "executor"
        return "replan" if state["replan_count"] <= MAX_REPLAN else "finish"

    def _final(self, state: AgentState) -> dict:
        resp = self.llm.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content":
                       f"任务：{state['task']}\n各步骤结果：\n" + "\n".join(state["findings"])
                       + "\n请用 3-5 句话给出最终回答。"}],
            temperature=0)
        return {"answer": resp.choices[0].message.content}

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self.planner)
        g.add_node("executor", self.executor)
        g.add_node("reviewer", self.reviewer)
        g.add_node("final", self._final)
        g.set_entry_point("planner")
        g.add_edge("planner", "executor")
        g.add_edge("executor", "reviewer")
        g.add_conditional_edges("reviewer", self._after_reviewer,
                                {"executor": "executor", "replan": "planner", "finish": "final"})
        g.add_edge("final", END)
        return g.compile()

    def run(self, task: str) -> str:
        init: AgentState = {"task": task, "plan": [], "current": 0, "findings": [],
                            "verdict": False, "replan_count": 0, "answer": ""}
        final = self.graph.invoke(init)
        return final["answer"]


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "在必应搜索 OpenManus，打开第一个搜索结果链接，总结这个项目是做什么的"
    agent = HierarchicalAgent()
    try:
        answer = agent.run(task)
        print(f"\n[最终回答]\n{answer}")
    finally:
        agent.close()
