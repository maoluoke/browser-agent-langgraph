"""Step 3：视觉定位兜底（Qwen-VL 截图坐标定位）

背景：
  DOM 解析在复杂弹窗、canvas 渲染、反爬注入等场景会失效（click 返回 ok=false）。
  此时截取当前页面，让 Qwen-VL 直接根据截图返回目标元素的像素坐标，
  再用 page.mouse.click(x, y) 完成操作 —— 即「DOM + Vision 双重定位」。

成本控制：
  视觉调用按 token 计费（截图约 1-2k token/次），因此策略为 DOM 优先、
  Vision 仅在 DOM 定位失败时兜底，不进入主循环。

验证（直接运行本文件）：
  /home/zkx456/other/bin/python3.13 03_vision_locator.py
  -> 打开必应，纯视觉方式定位搜索框并点击、输入、提交，检查结果页标题
"""

import base64
import importlib.util
import os
import re

from openai import OpenAI

_spec = importlib.util.spec_from_file_location(
    "browser_tools", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_browser_tools.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BrowserTool = _mod.BrowserTool

_spec2 = importlib.util.spec_from_file_location(
    "react_agent", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_react_agent.py"))
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 本账号 VL 系列均未授权（403），改用已验证可用的全模态模型 qwen3.8-omni-flash 做截图定位；
# 若后续开通了 qwen3-vl-plus（GUI 定位精度更高），用 VISION_MODEL 环境变量切换即可
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3.8-omni-flash")
VIEWPORT = (1280, 800)  # 与 BrowserTool 视口一致，坐标系基准

_LOCATE_PROMPT = """这是一张 {w}x{h} 像素的浏览器页面截图。
请定位以下元素：{desc}
只输出该元素中心点的像素坐标，格式为 (x, y)，不要输出任何其他内容。"""


class VisionLocator:
    """截图 + Qwen-VL -> 元素像素坐标"""

    def __init__(self):
        self.llm = OpenAI(api_key=_mod2._load_env_from_bashrc("DASHSCOPE_API_KEY"),
                          base_url=DASHSCOPE_BASE_URL, timeout=120)

    def locate(self, screenshot: bytes, desc: str) -> tuple[int, int] | None:
        """根据截图定位元素，返回 (x, y)；失败返回 None"""
        b64 = base64.b64encode(screenshot).decode()
        resp = self.llm.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": _LOCATE_PROMPT.format(w=VIEWPORT[0], h=VIEWPORT[1], desc=desc)},
            ]}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\(?\s*(\d+)\s*[,，]\s*(\d+)\s*\)?", text)
        if not m:
            return None
        x, y = int(m.group(1)), int(m.group(2))
        if 0 <= x <= VIEWPORT[0] and 0 <= y <= VIEWPORT[1]:
            return (x, y)
        return None


if __name__ == "__main__":
    bt = BrowserTool().start()
    vl = VisionLocator()
    try:
        bt.open("https://www.bing.com")
        shot = bt.screenshot()
        print(f"[截图] {len(shot)} bytes，调用 {VISION_MODEL} 定位搜索框...")
        print("（若报 403 access_denied，请先到百炼控制台为该工作空间开通 VL 模型）")

        pos = vl.locate(shot, "搜索输入框（页面中央的长条形输入框）")
        print(f"[定位] 坐标: {pos}")
        assert pos, "视觉定位失败"

        # 纯坐标操作：点击 -> 键盘输入 -> 回车
        bt.click_xy(*pos)
        bt.page.keyboard.type("OpenManus")
        bt.page.keyboard.press("Enter")
        bt.page.wait_for_timeout(2000)
        print(f"[结果] 页面标题: {bt.page.title()}")
        assert "OpenManus" in bt.page.title()
        print("[完成] 阶段3验证通过：视觉定位 -> 坐标点击 -> 输入提交 全链路可用")
    finally:
        bt.close()
