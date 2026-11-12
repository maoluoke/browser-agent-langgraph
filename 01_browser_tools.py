"""Step 1：浏览器工具封装（Playwright）

功能：
  - 启动/关闭 Chromium（headless 可配）
  - open(url) 打开页面
  - elements() 提取精简 DOM：仅可见、可交互元素，注入 data-agent-eid 编号
  - click / type_text / press_key / scroll 按编号或坐标执行动作
  - screenshot() 截图（供阶段3视觉定位兜底使用）

设计要点：
  - DOM 提取结果只保留 eid/tag/type/text/href/中心坐标，控制 token 消耗
  - 每次动作后 DOM 会变化，必须重新调用 elements() 获取最新编号
  - 坐标 (x, y) 随 DOM 一起返回，视觉定位时可直接比对，无需二次计算

验证（直接运行本文件）：
  /home/zkx456/other/bin/python3.13 01_browser_tools.py
  -> 打开百度，搜索关键词，打印元素列表与结果页标题
"""

import os
import re
import sys

from playwright.sync_api import sync_playwright, Page, Browser

try:
    from tf_playwright_stealth import stealth_sync
    _HAS_STEALTH = True
except ImportError:
    _HAS_STEALTH = False  # 未安装 tf-playwright-stealth 时静默降级

# 注入页面提取可交互元素的 JS：返回编号、标签、文本、中心坐标
_EXTRACT_JS = """
() => {
  // 清理上一次注入的编号，避免残留
  document.querySelectorAll('[data-agent-eid]').forEach(el => el.removeAttribute('data-agent-eid'));
  const sel = 'a, button, input, select, textarea, [role="button"], [role="link"], [role="tab"], [onclick], [contenteditable="true"]';
  const raw = [];
  let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (rect.width < 2 || rect.height < 2) continue;
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    const tag = el.tagName.toLowerCase();
    const text = (el.innerText || el.value || el.getAttribute('aria-label')
                  || el.getAttribute('placeholder') || el.getAttribute('title') || '').trim();
    // 优先级：表单控件(0) > 有文字的按钮(1) > 链接(2)；无文字与页脚/备案信息降到最低
    // 交易类网站（携程等）导航与页脚元素极多，若不排序会把表单控件挤出 120 上限
    let score = 2;
    if (tag === 'input' || tag === 'select' || tag === 'textarea') score = 0;
    else if (tag === 'button' || el.getAttribute('role') === 'button') score = 1;
    if (!text && !['input', 'select', 'textarea'].includes(tag)) score += 2;
    if (el.closest('footer') || /ICP|Copyright|备案|营业执照/.test(text)) score += 3;
    raw.push({
      order: n++, score: score, el: el, tag: tag,
      type: el.getAttribute('type') || '',
      text: text.slice(0, 60),
      href: (el.getAttribute('href') || '').slice(0, 80),
      x: Math.round(rect.x + rect.width / 2),
      y: Math.round(rect.y + rect.height / 2),
    });
  }
  raw.sort((a, b) => a.score - b.score || a.order - b.order);
  const results = [];
  raw.slice(0, 120).forEach((item, i) => {   // 上限保护，避免超长页面撑爆上下文
    const eid = 'e' + i;
    item.el.setAttribute('data-agent-eid', eid);
    results.push({eid: eid, tag: item.tag, type: item.type, text: item.text,
                  href: item.href, x: item.x, y: item.y});
  });
  return results;
}
"""


class BrowserTool:
    """Playwright 同步 API 封装：一个实例对应一个浏览器页面"""

    def __init__(self, headless: bool | None = None, stealth: bool = True,
                 slow_mo: int | None = None, profile_dir: str | None = None):
        # 默认无头；调试时设 AGENT_HEADLESS=0 弹出真实窗口，AGENT_SLOW_MO=毫秒 放慢每步操作
        if headless is None:
            headless = os.getenv("AGENT_HEADLESS", "1").lower() not in ("0", "false", "no")
        self.headless = headless
        self.slow_mo = slow_mo if slow_mo is not None else int(os.getenv("AGENT_SLOW_MO", "0"))
        self.stealth = stealth and _HAS_STEALTH
        # 登录态持久化：指定目录后复用同一浏览器 profile（Cookie/本地存储都保留）
        if profile_dir is None:
            profile_dir = os.getenv("AGENT_PROFILE_DIR", "")
        self.profile_dir = profile_dir or None
        self._pw = None
        self._browser: Browser | None = None
        self._ctx = None
        self.page: Page | None = None

    def start(self):
        self._pw = sync_playwright().start()
        ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/131.0.0.0 Safari/537.36")
        common = dict(
            headless=self.headless,
            slow_mo=self.slow_mo,  # 每个动作后延迟，肉眼可看清点击/输入过程
            args=["--disable-blink-features=AutomationControlled"],
        )
        if self.profile_dir:
            # 持久化 profile：首次登录一次，后续所有运行自动带着登录态
            os.makedirs(self.profile_dir, exist_ok=True)
            self._ctx = self._pw.chromium.launch_persistent_context(
                self.profile_dir,
                viewport={"width": 1280, "height": 800},
                user_agent=ua,
                **common,
            )
            self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        else:
            self._browser = self._pw.chromium.launch(**common)
            # 固定视口 1280x800：视觉定位的坐标系以此为基准
            # 使用真实桌面 UA，降低 headless 被反爬识别的概率
            self._ctx = self._browser.new_context(
                viewport={"width": 1280, "height": 800}, user_agent=ua)
            self.page = self._ctx.new_page()
        if self.stealth:
            stealth_sync(self.page)  # 抹除 navigator.webdriver 等自动化指纹
        return self

    def close(self):
        if self._ctx:
            self._ctx.close()   # 持久化模式下关闭即自动落盘 Cookie
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    # ---------- 导航 ----------
    def open(self, url: str) -> str:
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self.page.wait_for_timeout(1000)  # 等动态内容渲染
        return self.page.title()

    # ---------- DOM 提取 ----------
    def elements(self) -> list[dict]:
        """提取可交互元素；页面正在跳转/刷新时返回空列表，由调用方下一轮重取"""
        try:
            return self.page.evaluate(_EXTRACT_JS)
        except Exception:
            return []

    def page_text(self, max_len: int = 2000) -> str:
        """提取页面正文文本，供 LLM 阅读内容用"""
        try:
            text = self.page.evaluate("() => document.body.innerText")
        except Exception:
            return ""   # 跳转中读不到正文，调用方按"暂无内容"处理
        return text[:max_len]

    # ---------- 动作执行 ----------
    def _sel(self, eid: str) -> str:
        return f'[data-agent-eid="{eid}"]'

    def click(self, eid: str) -> bool:
        try:
            self.page.click(self._sel(eid), timeout=5000)
            self.page.wait_for_timeout(800)
            return True
        except Exception:
            return False  # 返回 False 触发视觉定位兜底

    def click_xy(self, x: int, y: int) -> None:
        """视觉兜底：直接按视口坐标点击"""
        self.page.mouse.click(x, y)
        self.page.wait_for_timeout(800)

    def type_text(self, eid: str, text: str, submit: bool = False,
                  delay_ms: int = 30) -> bool:
        """逐字输入：先点聚焦、清空，再按字符输入。
        React/Vue 受控组件只认真实键盘事件，直接改写 value 不会触发 onChange。"""
        try:
            sel = self._sel(eid)
            self.page.click(sel, timeout=5000)
            self.page.fill(sel, "")
            self.page.type(sel, text, delay=delay_ms)
            if submit:
                self.page.press(sel, "Enter")
                self.page.wait_for_timeout(1500)
            return True
        except Exception:
            # 兼容性兜底：部分输入框不支持逐字输入，退回 fill
            try:
                self.page.fill(self._sel(eid), text, timeout=5000)
                if submit:
                    self.page.press(self._sel(eid), "Enter")
                    self.page.wait_for_timeout(1500)
                return True
            except Exception:
                return False

    def press_key(self, key: str) -> None:
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(800)

    def scroll(self, direction: str = "down", amount: int = 600) -> None:
        dy = amount if direction == "down" else -amount
        self.page.mouse.wheel(0, dy)
        self.page.wait_for_timeout(600)

    # ---------- 交易类网站常用交互 ----------
    def click_text(self, text: str, exact: bool = False) -> bool:
        """按可见文字点击：日期、城市、车次等无稳定 id 的元素用这个最可靠"""
        try:
            self.page.get_by_text(text, exact=exact).first.click(timeout=6000)
            self.page.wait_for_timeout(800)
            return True
        except Exception:
            return False

    def click_nth(self, text: str, index: int = 0, exact: bool = False) -> bool:
        """点击第 index 个（从 0 开始）文字匹配的元素。
        结果页常有多个同名「预订」按钮，按顺序对应屏幕上从上到下的车次/酒店。"""
        try:
            self.page.get_by_text(text, exact=exact).nth(index).click(timeout=6000)
            self.page.wait_for_timeout(800)
            return True
        except Exception:
            return False

    def select_suggestion(self, text: str) -> bool:
        """从自动补全下拉中选中「首行文字完全等于 text」的条目。
        携程候选条目形如「北京\\nbeijing」「北京北\\nbeijingbei」，只比首行；
        否则 click_text(exact) 会点到「北京北」这类把「北京」高亮成前缀的项。"""
        target = re.sub(r"\s+", "", text)
        try:
            items = self.page.locator("li, [role='option']")
            for i in range(min(items.count(), 80)):
                it = items.nth(i)
                try:
                    if not it.is_visible():
                        continue
                    first = it.inner_text().strip().split("\n")[0]
                except Exception:
                    continue
                if re.sub(r"\s+", "", first) == target:
                    it.click(timeout=5000)
                    self.page.wait_for_timeout(800)
                    return True
        except Exception:
            pass
        return self.click_text(text, exact=True)  # 兜底：没有 li 结构时退回文字点击

    def select_option(self, eid: str, value: str) -> bool:
        """下拉框选择：依次尝试 value / label / 包含匹配"""
        sel = self._sel(eid)
        for kwargs in ({"value": value}, {"label": value}):
            try:
                self.page.select_option(sel, timeout=5000, **kwargs)
                self.page.wait_for_timeout(500)
                return True
            except Exception:
                continue
        return False

    def check(self, eid: str, checked: bool = True) -> bool:
        """勾选/取消勾选单选、复选框"""
        try:
            (self.page.check if checked else self.page.uncheck)(self._sel(eid), timeout=5000)
            self.page.wait_for_timeout(500)
            return True
        except Exception:
            return False

    def wait_for(self, keyword: str | None = None, selector: str | None = None,
                 timeout: float = 10) -> bool:
        """等待文字出现或元素出现；交易网站结果页/支付页加载慢，必须显式等待"""
        try:
            if selector:
                self.page.wait_for_selector(selector, timeout=int(timeout * 1000))
                return True
            if keyword:
                self.page.get_by_text(keyword).first.wait_for(timeout=int(timeout * 1000))
                return True
        except Exception:
            return False
        return False

    def switch_tab(self, index: int = -1) -> str:
        """切换标签页（支付页常在新 tab 打开），index=-1 表示最后一个"""
        try:
            pages = self._ctx.pages
            if not pages:
                return "无可用标签页"
            self.page = pages[index]
            self.page.wait_for_timeout(500)
            return self.page.title()
        except Exception:
            return "切换失败"

    def pause_for_human(self, prompt: str = "请在浏览器中完成人工操作（登录/验证码/支付），完成后按回车继续") -> str:
        """人工接管：登录、滑块验证、支付等关键环节必须由人完成"""
        print(f"\n[需要人工操作] {prompt}")
        if not sys.stdin.isatty():
            return "（非交互环境，跳过人工等待）"
        input(">>> 完成后按回车继续...")
        return "人工操作已完成"

    def screenshot(self, path: str | None = None) -> bytes:
        return self.page.screenshot(path=path, type="png")


if __name__ == "__main__":
    bt = BrowserTool().start()
    try:
        # 注：百度对 headless 按 IP+行为风控，会弹安全验证页，故自测改用必应
        title = bt.open("https://www.bing.com")
        print(f"[打开] 标题: {title}")

        els = bt.elements()
        print(f"[DOM] 提取到 {len(els)} 个可交互元素，前 10 个:")
        for el in els[:10]:
            print(f"  {el['eid']:>4} <{el['tag']}> {el['text'][:30]!r} ({el['x']},{el['y']})")

        # 找搜索框输入并提交（百度首页搜索框是 textarea，故两种标签都匹配）
        kw = next((e for e in els
                   if (e["tag"] == "input" and e["type"] in ("text", "search", ""))
                   or e["tag"] == "textarea"), None)
        if kw and bt.type_text(kw["eid"], "OpenManus 架构", submit=True):
            print(f"[搜索] 结果页标题: {bt.page.title()}")
            els2 = bt.elements()
            links = [e for e in els2 if e["tag"] == "a" and e["text"]][:5]
            print("[结果] 前 5 个链接:")
            for e in links:
                print(f"  {e['eid']:>4} {e['text'][:40]!r}")
        else:
            print("[搜索] 未找到搜索框或输入失败")
    finally:
        bt.close()
    print("[完成] 阶段1验证通过")
