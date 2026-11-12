"""Step 7：携程订票 / 订房（确定性流程，可选大模型接管）

为什么默认不用"大模型逐步点页面"：
  携程的表单是自定义组件，每一步都让模型决策要等一次网络响应（几十秒），
  既慢又容易重复输入、卡在中途。城市、日期、搜索这几步实测都有稳定路径，
  因此默认由代码直接执行；大模型只在最后读结果做归纳（--llm）。

安全边界：只做到"搜索/进入订单填写页"，绝不点提交订单与支付。

使用：
  # 火车票
  python3.13 07_ctrip_agent.py --mode train --from 上海 --to 北京 --date 2026-11-05

  # 酒店（首次需先登录，见下）
  python3.13 07_ctrip_agent.py --mode hotel --city 上海 --checkin 2026-11-05 --checkout 2026-11-07

  # 首次登录（登录态保存到 project3/.browser_profile，后续免登录）
  python3.13 07_ctrip_agent.py --login

  # 加 --llm 用大模型归纳结果；--agent --task "..." 走原来的自主决策模式
"""

import argparse
import importlib.util
import os
import re
from datetime import date, datetime

from openai import OpenAI

_dir = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_dir, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_ag = _load("agent_graph", "04_agent_graph.py")
_bt_mod = _load("browser_tools", "01_browser_tools.py")
_ra = _load("react_agent", "02_react_agent.py")
HierarchicalAgent = _ag.HierarchicalAgent
BrowserTool = _bt_mod.BrowserTool

DEFAULT_PROFILE = os.path.join(_dir, ".browser_profile")

# 提交订单 / 支付类危险动作：命中即拦截转人工（宁可多拦，不可漏拦）
PAYMENT_KEYWORDS = (
    "提交订单", "确认订单", "提交预订", "确认预订", "立即预订并支付",
    "立即支付", "确认支付", "去支付", "确认付款", "付款码",
    "免密支付", "密码支付", "收银台", "支付订单",
)
PAYMENT_PAGE_KEYWORDS = ("收银台", "支付订单", "订单支付", "确认付款")

TRAIN_URL = "https://trains.ctrip.com/"
HOTEL_URL = "https://hotels.ctrip.com/"

# 用来判定"搜索真的生效了"的结果页特征词（搜索表单页本身不含这些词）
TRAIN_RESULT_KEYWORDS = ("余票", "二等座", "一等座", "商务座", "无座")
HOTEL_RESULT_KEYWORDS = ("筛选", "每晚", "评分", "¥")

# ---------- 大模型（仅用于结果归纳 / 自主模式） ----------
def _llm():
    return OpenAI(api_key=_ra._load_env_from_bashrc("DASHSCOPE_API_KEY"),
                  base_url=_ra.DASHSCOPE_BASE_URL, timeout=120)


# ---------- 通用小工具 ----------
def _sel(eid: str) -> str:
    return f'[data-agent-eid="{eid}"]'


def _input_value(bt, sel: str) -> str:
    """读取输入框当前值；页面跳转中读不到时返回空串，由调用方重试"""
    try:
        return _input_value(bt, sel) or ""
    except Exception:
        return ""


def _set_city(bt, index: int, city: str, y_min: int = 200, y_max: int = 520) -> bool:
    """把第 index 个（按 x 排序）城市输入框设为目标城市。
    必须先清空再逐字输入，等下拉出现后用 select_suggestion 选完整条目，
    否则携程会把「北京」当成「北京北」的前缀。"""
    for _ in range(3):
        els = sorted([e for e in bt.elements()
                      if e["tag"] == "input" and y_min < e["y"] < y_max], key=lambda e: e["x"])
        if len(els) <= index:
            return False
        eid = els[index]["eid"]
        if _input_value(bt, _sel(eid)) == city:
            return True
        bt.page.click(_sel(eid), click_count=3)
        bt.page.keyboard.press("Backspace")
        bt.page.type(_sel(eid), city, delay=80)
        bt.page.wait_for_timeout(1200)
        bt.select_suggestion(city)
        bt.page.wait_for_timeout(400)
        if _input_value(bt, _sel(eid)) == city:
            return True
    return False


def _wait_any(bt, keywords, timeout: float = 8) -> str | None:
    """轮询页面文本，出现任一关键词即返回该关键词"""
    for _ in range(int(timeout * 2)):
        text = bt.page_text(2500)
        for kw in keywords:
            if kw in text:
                return kw
        bt.page.wait_for_timeout(500)
    return None


def _search_buttons(bt):
    """表单里的搜索按钮：取页面最靠下的那几个（页头也有一个叫搜索的入口，点了没反应）"""
    els = bt.elements()
    btns = [e for e in els
            if e["tag"] in ("button", "a", "div", "span")
            and e["text"].strip().replace(" ", "") in ("搜索", "查询", "立即搜索", "搜一下")]
    btns.sort(key=lambda e: -e["y"])
    return btns[:3]


def _vision_click_search(bt, keywords) -> bool:
    """最后兜底：截图交给视觉模型定位「搜索」按钮，再按坐标点击"""
    try:
        vl = _load("vision_locator", "03_vision_locator.py").VisionLocator()
        pos = vl.locate(bt.screenshot(), "表单最后一行右侧的搜索按钮")
    except Exception as e:
        print(f"    [Vision] 不可用（{type(e).__name__}），跳过")
        return False
    if not pos:
        return False
    print(f"    [Vision] 视觉定位到 {pos}，按坐标点击")
    bt.click_xy(*pos)
    return _wait_any(bt, keywords, 8) is not None


def _click_search(bt, keywords) -> bool:
    """点击搜索并验证结果真的出现；失败依次退化为：回车提交、视觉定位点击"""
    bt.page.keyboard.press("Escape")                       # 先关掉日历等弹层，避免遮挡按钮
    try:
        bt.page.evaluate("() => window.scrollTo(0, 0)")
    except Exception:
        pass                                               # 页面正在跳转时滚动会失败，忽略
    bt.page.wait_for_timeout(400)
    for e in _search_buttons(bt):
        print(f"    点候选搜索按钮「{e['text'][:12]}」{e['tag']} @({e['x']},{e['y']})")
        bt.page.mouse.click(e["x"], e["y"])                # 按坐标点，绕过可见性/遮挡判断
        hit = _wait_any(bt, keywords, 8)
        if hit:
            print(f"    搜索生效（结果页出现「{hit}」）")
            return True
        print("    未出现结果，换下一个候选")
        bt.page.keyboard.press("Escape")
    # 兜底1：在到达城市输入框里按回车提交查询
    els = sorted([e for e in bt.elements()
                  if e["tag"] == "input" and 200 < e["y"] < 560], key=lambda e: e["x"])
    if els:
        print("    兜底：在输入框按 Enter 提交")
        bt.page.click(_sel(els[-1]["eid"]))
        bt.page.keyboard.press("Enter")
        if _wait_any(bt, keywords, 8):
            return True
    # 兜底2：视觉定位
    return _vision_click_search(bt, keywords)


def _month_label(bt, target: date) -> None:
    """尽量切到目标月份；页面没有该月份标签时忽略"""
    for label in (f"{target.year}年{target.month}月", f"{target.year}年 {target.month}月"):
        try:
            if bt.page.get_by_text(label).first.is_visible():
                bt.click_text(label)
                bt.page.wait_for_timeout(300)
                return
        except Exception:
            continue


# ---------- 火车票 ----------
def _train_date_text(bt) -> str:
    try:
        return bt.page.evaluate(
            "() => { const el = document.querySelector('div.date, .search-date');"
            " return el ? el.innerText.replace(/\\n/g, ' ') : ''; }")
    except Exception:
        return ""


def _pick_train_date(bt, target: date) -> bool:
    want = f"{target.year}-{target.month:02d}-{target.day:02d}"
    if want in _train_date_text(bt):
        return True
    bt.click_text("出发日期")
    bt.page.wait_for_timeout(600)
    _month_label(bt, target)
    bt.click_nth(str(target.day), 0, exact=True)   # 日历格子只有数字
    bt.page.wait_for_timeout(700)
    return want in _train_date_text(bt)


def run_train(bt, args) -> str:
    bt.open(TRAIN_URL)
    print(f"  城市: {args.from_city} → {args.to_city}   日期: {args.date}")
    if not _set_city(bt, 0, args.from_city):
        print(f"  [警告] 出发城市未成功设为 {args.from_city}")
    if not _set_city(bt, 1, args.to_city):
        print(f"  [警告] 到达城市未成功设为 {args.to_city}")
    print(f"  日期选择: {'成功' if _pick_train_date(bt, args.date) else '失败'}"
          f"（{_train_date_text(bt)[:40]}）")
    print("  点击搜索...")
    if not _click_search(bt, TRAIN_RESULT_KEYWORDS):
        print("  [警告] 搜索未生效（可能被弹层拦截或触发风控验证）")
        return ""
    print(f"  结果页: {bt.page.title()}")
    return bt.page_text(4000)


# ---------- 酒店 ----------
def _hotel_inputs(bt):
    # 表单在 y≈300 一行：左起目的地、入住、离店，右侧是"位置/品牌/酒店"
    # 不能靠"目的地"占位文字找：携程会记住上次搜索，字段一打开就是已填值
    ins = [e for e in bt.elements() if e["tag"] == "input" and 250 < e["y"] < 420 and e["x"] < 700]
    date_ins = [e for e in ins if re.search(r"\d+月\d+日", e["text"])]
    dest = next((e for e in ins if not re.search(r"\d+月\d+日", e["text"])), None)
    if dest:
        date_ins.sort(key=lambda e: e["x"])
        date_ins = [e for e in date_ins if e["x"] > dest["x"]]
    return dest, date_ins


def _pick_hotel_date(bt, eid: str, target: date) -> bool:
    want = f"{target.month}月{target.day}日"
    sel = _sel(eid)
    if want in _input_value(bt, sel):
        return True
    bt.page.click(sel)
    bt.page.wait_for_timeout(600)
    _month_label(bt, target)
    bt.click_nth(str(target.day), 0, exact=True)
    bt.page.wait_for_timeout(800)
    return want in _input_value(bt, sel)


def run_hotel(bt, args) -> str:
    bt.open(HOTEL_URL)
    dest, dates = None, []
    for _ in range(8):                     # 酒店首页表单异步渲染，最多等 8 秒
        dest, dates = _hotel_inputs(bt)
        if dest and len(dates) >= 2:
            break
        bt.page.wait_for_timeout(1000)
    if not dest or len(dates) < 2:
        print("  [警告] 未识别到目的地/入住离店输入框")
        return ""
    if not _set_city_eid(bt, dest["eid"], args.city, y_min=200, y_max=520):
        print(f"  [警告] 目的地未成功设为 {args.city}")
    dest, dates = _hotel_inputs(bt)
    print(f"  日期: {args.checkin} 入住 / {args.checkout} 离店")
    print(f"  入住选择: {'成功' if _pick_hotel_date(bt, dates[0]['eid'], args.checkin) else '失败'}")
    _, dates = _hotel_inputs(bt)
    print(f"  离店选择: {'成功' if _pick_hotel_date(bt, dates[-1]['eid'], args.checkout) else '失败'}")
    print("  点击搜索...")
    ok = _click_search(bt, HOTEL_RESULT_KEYWORDS)
    print(f"  搜索{'成功' if ok else '未生效'}")
    bt.page.wait_for_timeout(1500)
    if "passport.ctrip.com" in bt.page.url:
        print("  [需要登录] 酒店搜索会强制跳转登录页")
        bt.pause_for_human("请在浏览器中登录携程账号，登录后按回车继续")
        bt.wait_for("酒店", timeout=30)
        if "passport.ctrip.com" in bt.page.url:
            print("  仍未登录，请先执行 --login 登录一次")
            return ""
    print(f"  结果页: {bt.page.title()}")
    return bt.page_text(3500)


def _set_city_eid(bt, eid: str, city: str, y_min: int = 200, y_max: int = 520) -> bool:
    """指定 eid 的输入框设为城市（酒店目的地用）"""
    for _ in range(3):
        sel = _sel(eid)
        try:
            if _input_value(bt, sel) == city:
                return True
            bt.page.click(sel, click_count=3)
            bt.page.keyboard.press("Backspace")
            bt.page.type(sel, city, delay=80)
            bt.page.wait_for_timeout(1200)
            bt.select_suggestion(city)
            bt.page.wait_for_timeout(400)
            return _input_value(bt, sel) == city
        except Exception:
            els = sorted([e for e in bt.elements()
                          if e["tag"] == "input" and y_min < e["y"] < y_max], key=lambda e: e["x"])
            if not els:
                return False
            eid = els[0]["eid"]
    return False


# ---------- 结果归纳（可选） ----------
def llm_summary(text: str, mode: str, extra: str = "") -> str:
    if mode == "train":
        ask = ("下面是携程火车票搜索结果页的文本。请列出最早出发的三趟车次，"
               "格式：车次号 出发时间-到达时间 出发站-到达站 历时 票价。只输出这三行。")
    else:
        ask = ("下面是携程酒店搜索结果页的文本。请推荐评分最高的三家酒店，"
               "格式：酒店名 评分 价格 位置。只输出这三行。")
    resp = _llm().chat.completions.create(
        model=_ra.AGENT_MODEL,
        messages=[{"role": "user", "content": f"{ask}\n用户需求：{extra}\n\n页面文本：\n{text}"}],
        temperature=0)
    return resp.choices[0].message.content


# ---------- 大模型自主模式（保留原有能力） ----------
TRAIN_PLANNER_PROMPT = """你是携程火车票任务的规划器。把用户需求拆解为 2-6 个有序、可独立验证的子任务。
浏览器当前已打开携程火车票页面，不要规划"打开网站"这类步骤。
参考流程：选出发城市 → 选到达城市 → 选出发日期 → 点查询 → 筛选车次 → 进入预订页。
硬性约束：
- 最后一步固定为「到达订单确认/填写页后交还用户，由人工核对并自行提交订单与支付」
- 绝不规划提交订单、支付、付款类子任务
只输出 JSON 数组"""

TRAIN_EXECUTOR_PROMPT = """你是携程火车票执行器，只完成当前子任务。
页面上可交互元素已编号（eid），格式 eid<标签>文本@y坐标；每次动作后编号会刷新。
实测有效的操作路径：
- 城市：type_text 写入输入框后，用 select_suggestion 选城市全称（不要用 click_text，
  否则会点到「北京北」这类含前缀的站点）；选中后不要再重复 type_text
- 日期：click_text「出发日期」打开日历；日历里只有纯数字，用 click_nth(数字, 0, exact=true)
- 查询：点 tag=button 的「搜索」
- 结果页：用 click_nth("预订", index) 选第 index 条车次
- 已进入订单填写/确认页：立即 finish，不要再点任何提交/支付按钮
- 登录/滑块/验证码：调用 pause_for_human
子任务达成后立即调用 finish。"""

HOTEL_PLANNER_PROMPT = """你是携程酒店预订任务的规划器。把用户需求拆解为 2-6 个有序子任务。
浏览器当前已打开携程酒店页面，不要规划"打开网站"这类步骤。
参考流程：填目的地 → 选入住 → 选离店 → 点搜索 → 筛选酒店 → 进入房型预订页。
硬性约束：最后一步固定为「到达订单填写页后交还用户」，绝不规划提交订单与支付。
只输出 JSON 数组"""

HOTEL_EXECUTOR_PROMPT = """你是携程酒店执行器，只完成当前子任务。
实测路径：目的地 type_text + select_suggestion；日期点输入框后用 click_nth(数字, 0, exact=true)；
搜索点 tag=button 的「搜索」；搜索会跳转登录页(passport.ctrip.com)，此时调用 pause_for_human。
已进入订单填写页立即 finish，不要点提交/支付。子任务达成后立即调用 finish。"""

MODES = {
    "train": dict(planner=TRAIN_PLANNER_PROMPT, executor=TRAIN_EXECUTOR_PROMPT, url=TRAIN_URL),
    "hotel": dict(planner=HOTEL_PLANNER_PROMPT, executor=HOTEL_EXECUTOR_PROMPT, url=HOTEL_URL),
}


class CtripAgent(HierarchicalAgent):
    """阶段4 三层状态机 + 代码级安全闸门"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._login_paused = False

    def _login_redirect(self) -> bool:
        try:
            url = self.bt.page.url or ""
        except Exception:
            return False
        if "passport.ctrip.com" not in url:
            self._login_paused = False
            return False
        if self._login_paused:
            return False
        self._login_paused = True
        return True

    def _danger(self, name: str, args: dict) -> str | None:
        text = ""
        if name == "click_text":
            text = args.get("text", "")
        elif name == "click":
            text = args.get("desc", "") or ""
            try:
                for el in self.bt.elements():
                    if el["eid"] == args.get("eid"):
                        text = f"{text} {el['text']}"
                        break
            except Exception:
                pass
        elif name != "press_key":
            return None
        hit = next((k for k in PAYMENT_KEYWORDS if k and k in text), None)
        if hit:
            return hit
        try:
            title = self.bt.page.title() or ""
        except Exception:
            title = ""
        if any(k in title for k in PAYMENT_PAGE_KEYWORDS):
            return f"页面已进入支付环节（{title[:30]}）"
        return None

    def _exec_tool(self, name: str, args: dict) -> str:
        if self._login_redirect():
            msg = self.bt.pause_for_human("页面跳转到携程登录页，请人工登录，完成后按回车继续")
            return f"【需要登录】{msg}。登录后请重新观察页面并继续当前子任务。"
        hit = self._danger(name, args)
        if hit:
            msg = self.bt.pause_for_human(
                f"检测到「{hit}」，属于提交订单/支付环节，必须由人工核对并操作")
            return (f"【安全边界】{msg}。提交订单/支付已交由人工处理，"
                    f"本子任务到此结束，请立即调用 finish(answer='已到达「{hit}」环节，已交人工处理')")
        return super()._exec_tool(name, args)


def do_login(profile: str, slow_mo: int) -> None:
    bt = BrowserTool(headless=False, slow_mo=slow_mo, profile_dir=profile).start()
    try:
        bt.open("https://www.ctrip.com/")
        print("[登录模式] 请在浏览器窗口完成登录（扫码/账号密码/短信均可）")
        bt.pause_for_human("登录完成并看到已登录状态后按回车")
        print(f"[登录模式] 登录态已保存：{profile}")
    finally:
        bt.close()


def _parse_date(s: str | None, field: str) -> date:
    if not s:
        raise SystemExit(f"缺少参数 --{field}，格式如 2026-11-05")
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        m = re.match(r"(\d{1,2})月(\d{1,2})日", s.strip())
        if m:
            return date(date.today().year, int(m.group(1)), int(m.group(2)))
        raise SystemExit(f"--{field} 无法解析：{s}（请用 2026-11-05 格式）")


def main() -> None:
    ap = argparse.ArgumentParser(description="携程订票/订房（止步于订单页，支付由人工完成）")
    ap.add_argument("--mode", choices=["train", "hotel"], default="train")
    ap.add_argument("--from", dest="from_city", help="出发城市（火车票）")
    ap.add_argument("--to", dest="to_city", help="到达城市（火车票）")
    ap.add_argument("--date", help="出发日期 YYYY-MM-DD（火车票）")
    ap.add_argument("--city", help="目的地城市（酒店）")
    ap.add_argument("--checkin", help="入住日期 YYYY-MM-DD（酒店）")
    ap.add_argument("--checkout", help="离店日期 YYYY-MM-DD（酒店）")
    ap.add_argument("--llm", action="store_true", help="用大模型归纳结果（默认只打印页面文本）")
    ap.add_argument("--agent", action="store_true", help="改用 LLM 自主决策模式（配合 --task）")
    ap.add_argument("--task", default="", help="--agent 模式下的自然语言任务")
    ap.add_argument("--login", action="store_true", help="仅登录携程并保存登录态")
    ap.add_argument("--profile", default=os.getenv("AGENT_PROFILE_DIR") or DEFAULT_PROFILE)
    ap.add_argument("--headless", action="store_true", help="无头模式（默认弹真实窗口）")
    ap.add_argument("--slow-mo", type=int, default=int(os.getenv("AGENT_SLOW_MO", "200")))
    args = ap.parse_args()

    if args.login:
        do_login(args.profile, args.slow_mo)
        return

    print("=" * 60)
    print(f"[携程] 模式: {'LLM 自主' if args.agent else '确定性流程'}   profile: {args.profile}")
    print("[携程] 安全边界: 只做到搜索/订单填写页；提交订单与支付由人工完成")
    print("=" * 60)

    if args.agent:
        cfg = MODES[args.mode]
        task = args.task or ("查询 2026-11-05 上海到北京的高铁二等座" if args.mode == "train"
                             else "查询上海 2026-11-05 入住 2 晚的酒店")
        agent = CtripAgent(headless=args.headless, vision=True, slow_mo=args.slow_mo,
                           profile_dir=args.profile,
                           planner_prompt=cfg["planner"], executor_prompt=cfg["executor"])
        try:
            print(f"[携程] 入口页: {agent.bt.open(cfg['url'])}")
            print(f"\n[最终回答]\n{agent.run('（浏览器已打开 ' + cfg['url'] + '）任务：' + task)}")
            agent.bt.pause_for_human("核对/下单/支付完成后按回车关闭浏览器")
        finally:
            agent.close()
        return

    # ---- 确定性流程 ----
    if args.mode == "train":
        if not (args.from_city and args.to_city):
            raise SystemExit("火车票需要 --from 和 --to，例如 --from 上海 --to 北京 --date 2026-11-05")
        args.date = _parse_date(args.date, "date")
    else:
        if not (args.city and args.checkin and args.checkout):
            raise SystemExit("酒店需要 --city --checkin --checkout，例如 --city 上海 --checkin 2026-11-05 --checkout 2026-11-07")
        args.checkin = _parse_date(args.checkin, "checkin")
        args.checkout = _parse_date(args.checkout, "checkout")

    bt = BrowserTool(headless=args.headless, slow_mo=args.slow_mo,
                     profile_dir=args.profile).start()
    try:
        text = run_train(bt, args) if args.mode == "train" else run_hotel(bt, args)
        if not text:
            print("[结果] 未获取到结果页，请查看浏览器窗口状态")
        elif args.llm:
            extra = (f"{args.from_city}到{args.to_city} {args.date}" if args.mode == "train"
                     else f"{args.city} {args.checkin}-{args.checkout}")
            print(f"\n[大模型归纳]\n{llm_summary(text, args.mode, extra)}")
        else:
            print(f"\n[结果页文本（前 1200 字）]\n{text[:1200]}")
            print("\n提示：加 --llm 可让大模型归纳成三行结论")
        bt.pause_for_human("查看结果后按回车关闭浏览器（如需下单请自行操作，程序不会提交订单）")
    finally:
        bt.close()


if __name__ == "__main__":
    main()
