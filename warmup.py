"""warmup.py —— 完整版，先跑通

三个函数 = 你 Agent 循环里的三块砖：分发 / 裁剪 / 防御
运行：python warmup.py
"""

import json
from datetime import datetime


# ---------- 三个假工具（真的实现不重要，重点是分发怎么做的） ----------
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expr: str):
    return str(eval(expr))          # 真实项目里 eval 有风险，之后改白名单


def get_weather(city: str):
    return f"{city}：晴，26℃"


# ---------- 1. 字典分发 ----------
def dispatch(name, args):
    table = {
        "get_current_time": lambda a: get_current_time(),
        "calculate":        lambda a: calculate(a.get("expr", "")),
        "get_weather":      lambda a: get_weather(a.get("city", "")),
    }
    fn = table.get(name)
    if fn is None:
        return f"未知工具: {name}"
    return fn(args)


# ---------- 2. 消息裁剪：保留第 0 条 + 最后 keep 条 ----------
# 注意：这是最朴素的一版，留着当「改之前」的对照。
# 它会按条数一刀切，把 tool 消息和它的 assistant(tool_calls) 父消息拆散，
# 服务端按 tool_call_id 找不到父消息就返回 400。
# 真正在跑的实现见 react_loop.py 的 group_by_turn / take_last / trim_by_turn。
def trim_messages(messages, keep):
    if len(messages) <= keep + 1:
        return messages
    return [messages[0]] + messages[-keep:]


# ---------- 3. 安全解析：任何异常都不许往外抛 ----------
def safe_args(raw):
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(args, dict):
        return {}
    return args


if __name__ == "__main__":
    print(dispatch("get_current_time", {}))
    print(dispatch("calculate", {"expr": "25*4+10"}))
    print(dispatch("get_weather", {"city": "西安"}))
    print(dispatch("不存在的工具", {}))

    msgs = [{"i": 0}] + [{"i": i} for i in range(1, 9)]
    print(trim_messages(msgs, keep=3))

    print(safe_args('{"expr": "1+1"}'))
    print(safe_args("这不是json"))
    print(safe_args("[1,2]"))
