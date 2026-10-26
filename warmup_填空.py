"""warmup_填空.py —— warmup.py 挖掉 4 个空

正确输出（warmup.py 跑出来的，记住它）：
    2026-10-26 15:27:57
    110
    西安：晴，26℃
    未知工具: 不存在的工具
    [{'i': 0}, {'i': 6}, {'i': 7}, {'i': 8}]
    {'expr': '1+1'}
    {}
    {}

规则：
  1. 先跑 python warmup.py，看一遍正确输出
  2. 打开本文件，把 4 处 `...` 换成真代码（每处 1 行）
  3. 跑 python warmup_填空.py，输出必须和上面一模一样
  4. 卡住超过 10 分钟：允许偷看 warmup.py 那一行，看完关掉，自己重新敲一遍
"""

import json
from datetime import datetime


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expr: str):
    return str(eval(expr))


def get_weather(city: str):
    return f"{city}：晴，26℃"


def dispatch(name, args):
    table = {
        "get_current_time": lambda a: get_current_time(),
        "calculate":        lambda a: calculate(a.get("expr", "")),
        "get_weather":      lambda a: get_weather(a.get("city", "")),
    }
    # 空 1：从 table 里按 name 取函数，取不到要拿到 None（不是报错）
    # 提示：字典有个方法叫 get，找不到时返回 None
    fn = table.get(name, None)
    if fn is None:
        return f"未知工具: {name}"
    return fn(args)


def trim_messages(messages, keep):
    # 空 2：长度不够就原样返回，别瞎裁（keep+1 是因为第 0 条要留）
    if len(messages) <= keep + 1:
        return messages
    # 空 3：第 0 条 + 最后 keep 条，拼成一个新列表
    # 提示：messages[-keep:] 就是最后 keep 条；两个列表可以 + 起来
    return[messages[0]] + messages[-keep:]


def safe_args(raw):
    try:
        # 空 4：把 raw 这个字符串解析成 Python 对象
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
