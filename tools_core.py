"""tools_core.py —— 4 个工具的纯实现（不含任何 LLM / 协议代码）

为什么单独抽出来：
  昨天的 react_loop.py 里，工具实现、OpenAI schema、循环逻辑揉在一个文件。
  今天要把它做成 MCP Server，Server 只该知道「工具怎么算」，
  不该知道 DeepSeek 的 key、不该知道 chat/completions 接口长什么样。
  所以先把工具层单独剥出来——这叫「解耦」，面试问架构时可以直接讲。
"""

import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent


def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


ALLOWED_CHARS = set("0123456789+-*/().% ")


def calculate(expr: str) -> str:
    """计算数学表达式。只允许数字和四则运算符（白名单，不用裸 eval）"""
    if not expr:
        raise ValueError("expr 不能为空")
    if not set(expr) <= ALLOWED_CHARS:
        raise ValueError(f"表达式含非法字符，只支持 + - * / ( ) . %：{expr}")
    return str(eval(expr, {"__builtins__": {}}, {}))


def get_weather(city: str) -> str:
    """查询城市天气（演示数据，真实项目替换为和风天气 API）"""
    if not city:
        raise ValueError("city 不能为空")
    fake = {"西安": "晴 26℃", "北京": "多云 18℃", "上海": "小雨 22℃"}
    return fake.get(city, f"{city}: 暂无数据")


def read_file(path: str) -> str:
    """读取本地文本文件，最多 2000 字符。文件不存在时抛 FileNotFoundError"""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    return p.read_text(encoding="utf-8", errors="replace")[:2000]


def list_dir(path: str = "."):
    """读目录,目录内的文件名是不是用、链接,目录不存在给FileNotFoundError"""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"目录不存在: {p}")
    return "、".join([f.name for f in p.iterdir()])


def dispatch(name: str, args: dict) -> str:
    """字典分发：名字 -> 函数。和昨天 warmup 里写的是同一个东西"""
    table = {
        "get_current_time": lambda a: get_current_time(),
        "calculate":        lambda a: calculate(a.get("expr", "")),
        "get_weather":      lambda a: get_weather(a.get("city", "")),
        "read_file":        lambda a: read_file(a.get("path", "")),
    }
    fn = table.get(name)
    if fn is None:
        raise ValueError(f"未知工具: {name}")
    return fn(args)


def safe_args(raw) -> dict:
    """把模型给的 arguments 字符串安全解析成 dict，任何异常都返回 {}"""
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


if __name__ == "__main__":
    # 自测：不依赖网络和 API key，纯本地跑
    print("time    ->", get_current_time())
    print("calc    ->", calculate("25*4+10"))
    print("weather ->", get_weather("西安"))
    print("dispatch->", dispatch("calculate", {"expr": "2*10"}))
    print("safe    ->", safe_args('{"expr": "1+1"}'), safe_args("坏JSON"))
    try:
        read_file("not_exist.txt")
    except FileNotFoundError as e:
        print("read    -> 预期异常:", e)
