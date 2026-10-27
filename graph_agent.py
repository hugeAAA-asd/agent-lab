"""graph_agent.py —— LangGraph 版 ReAct 循环（和 react_loop.py 做同一件事）

和 py_project01/main.py 的三个区别：
  1. 有真正的循环：工具结果回灌后继续调模型，直到模型不再要工具
  2. 有 MAX_STEPS：防止死循环烧钱
  3. 工具异常会被捕获、作为 tool 消息回灌——Agent 不会崩

运行：python react_loop.py
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("错误：.env 里没有 DEEPSEEK_API_KEY（先从 py_project01 复制一份）")

URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "deepseek-flash"

MAX_STEPS = 6     # 最多几轮「模型 → 工具 → 模型」
KEEP_MSG = 12     # 历史消息保留条数（不含 system）


# ==================== 三块砖：就是你 warmup 里写过的那三个 ====================
def dispatch(name, args):
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


def trim_messages(messages, keep):
    if len(messages) <= keep + 1:
        return messages
    # 隐患（面试可讲）：简单按条数裁剪，会把 tool 消息和它的
    # assistant(tool_calls) 父消息拆散，API 直接 400。
    # 正确做法是按「assistant + 它所有 tool」成组裁剪。
    return [messages[0]] + messages[-keep:]


def safe_args(raw):
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(args, dict):
        return {}
    return args


# ==================== 4 个工具的真实实现 ====================
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


ALLOWED_CHARS = set("0123456789+-*/().% ")


def calculate(expr: str):
    """eval 白名单版：只允许数字和四则运算符（原项目直接 eval，可执行任意代码）"""
    if not expr:
        raise ValueError("expr 不能为空")
    if not set(expr) <= ALLOWED_CHARS:
        raise ValueError(f"表达式含非法字符，只支持 + - * / ( ) . %：{expr}")
    return str(eval(expr, {"__builtins__": {}}, {}))


def get_weather(city: str):
    """假数据版，真实项目换成和风天气 API。接口稳定才能专心练循环。"""
    if not city:
        raise ValueError("city 不能为空")
    fake = {"西安": "晴 26℃", "北京": "多云 18℃", "上海": "小雨 22℃"}
    return fake.get(city, f"{city}: 暂无数据")


def read_file(path: str):
    """真读本地文件——文件不存在时会抛异常，正好用来测错误回灌"""
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    return p.read_text(encoding="utf-8", errors="replace")[:2000]


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，例如 25*4+10",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "数学表达式"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 西安"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取当前目录下的文本文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件名或路径"}},
                "required": ["path"],
            },
        },
    },
]


# ==================== 调模型 ====================
SYSTEM_PROMPT = """你是一个会使用工具的助手。
规则：
1. 需要实时信息或计算时必须调用工具，禁止凭记忆编造。
2. 一次只推进一步，拿到工具结果后再决定下一步。
3. 工具返回错误时，分析原因、换参数或换方式，不要重复同一个调用。
4. 信息齐了再给最终答案，最终答案要简短，不要复述中间过程。"""


def call_llm(messages):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",
    }
    r = requests.post(URL, json=body, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


# ==================== 执行一批工具，返回要回灌的消息 ====================
def run_tools(tool_calls):
    results = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = safe_args(tc["function"]["arguments"])
        print(f"  → 调用 {name}({args})")
        try:
            out = str(dispatch(name, args))
        except Exception as e:
            # 关键：异常不往外抛，转成文本回灌给模型
            out = f"工具执行失败: {type(e).__name__}: {e}"
            print(f"  ! 异常已捕获，回灌给模型")
        print(f"  ← {out[:120]}")
        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": out,
        })
    return results


# ==================== 主循环：这就是 ReAct ====================


# ==================== 上面全部和 react_loop.py 一模一样 ====================
# 从这里开始才是 LangGraph。注意：4 个工具、call_llm、run_tools 一行都没改。

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError


def append_messages(left, right):
    """reducer：节点返回的新消息，怎么合并进 state。这里是简单追加。"""
    return left + right


class AgentState(TypedDict):
    """图的「共享内存」：只有一个字段，就是消息列表"""
    messages: Annotated[list, append_messages]


def agent_node(state):
    """节点 A：调模型。输入 state，输出要追加的消息。"""
    messages = trim_messages(state["messages"], KEEP_MSG)
    data = call_llm(messages)
    msg = data["choices"][0]["message"]
    msg.pop("reasoning_content", None)
    n = len(msg.get("tool_calls") or [])
    print(f"  [agent] 返回：content={repr(msg.get('content'))[:20]} tool_calls={n}")
    return {"messages": [msg]}


def tools_node(state):
    """节点 B：执行工具。"""
    last = state["messages"][-1]
    print(f"  [tools] 执行 {len(last['tool_calls'])} 个工具")
    return {"messages": run_tools(last["tool_calls"])}


def should_continue(state):
    """条件边：决定 agent 之后去哪。这就是你手写的那个 if。"""
    last = state["messages"][-1]
    if last.get("tool_calls"):
        return "tools"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")          # 工具跑完回 agent，这就是循环

    return g.compile()


APP = build_graph()


def run_graph(user_input, max_steps=MAX_STEPS):
    """对外接口：和 react_loop.run() 的签名、返回值保持一致"""
    init = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    try:
        result = APP.invoke(
            {"messages": init},
            {"recursion_limit": max_steps * 2 + 1},
        )
        return result["messages"][-1].get("content") or "(空)"
    except GraphRecursionError:
        return f"【兜底】达到 max_steps={max_steps} 上限，LangGraph 帮你拦住了"


if __name__ == "__main__":
    cases = [
        ("用例1 多步", "查一下西安天气，然后算一下 25*4+10"),
        ("用例2 错误回灌", "读一下 not_exist.txt 这个文件，告诉我里面写了什么"),
        ("用例3 单步", "现在几点了？"),
    ]
    for title, q in cases:
        print("\n" + "=" * 55)
        print(f"{title}：{q}")
        print("=" * 55)
        print("答案:", run_graph(q))

    print("\n" + "=" * 55)
    print("用例4 防死循环：max_steps=1")
    print("=" * 55)
    print("答案:", run_graph("查一下北京天气，然后把温度和 10 相加", max_steps=1))
