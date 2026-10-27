"""mcp_agent.py —— Agent 的工具**不再写死在代码里**，运行时从 MCP Server 拉

和昨天 react_loop.py 的三个差别：
  1. 工具清单：昨天是硬编码的 TOOLS_SCHEMA，今天是 list_tools() 拉回来的
  2. 工具执行：昨天是本地 dispatch()，今天是 call_tool()（跨进程 JSON-RPC）
  3. 加新工具：昨天要改 4 个地方（实现/schema/dispatch/提词），今天只要在 Server 里加个函数

这一条就是 MCP 存在的理由。面试被问「为什么要用 MCP 而不是自己写 function calling」，
答案就在上面第 3 点：工具从「代码的一部分」变成了「可插拔的外部服务」。

运行：python mcp_agent.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("错误：.env 里没有 DEEPSEEK_API_KEY")

URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "deepseek-flash"

MAX_STEPS = 6
KEEP_MSG = 12

SYSTEM_PROMPT = (
    "你是一个会使用工具的助手。规则：\n"
    "1. 需要实时信息或计算时必须调用工具，不要凭空编造。\n"
    "2. 一次可以调用多个工具，它们会并行返回。\n"
    "3. 工具返回失败时，先读懂错误原因，换一种做法或向用户说明，不要原样重试。\n"
    "4. 任务完成后直接用中文回答，不要再调用工具。"
)


# ==================== 和昨天一样的部分：调模型 + 裁剪 + 安全解析 ====================
def call_llm(messages, tools):
    payload = {"model": MODEL, "messages": messages, "tools": tools, "temperature": 0}
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"接口错误 {r.status_code}: {r.text[:300]}")
    return r.json()


def trim_messages(messages, keep):
    if len(messages) <= keep + 1:
        return messages
    return [messages[0]] + messages[-keep:]


def safe_args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}


def to_openai_tools(mcp_tools):
    """MCP Tool -> OpenAI function calling schema（就是换个壳）"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            },
        }
        for t in mcp_tools
    ]


# ==================== 今天的核心：执行工具改成跨进程调用 ====================
async def run_tools(session, tool_calls):
    """和昨天 run_tools 的作用一样：返回一批 role=tool 的消息。

    差别只有中间那一行——昨天是 dispatch(name, args)，
    今天是 await session.call_tool(name, args)，一次网络/进程往返。
    """
    out = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = safe_args(tc["function"]["arguments"])
        print(f"  → 调用 {name}({json.dumps(args, ensure_ascii=False)})")
        try:
            result = await session.call_tool(name, args)
            text = " | ".join(getattr(c, "text", str(c)) for c in (result.content or []))
            if getattr(result, "is_error", False):
                # 关键：isError 也是「结果」，不是「程序异常」，照样回灌给模型
                print(f"  ! 工具返回错误，已回灌：{text[:80]}")
                text = f"工具执行失败: {text}"
            else:
                print(f"  ← {text[:80]}")
        except Exception as e:
            print(f"  ! 调用本身失败：{e}")
            text = f"工具调用失败: {type(e).__name__}: {e}"
        out.append({"role": "tool", "tool_call_id": tc["id"], "content": text})
    return out


async def run(session, tools_schema, user_input, max_steps=MAX_STEPS):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for step in range(1, max_steps + 1):
        messages = trim_messages(messages, KEEP_MSG)
        data = call_llm(messages, tools_schema)
        msg = data["choices"][0]["message"]
        msg.pop("reasoning_content", None)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            print(f"[step {step}] ✓ 模型给出最终答案")
            return msg.get("content") or "(空)"

        print(f"[step {step}] 模型要求调用 {len(tool_calls)} 个工具")
        messages.append(msg)
        messages.extend(await run_tools(session, tool_calls))

    return f"【兜底】已用满 {max_steps} 步仍未得到最终答案。"


SERVERS = {
    # 自己写的 Server：4 个本地工具
    "local": lambda: StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "mcp_server.py")],
        cwd=str(BASE_DIR),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    ),
    # 官方 filesystem Server（Node 版，代码不在我的仓库里）
    # Windows 注意：npx 是 .cmd 脚本，Python 的 CreateProcess 找不到它，
    # 必须写 "npx.cmd"，否则报 WinError 2「系统找不到指定的文件」。
    "fs": lambda: StdioServerParameters(
        command="npx.cmd",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(BASE_DIR)],
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    ),
}

# 两套 Server 能力不同，用例也得跟着换
CASES = {
    "local": [
        ("用例1 多步", "查一下西安天气，然后算一下 25*4+10", MAX_STEPS),
        ("用例2 工具失败回灌", "读一下 not_exist.txt 这个文件，告诉我里面写了什么", MAX_STEPS),
        ("用例3 单步", "现在几点了？", MAX_STEPS),
        ("用例4 防死循环", "查一下北京天气，然后把温度和 10 相加", 1),
    ],
    "fs": [
        ("用例1 多步", "先看看当前目录有哪些文件，再读一下 tools_core.py 的开头 100 个字符", MAX_STEPS),
        ("用例2 工具失败回灌", "读一下 not_exist.txt 这个文件，告诉我里面写了什么", MAX_STEPS),
        ("用例3 单步", "看看当前目录下有哪些文件和文件夹", MAX_STEPS),
        ("用例4 防死循环", "列一下当前目录的文件，然后找出最大的那个", 1),
    ],
}


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "local"
    if which not in SERVERS:
        raise SystemExit(f"用法：python mcp_agent.py [local|fs]   收到的是 {which!r}")

    server = SERVERS[which]()

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools_schema = to_openai_tools(mcp_tools)
            print(f"Server = {which}")
            print(f"已从 MCP Server 拉到 {len(mcp_tools)} 个工具：")
            print("  " + "、".join(t.name for t in mcp_tools) + "\n")

            for title, q, limit in CASES[which]:
                print("=" * 58)
                print(f"{title}：{q}")
                print("=" * 58)
                print("答案:", await run(session, tools_schema, q, max_steps=limit))
                print()


if __name__ == "__main__":
    asyncio.run(main())
