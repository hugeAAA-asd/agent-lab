"""通过 MCP 动态发现和调用工具的 ReAct Agent。"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from context import trim_by_turn
from tools_core import BASE_DIR, safe_args

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
MAX_STEPS = 6
KEEP_MSG = 12

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """你是一个会使用工具的助手。规则：
1. 需要实时信息或计算时必须调用工具，不要凭空编造。
2. 可以调用多个互不依赖的工具，当前实现会串行执行并返回结果。
3. 工具返回失败时，先读懂错误原因，换一种做法或向用户说明，不要原样重试。
4. 任务完成后直接用中文回答，不要再调用工具。"""


def call_llm(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """使用运行时发现的工具 schema 请求模型。"""
    if not API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 后填写")
    response = requests.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "tools": tools, "temperature": 0},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """将 MCP Tool 的输入 schema 转换为 OpenAI function calling 格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in mcp_tools
    ]


async def run_tools(session: Any, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨进程执行 MCP 工具，并把协议错误或调用异常回灌给模型。"""
    output_messages = []
    for tool_call in tool_calls:
        name = tool_call["function"]["name"]
        args = safe_args(tool_call["function"].get("arguments"))
        print(f"  → 调用 {name}({json.dumps(args, ensure_ascii=False)})")
        try:
            result = await session.call_tool(name, args)
            text = " | ".join(getattr(item, "text", str(item)) for item in (result.content or []))
            if getattr(result, "is_error", False):
                text = f"工具执行失败: {text}"
            print(f"  ← {text[:120]}")
        except Exception as exc:
            logger.warning("MCP 工具 %s 调用失败: %s", name, exc)
            text = f"工具调用失败: {type(exc).__name__}: {exc}"
        output_messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": text})
    return output_messages


async def run(
    session: Any,
    tools_schema: list[dict[str, Any]],
    user_input: str,
    max_steps: int = MAX_STEPS,
) -> str:
    """运行 MCP 版 ReAct 循环，处理轮数上限和输出截断。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for step in range(1, max_steps + 1):
        messages = trim_by_turn(messages, KEEP_MSG)
        data = call_llm(messages, tools_schema)
        choice = data["choices"][0]
        message = dict(choice["message"])
        message.pop("reasoning_content", None)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print(f"[step {step}] finish_reason = {choice.get('finish_reason')}")
            if choice.get("finish_reason") == "length":
                return "【截断】回答超出长度上限，只生成了一部分。请把问题拆小一点。"
            return message.get("content") or "(空)"

        print(f"[step {step}] 模型要求调用 {len(tool_calls)} 个工具")
        messages.append(message)
        messages.extend(await run_tools(session, tool_calls))

    return f"【兜底】已用满 {max_steps} 步仍未得到最终答案。"


SERVERS = {
    "local": lambda: StdioServerParameters(
        command=sys.executable,
        args=[str(BASE_DIR / "mcp_server.py")],
        cwd=str(BASE_DIR),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    ),
    "fs": lambda: StdioServerParameters(
        command="npx.cmd" if os.name == "nt" else "npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(BASE_DIR)],
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    ),
}

CASES = {
    "local": [
        ("用例1 多步", "查一下西安天气，然后算一下 25*4+10", MAX_STEPS),
        ("用例2 工具失败回灌", "读一下 not_exist.txt，告诉我里面写了什么", MAX_STEPS),
        ("用例3 单步", "现在几点了？", MAX_STEPS),
        ("用例4 轮数上限", "查一下北京天气，然后把温度和 10 相加", 1),
    ],
    "fs": [
        ("用例1 多步", "先看看当前目录有哪些文件，再读一下 tools_core.py 的开头", MAX_STEPS),
        ("用例2 工具失败回灌", "读一下 not_exist.txt，告诉我里面写了什么", MAX_STEPS),
        ("用例3 单步", "看看当前目录下有哪些文件和文件夹", MAX_STEPS),
        ("用例4 轮数上限", "列一下当前目录的文件，然后找出最大的那个", 1),
    ],
}


async def main() -> None:
    """连接指定 MCP Server 并运行人工演示用例。"""
    which = sys.argv[1] if len(sys.argv) > 1 else "local"
    if which not in SERVERS:
        raise SystemExit(f"用法：python mcp_agent.py [local|fs]，收到的是 {which!r}")

    async with stdio_client(SERVERS[which]()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools_schema = to_openai_tools(mcp_tools)
            print(f"Server = {which}")
            print(f"已从 MCP Server 拉到 {len(mcp_tools)} 个工具：")
            print("  " + "、".join(tool.name for tool in mcp_tools) + "\n")

            for title, question, limit in CASES[which]:
                print("=" * 58)
                print(f"{title}：{question}")
                print("=" * 58)
                print("答案:", await run(session, tools_schema, question, max_steps=limit))
                print()


if __name__ == "__main__":
    asyncio.run(main())
