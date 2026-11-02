"""启动本地 MCP Server，验证握手、工具发现、调用与 schema 转换。

运行：python mcp_client.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
SERVER_SCRIPT = BASE_DIR / "mcp_server.py"


def make_server_params():
    """使用当前 Python 环境启动 Server，避免子进程找不到 mcp 包。"""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(BASE_DIR),
        # 关键：Windows 子进程默认用 GBK 写 stderr，父进程按 UTF-8 读 -> 中文全乱码。
        # 强制子进程用 UTF-8，日志才看得懂。（stdout 是协议通道，本来就是 UTF-8）
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )


async def main():
    print("=" * 60)
    print("① 启动 Server 子进程并握手（initialize）")
    print("=" * 60)

    async with stdio_client(make_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print("服务端名称:", info.server_info.name)
            print("服务端版本:", info.server_info.version)
            print("协议版本  :", info.protocol_version)
            print("服务端能力:", info.capabilities.model_dump(exclude_none=True))

            print()
            print("=" * 60)
            print("② 拉工具清单（list_tools）—— 这些名字我一行都没写在客户端里")
            print("=" * 60)
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"\n· {t.name}")
                print(f"  描述  : {t.description}")
                print(f"  入参  : {json.dumps(t.input_schema, ensure_ascii=False)}")

            print()
            print("=" * 60)
            print("③ 逐个调用（call_tool）—— 每次都是一次跨进程 JSON-RPC")
            print("=" * 60)
            cases = [
                ("get_current_time", {}),
                ("calculate", {"expr": "25*4+10"}),
                ("get_weather", {"city": "西安"}),
                ("read_file", {"path": "not_exist.txt"}),  # 故意失败：看错误怎么回来
                ("calculate", {"expr": "__import__('os')"}),  # 故意非法：看白名单拦不拦
            ]
            for name, args in cases:
                result = await session.call_tool(name, args)
                text = " | ".join(getattr(c, "text", str(c)) for c in (result.content or []))
                flag = "ERROR" if getattr(result, "is_error", False) else "OK   "
                print(f"[{flag}] {name}({json.dumps(args, ensure_ascii=False)})")
                print(f"        -> {text}")

            print()
            print("=" * 60)
            print("④ 关键动作：把 MCP 的 inputSchema 翻译成 OpenAI 的 tools 格式")
            print("=" * 60)
            openai_tools = to_openai_tools(tools.tools)
            print(json.dumps(openai_tools, ensure_ascii=False, indent=2)[:1200])
            print("\n↑ 这段就是昨天的 TOOLS_SCHEMA，只不过现在是自动生成的")


def to_openai_tools(mcp_tools):
    """将 MCP 工具描述转换为 OpenAI function calling schema。

    MCP 负责工具发现与调用，function calling 负责模型表达调用意图。
    """
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


if __name__ == "__main__":
    asyncio.run(main())
