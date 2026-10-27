"""mcp_client.py —— MCP 客户端：拉起 Server 进程，列工具，逐个调用

这一步是整个 MCP 学习里最关键的一步，因为它让你亲眼看到：
  · 工具列表**不是写死在 Agent 代码里的**，是运行时从 Server 拉回来的
  · 调用工具**不是本地函数调用**，是发一条 JSON-RPC 过去、等结果回来
  · 而且 Server 是**另一个进程**（这里是子进程，远程就是一个 URL）

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
    """告诉客户端：怎么启动这个 Server。
    注意 command 用的是当前 venv 的 python，不是系统 python——不然找不到 mcp 包。
    """
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
                ("read_file", {"path": "not_exist.txt"}),          # 故意失败：看错误怎么回来
                ("calculate", {"expr": "__import__('os')"}),        # 故意非法：看白名单拦不拦
            ]
            for name, args in cases:
                result = await session.call_tool(name, args)
                text = " | ".join(
                    getattr(c, "text", str(c)) for c in (result.content or [])
                )
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
    """MCP Tool -> OpenAI function calling schema

    两边结构其实几乎一样，差别只有外层包了一层：
        MCP:    {name, description, inputSchema}
        OpenAI: {"type": "function", "function": {name, description, parameters}}
    所以这个函数就是「改个壳」。面试问「MCP 和 function calling 什么关系」，
    答案就在这 8 行里：**MCP 管工具怎么被发现和调用，function calling 管模型怎么表达
    「我要调它」。两者是上下层，不是竞品。**
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
