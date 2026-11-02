"""将五个本地工具暴露为 MCP Server（stdio 传输）。

本模块不包含模型调用逻辑，工具实现与 Agent 通过 MCP 协议解耦。
直接运行会等待协议输入；功能验证请执行 python mcp_client.py。
"""

import functools
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import tools_core as T

# stdio 模式下 stdout 是协议通道，任何 print 都会污染协议直接崩。
# 所以日志一律走 stderr。
print(f"[mcp_server] 启动中，工作目录 {Path(__file__).parent}", file=sys.stderr)

mcp = MCPServer(
    name="agent-lab-tools",
    version="0.1.0",
    instructions="本地工具集：时间、四则运算、天气（演示数据）、读文件、列目录。所有返回值都是字符串。",
)


def safe(fn):
    """将工具异常转为 ToolError，使客户端收到可回灌的错误原因。"""

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


# ==================== 5 个工具：函数名 = 工具名，docstring = 描述 ====================
# 注意：MCP 的 inputSchema 是**从函数签名和类型注解自动生成**的，
# 不像昨天那样手写 300 行 JSON。这是框架最直接的收益。


@mcp.tool()
@safe
def get_current_time() -> str:
    """获取当前日期和时间，返回格式 'YYYY-MM-DD HH:MM:SS'。用户问'现在几点''今天几号'时使用。"""
    return T.get_current_time()


@mcp.tool()
@safe
def calculate(expr: str) -> str:
    """计算数学表达式并返回结果字符串。只支持数字和 + - * / ( ) . % 运算符，例如 '25*4+10'。"""
    return T.calculate(expr)


@mcp.tool()
@safe
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。city 用中文城市名，例如 '西安'。返回如 '晴 26℃'。"""
    return T.get_weather(city)


@mcp.tool()
@safe
def read_file(path: str) -> str:
    """读取本地文本文件内容，最多返回 2000 字符。path 为相对 agent-lab 目录的路径，例如 'README.md'。"""
    return T.read_file(path)


@mcp.tool()
@safe
def list_dir(path: str = ".") -> str:
    """列出指定目录下的文件名，用「、」连接成一个字符串返回。

    path 为相对 agent-lab 目录的路径，默认当前目录。
    当用户想知道某个目录里有哪些文件，或需要确认某个文件是否存在时使用。
    """
    return T.list_dir(path)


if __name__ == "__main__":
    # transport="stdio"：客户端启动这个进程，通过 stdin/stdout 收发 JSON-RPC
    # 另两种是 "sse" / "streamable-http"，用于远程部署
    mcp.run(transport="stdio")
