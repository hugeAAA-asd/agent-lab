"""MCP schema 适配与错误回灌测试，不启动真实 Server。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import mcp_agent


def _tool_call(call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "calculate", "arguments": '{"expr": "1+1"}'},
    }


def test_to_openai_tools() -> None:
    tool = SimpleNamespace(
        name="calculate",
        description="计算",
        input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
    )
    converted = mcp_agent.to_openai_tools([tool])

    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "calculate"
    assert converted[0]["function"]["parameters"] == tool.input_schema


class _Session:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def call_tool(self, name, args):
        if self.error:
            raise self.error
        return self.result


def test_mcp_error_result_is_fed_back() -> None:
    result = SimpleNamespace(
        content=[SimpleNamespace(text="FileNotFoundError: missing")], is_error=True
    )
    messages = asyncio.run(mcp_agent.run_tools(_Session(result=result), [_tool_call()]))

    assert messages[0]["role"] == "tool"
    assert messages[0]["content"].startswith("工具执行失败")
    assert "missing" in messages[0]["content"]


def test_mcp_call_exception_is_fed_back() -> None:
    session = _Session(error=ConnectionError("server closed"))
    messages = asyncio.run(mcp_agent.run_tools(session, [_tool_call()]))

    assert "工具调用失败" in messages[0]["content"]
    assert "server closed" in messages[0]["content"]


def test_mcp_length_finish() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "半句话"},
            }
        ]
    }
    with patch.object(mcp_agent, "call_llm", return_value=response):
        answer = asyncio.run(mcp_agent.run(None, [], "写长文"))

    assert answer.startswith("【截断】")
