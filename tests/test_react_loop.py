"""手写 ReAct 循环测试；模型响应全部由 mock 构造。"""

from unittest.mock import patch

import react_loop


def _response(content=None, finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"finish_reason": finish, "message": message}]}


def _calculate_call(call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "calculate", "arguments": '{"expr": "1+1"}'},
    }


def test_normal_finish() -> None:
    with patch.object(react_loop, "call_llm", return_value=_response("完成")):
        assert react_loop.run("你好") == "完成"


def test_length_finish_is_not_returned_as_complete_answer() -> None:
    response = _response("只写了半句", finish="length")
    with patch.object(react_loop, "call_llm", return_value=response):
        assert react_loop.run("写长文").startswith("【截断】")


def test_tool_result_is_appended_before_next_model_call() -> None:
    responses = [_response(tool_calls=[_calculate_call()]), _response("结果是 2")]
    captured = []

    def fake_call(messages):
        captured.append(messages)
        return responses.pop(0)

    with patch.object(react_loop, "call_llm", side_effect=fake_call):
        assert react_loop.run("计算 1+1") == "结果是 2"

    assert captured[1][-2]["role"] == "assistant"
    assert captured[1][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "2",
    }


def test_max_steps_returns_fallback() -> None:
    response = _response(tool_calls=[_calculate_call()])
    with patch.object(react_loop, "call_llm", return_value=response):
        assert "已用满 1 步" in react_loop.run("持续调用", max_steps=1)


def test_tool_exception_is_returned_as_tool_message() -> None:
    unknown_call = {
        "id": "bad-1",
        "type": "function",
        "function": {"name": "unknown", "arguments": "{}"},
    }
    result = react_loop.run_tools([unknown_call])

    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "bad-1"
    assert "工具执行失败" in result[0]["content"]
    assert "未知工具" in result[0]["content"]


def test_handwritten_schema_contains_five_tools() -> None:
    names = {item["function"]["name"] for item in react_loop.TOOLS_SCHEMA}
    assert names == {"get_current_time", "calculate", "get_weather", "read_file", "list_dir"}
