"""LangGraph 状态、路由、超限收敛与截断分支测试。"""

from unittest.mock import patch

from langgraph.graph import END

import graph_agent


def test_append_messages() -> None:
    assert graph_agent.append_messages([{"role": "user"}], [{"role": "assistant"}]) == [
        {"role": "user"},
        {"role": "assistant"},
    ]


def test_should_continue() -> None:
    assert graph_agent.should_continue({"messages": [{"tool_calls": [{}]}]}) == "tools"
    assert graph_agent.should_continue({"messages": [{"content": "done"}]}) == END


def test_should_continue_goes_to_finalize_when_exhausted() -> None:
    """轮次用尽但模型还想要工具 —— 不是结束，是去收尾。"""
    state = {"messages": [{"tool_calls": [{}]}], "steps": 2, "max_steps": 2}
    assert graph_agent.should_continue(state) == "finalize"


def test_agent_node_marks_length_finish() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "半句话"},
            }
        ]
    }
    state = {"messages": [{"role": "system", "content": "rules"}]}
    with patch.object(graph_agent, "call_llm", return_value=response):
        result = graph_agent.agent_node(state)

    assert result["messages"][0]["content"].startswith("【截断】")


def test_agent_node_marks_empty_answer() -> None:
    """空 content + 无 tool_calls 不是答案，不能原样交出去。"""
    response = {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]
    }
    with patch.object(graph_agent, "call_llm", return_value=response):
        result = graph_agent.agent_node({"messages": [{"role": "system", "content": "x"}]})

    assert result["messages"][0]["content"].startswith("【空回答】")


def test_agent_node_counts_steps() -> None:
    response = {"choices": [{"finish_reason": "stop", "message": {"content": "好"}}]}
    with patch.object(graph_agent, "call_llm", return_value=response):
        result = graph_agent.agent_node(
            {"messages": [{"role": "system", "content": "x"}], "steps": 2}
        )

    assert result["steps"] == 3


def test_tools_node_blocks_duplicate_call() -> None:
    """同工具同参数第二次出现：不执行，直接回提示，但仍要回 tool 消息配对。"""
    first = {
        "id": "c1",
        "function": {"name": "calculate", "arguments": '{"expr": "1+1"}'},
    }
    dup = {"id": "c2", "function": {"name": "calculate", "arguments": '{"expr":"1+1"}'}}
    state = {
        "messages": [
            {"role": "user", "content": "算一下"},
            {"role": "assistant", "content": None, "tool_calls": [first]},
            {"role": "tool", "tool_call_id": "c1", "content": "1+1 = 2"},
            {"role": "assistant", "content": None, "tool_calls": [dup]},
        ]
    }

    result = graph_agent.tools_node(state)

    assert len(result["messages"]) == 1
    assert result["messages"][0]["tool_call_id"] == "c2"
    assert "前面已经执行过" in result["messages"][0]["content"]


def test_finalize_node_answers_from_existing_messages() -> None:
    """超限后要尽量作答，而不是丢掉已经拿到的材料。"""
    response = {"choices": [{"message": {"content": "根据已查到的天气，应该是 32 度"}}]}
    state = {
        "messages": [{"role": "user", "content": "查天气再算温度"}],
        "steps": 2,
        "max_steps": 2,
    }
    with patch.object(graph_agent, "call_llm", return_value=response) as mock:
        result = graph_agent.finalize_node(state)

    assert result["stop_reason"] == "max_steps"
    content = result["messages"][0]["content"]
    assert "已达到 2 轮" in content
    assert "32 度" in content
    # 收尾请求必须关掉工具，否则模型又发起一轮调用
    assert mock.call_args.kwargs["with_tools"] is False


def test_finalize_node_survives_failure() -> None:
    state = {"messages": [{"role": "user", "content": "q"}], "steps": 2, "max_steps": 2}
    with patch.object(graph_agent, "call_llm", side_effect=RuntimeError("网络断了")):
        result = graph_agent.finalize_node(state)

    assert result["stop_reason"] == "error"
    assert "收尾请求失败" in result["messages"][0]["content"]


def test_run_graph_converts_steps_to_recursion_limit() -> None:
    result = {"messages": [{"role": "assistant", "content": "完成"}], "steps": 1}
    with patch.object(graph_agent.APP, "invoke", return_value=result) as invoke:
        assert graph_agent.run_graph("你好", max_steps=3) == "完成"

    assert invoke.call_args.args[1] == {"recursion_limit": graph_agent.recursion_limit(3)}
    assert graph_agent.recursion_limit(3) == 11
    # 初始状态要把业务上限一起带进图里，否则路由没法判断「是否用尽」
    assert invoke.call_args.args[0]["max_steps"] == 3
    assert invoke.call_args.args[0]["steps"] == 0


def test_recursion_limit_leaves_margin() -> None:
    """一轮两超步 + finalize + 边界；余量不足会在正常流程里误伤。"""
    for steps in (1, 3, 6):
        assert graph_agent.recursion_limit(steps) >= steps * 2 + 2


def test_finalize_runs_pending_tool_before_answering() -> None:
    """轮次掐断时工具还没执行：要先补跑，材料到手再收尾。

    这条对应真实故障：max_steps=1 时直接收尾会 400（assistant 的
    tool_calls 没有配对的 tool 消息），补跑之后才拿得到东西可答。
    """
    pending = {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city":"北京"}'}}
    state = {
        "messages": [
            {"role": "user", "content": "查北京天气再 +10"},
            {"role": "assistant", "content": None, "tool_calls": [pending]},
        ],
        "steps": 1,
        "max_steps": 1,
    }
    tool_msg = {"role": "tool", "tool_call_id": "c1", "content": "北京 晴 26℃"}
    response = {"choices": [{"message": {"content": "北京 26℃ + 10 = 36"}}]}

    with (
        patch.object(graph_agent, "run_tools", return_value=[tool_msg]) as run,
        patch.object(graph_agent, "call_llm", return_value=response) as llm,
    ):
        result = graph_agent.finalize_node(state)

    run.assert_called_once()
    sent = llm.call_args.args[0]
    assert any(m.get("role") == "tool" for m in sent)
    assert "36" in result["messages"][0]["content"]
    assert result["stop_reason"] == "max_steps"


def test_drop_unanswered_tool_calls() -> None:
    """没有 tool 回应的 assistant(tool_calls) 必须清掉，否则请求 400。"""
    unanswered = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "x", "arguments": "{}"}}],
    }
    messages = [{"role": "user", "content": "q"}, unanswered]
    assert graph_agent._drop_unanswered_tool_calls(messages) == [messages[0]]


def test_drop_keeps_answered_tool_calls() -> None:
    """已经配上 tool 回应的，原样保留。"""
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "结果"},
    ]
    assert graph_agent._drop_unanswered_tool_calls(messages) == messages
