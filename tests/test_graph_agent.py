"""LangGraph 状态、路由和截断分支测试。"""

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


def test_run_graph_converts_steps_to_recursion_limit() -> None:
    result = {"messages": [{"role": "assistant", "content": "完成"}]}
    with patch.object(graph_agent.APP, "invoke", return_value=result) as invoke:
        assert graph_agent.run_graph("你好", max_steps=3) == "完成"

    assert invoke.call_args.args[1] == {"recursion_limit": 7}
