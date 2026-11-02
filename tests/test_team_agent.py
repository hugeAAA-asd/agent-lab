"""Supervisor 编排：角色解析、工具越界拦截、上限收敛与账本隔离测试。"""

from unittest.mock import patch

import team_agent


def _reply(content: str | None, tool_calls: list | None = None) -> dict:
    message: dict = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def test_parse_route_english() -> None:
    assert team_agent.parse_route("researcher:查北京天气") == ("researcher", "查北京天气")


def test_parse_route_accepts_markdown_and_numbering() -> None:
    """模型爱加粗、爱编号，这两种写法都不该让解析失败。"""
    assert team_agent.parse_route("**researcher**:查北京天气")[0] == "researcher"
    assert team_agent.parse_route("1. calculator: 算 18+10")[0] == "calculator"
    assert team_agent.parse_route("`writer`：整理")[0] == "writer"


def test_parse_route_chinese_alias() -> None:
    """角色名被写成中文时按别名认，而不是当成未知角色丢掉这一单。"""
    assert team_agent.parse_route("研究员：查北京天气")[0] == "researcher"
    assert team_agent.parse_route("计算员:算一下")[0] == "calculator"


def test_parse_route_done_maps_to_writer() -> None:
    """DONE / FINISH 不是角色，但语义就是收尾，映射给 writer 而不是报错。"""
    assert team_agent.parse_route("DONE")[0] == "writer"
    assert team_agent.parse_route("finish")[0] == "writer"


def test_parse_route_garbage_converges_to_writer() -> None:
    """完全不按格式写：收敛到能出结果的节点，比抛异常好。"""
    assert team_agent.parse_route("我觉得应该先查天气再计算")[0] == "writer"
    assert team_agent.parse_route("")[0] == "writer"


def test_parse_route_name_only() -> None:
    name, instruction = team_agent.parse_route("calculator")
    assert name == "calculator"
    assert instruction == team_agent.DEFAULT_WORKER_INSTRUCTION


def test_route_from_supervisor_maps_unknown_to_writer() -> None:
    """状态里出现不存在的节点名时，图不该 KeyError。"""
    assert team_agent.route_from_supervisor({"next_agent": "hacker"}) == "writer"
    assert team_agent.route_from_supervisor({"next_agent": "calculator"}) == "calculator"
    assert team_agent.route_from_supervisor({}) == "writer"


def test_supervisor_calls_llm_and_counts_round() -> None:
    with patch.object(team_agent, "call_llm", return_value=_reply("researcher:查天气")) as mock:
        out = team_agent.supervisor_node({"task": "t", "ledger": [], "rounds": 0, "max_rounds": 3})

    assert out["next_agent"] == "researcher"
    assert out["rounds"] == 1
    # 调度员不该看见工具，否则它会自己动手而不是派活
    assert mock.call_args.kwargs["with_tools"] is False


def test_supervisor_forces_writer_without_calling_llm_when_exhausted() -> None:
    """上限到了就别再问模型：它只会把同一件事重派一遍。"""
    with patch.object(team_agent, "call_llm") as mock:
        out = team_agent.supervisor_node({"task": "t", "ledger": [], "rounds": 3, "max_rounds": 3})

    mock.assert_not_called()
    assert out["next_agent"] == "writer"
    assert out["stop_reason"] == "max_rounds"


def test_run_scoped_tools_blocks_out_of_scope() -> None:
    """schema 里不给只是「大概率不调」，不是「调不到」，所以代码里要拦。"""
    calls = [{"id": "c1", "function": {"name": "get_weather", "arguments": '{"city":"北京"}'}}]

    out = team_agent.run_scoped_tools(calls, ["calculate"])

    assert len(out) == 1
    # 必须回一条 tool 消息配对，否则下一次请求 400
    assert out[0]["tool_call_id"] == "c1"
    assert "没有 get_weather" in out[0]["content"]
    assert "calculate" in out[0]["content"]


def test_run_scoped_tools_runs_allowed() -> None:
    calls = [{"id": "c1", "function": {"name": "calculate", "arguments": '{"expr":"18+10"}'}}]
    out = team_agent.run_scoped_tools(calls, ["calculate"])
    assert out[0]["content"] == "28"


def test_worker_passes_only_its_own_tools_to_model() -> None:
    """计算员看不见查天气——这是编排真正省下的东西。"""
    with patch.object(team_agent, "call_llm", return_value=_reply("算好了")) as mock:
        team_agent.calculator_node({"task": "t", "ledger": [], "instruction": "算 18+10"})

    tools = mock.call_args.kwargs["tools"]
    assert [item["function"]["name"] for item in tools] == ["calculate"]


def test_worker_without_tools_disables_tool_calling() -> None:
    """写作者没有任何工具，就不该把 tools 递过去。"""
    with patch.object(team_agent, "call_llm", return_value=_reply("最终答案")) as mock:
        team_agent.writer_node({"task": "t", "ledger": [], "instruction": "收尾"})

    assert mock.call_args.kwargs["with_tools"] is False


def test_worker_returns_conclusion_into_ledger() -> None:
    with patch.object(team_agent, "call_llm", return_value=_reply("北京 多云 18℃")):
        out = team_agent.researcher_node({"task": "t", "ledger": [], "instruction": "查北京天气"})

    entry = out["ledger"][0]
    assert entry["agent"] == "researcher"
    assert entry["instruction"] == "查北京天气"
    assert entry["result"] == "北京 多云 18℃"


def test_worker_exhaustion_returns_raw_material() -> None:
    """用尽步数时不再调模型：worker 的输出是材料，不是给用户的答案。"""
    call = {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city":"北京"}'}}
    with patch.object(team_agent, "call_llm", return_value=_reply(None, [call])) as mock:
        out = team_agent.researcher_node({"task": "t", "ledger": [], "instruction": "一直查"})

    assert mock.call_count == team_agent.WORKER_STEPS
    result = out["ledger"][0]["result"]
    assert "原始工具结果" in result
    assert "18℃" in result


def test_worker_sees_upstream_notes_but_not_message_stream() -> None:
    """账本里只有结论，没有别人的工具调用流水。"""
    system = team_agent._worker_system(
        "writer",
        "查天气再算温度",
        [{"agent": "researcher", "instruction": "查北京", "result": "18℃"}],
    )

    assert "18℃" in system
    assert "查天气再算温度" in system
    assert "tool_call_id" not in system
    assert "get_weather" not in system


def test_ledger_reducer_appends() -> None:
    assert team_agent._append([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]


def test_full_team_researcher_then_writer() -> None:
    """端到端：派单 → 干活 → 再派单 → 收尾，答案是 writer 最后那句。"""
    script = [
        _reply("researcher:查北京天气"),
        _reply("北京 多云 18℃"),
        _reply("writer:整理答案"),
        _reply("北京当前多云 18℃"),
    ]
    with patch.object(team_agent, "call_llm", side_effect=script):
        answer = team_agent.run_team("北京天气怎么样", max_rounds=4)

    assert answer == "北京当前多云 18℃"


def test_full_team_stops_at_max_rounds_and_writes() -> None:
    """上限 1 轮：派一单 → 收口仍要出答案，不能返回空。"""
    script = [
        _reply("researcher:查北京天气"),
        _reply("北京 多云 18℃"),
        _reply("北京当前多云 18℃"),  # 上限触发的 writer 不再经过 supervisor
    ]
    with patch.object(team_agent, "call_llm", side_effect=script) as mock:
        answer = team_agent.run_team("北京天气怎么样", max_rounds=1)

    assert answer == "北京当前多云 18℃"
    assert mock.call_count == 3


def test_recursion_limit_leaves_margin() -> None:
    for rounds in (1, 2, 4):
        assert team_agent.recursion_limit(rounds) >= rounds * 2 + 3
