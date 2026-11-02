"""上下文裁剪的不变量测试。"""

import pytest

from context import group_by_turn, trim_by_turn


def _history() -> list[dict]:
    return [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]


def test_trim_keeps_system_and_complete_latest_turn() -> None:
    history = _history()
    trimmed = trim_by_turn(history, keep=3)

    assert trimmed[0]["role"] == "system"
    assert [message["role"] for message in trimmed[1:]] == ["user", "assistant", "tool"]
    assert trimmed[-1]["tool_call_id"] == trimmed[-2]["tool_calls"][0]["id"]
    assert len(history) == 6  # 不修改输入列表


def test_trim_may_exceed_keep_to_preserve_tool_pair() -> None:
    trimmed = trim_by_turn(_history(), keep=1)

    assert [message["role"] for message in trimmed] == ["system", "assistant", "tool"]


def test_short_history_returns_a_copy() -> None:
    history = [{"role": "system", "content": "rules"}]
    trimmed = trim_by_turn(history, keep=12)

    assert trimmed == history
    assert trimmed is not history


def test_orphan_tool_message_is_rejected() -> None:
    with pytest.raises(ValueError, match="tool 消息"):
        group_by_turn([{"role": "tool", "content": "orphan"}])


def test_first_message_must_be_system() -> None:
    with pytest.raises(ValueError, match="system"):
        trim_by_turn([{"role": "user", "content": "hi"}], keep=3)
