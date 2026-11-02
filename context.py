"""按完整回合裁剪 Agent 消息，避免破坏工具调用的父子关系。"""

from typing import Any

Message = dict[str, Any]


def group_by_turn(messages: list[Message]) -> list[list[Message]]:
    """将 assistant 与其后续连续 tool 消息归为一组，保持原始顺序。"""
    groups: list[list[Message]] = []
    for message in messages:
        if message.get("role") == "tool":
            if not groups:
                raise ValueError("tool 消息不能出现在消息列表开头")
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


def take_last(groups: list[list[Message]], keep: int) -> list[Message]:
    """从最近一组开始保留消息；单组可能使结果超过 keep 条。"""
    if keep < 1:
        raise ValueError("keep 必须大于等于 1")

    count = 0
    picked: list[Message] = []
    for group in reversed(groups):
        if count >= keep:
            break
        picked = group + picked
        count += len(group)
    return picked


def trim_by_turn(messages: list[Message], keep: int) -> list[Message]:
    """保留首条 system 和最近若干完整回合，不修改传入列表。

    与固定条数切片不同，本函数不会把 assistant(tool_calls) 与其 tool
    结果拆开。预算按消息条数计算，不是 token 数。
    """
    if not messages:
        return []
    if messages[0].get("role") != "system":
        raise ValueError("首条消息必须是 system")
    if len(messages) <= keep + 1:
        return list(messages)
    return [messages[0], *take_last(group_by_turn(messages[1:]), keep)]
