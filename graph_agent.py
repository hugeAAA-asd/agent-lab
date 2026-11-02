"""用 LangGraph 状态图实现与 react_loop.py 相同的 ReAct 闭环。"""

import sys
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from context import trim_by_turn
from react_loop import KEEP_MSG, MAX_STEPS, SYSTEM_PROMPT, call_llm, run_tools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def append_messages(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """LangGraph reducer：将节点产生的新消息追加到共享状态。"""
    return left + right


class AgentState(TypedDict):
    """图的共享状态。"""

    messages: Annotated[list[dict[str, Any]], append_messages]


def agent_node(state: AgentState) -> dict[str, list[dict[str, Any]]]:
    """调用模型，并返回需要追加的一条 assistant 消息。"""
    messages = trim_by_turn(state["messages"], KEEP_MSG)
    data = call_llm(messages)
    choice = data["choices"][0]
    message = dict(choice["message"])
    message.pop("reasoning_content", None)

    tool_calls = message.get("tool_calls") or []
    print(f"  [agent] content={repr(message.get('content'))[:20]} tool_calls={len(tool_calls)}")
    if not tool_calls and choice.get("finish_reason") == "length":
        message["content"] = "【截断】回答超出长度上限，只生成了一部分。请把问题拆小一点。"
    return {"messages": [message]}


def tools_node(state: AgentState) -> dict[str, list[dict[str, Any]]]:
    """执行最后一条 assistant 消息中声明的工具调用。"""
    last = state["messages"][-1]
    print(f"  [tools] 执行 {len(last['tool_calls'])} 个工具")
    return {"messages": run_tools(last["tool_calls"])}


def should_continue(state: AgentState) -> str:
    """有工具调用时进入 tools，否则结束图执行。"""
    return "tools" if state["messages"][-1].get("tool_calls") else END


def build_graph():
    """构建 agent → tools → agent 的状态图。"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


APP = build_graph()


def run_graph(user_input: str, max_steps: int = MAX_STEPS) -> str:
    """运行状态图；recursion_limit 按节点数换算，而不是直接等同业务轮数。"""
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    try:
        result = APP.invoke(
            {"messages": initial_messages},
            {"recursion_limit": max_steps * 2 + 1},
        )
        return result["messages"][-1].get("content") or "(空)"
    except GraphRecursionError:
        return f"【兜底】达到 max_steps={max_steps} 上限，LangGraph 已终止图执行"


def main() -> None:
    """运行与手写版一致的人工演示用例。"""
    cases = [
        ("用例1 多步", "查一下西安天气，然后算一下 25*4+10", MAX_STEPS),
        ("用例2 错误回灌", "读一下 not_exist.txt，告诉我里面写了什么", MAX_STEPS),
        ("用例3 单步", "现在几点了？", MAX_STEPS),
        ("用例4 轮数上限", "查一下北京天气，然后把温度和 10 相加", 1),
    ]
    for title, question, limit in cases:
        print("\n" + "=" * 55)
        print(f"{title}：{question}")
        print("=" * 55)
        print("答案:", run_graph(question, max_steps=limit))


if __name__ == "__main__":
    main()
