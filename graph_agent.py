"""用 LangGraph 状态图实现与 react_loop.py 相同的 ReAct 闭环。

三条链路的分工写在 README 里，这里只留一句对照：
    react_loop.run()   for 循环 + 三处 return，轮次是函数里的局部变量
    graph_agent        同一个循环被拆成节点，轮次必须放进共享状态

本版相对上一版重做了四件事，每件都对应一个真实失效：

① **轮次计数进状态，业务上限不再靠 GraphRecursionError 感知**
   上一版把 max_steps 换算成 recursion_limit，超限时抛异常，用户看到的是
   「LangGraph 已终止图执行」——这是崩溃，不是回答。而此刻手里的工具结果
   往往已经够答了。

② **超限走 finalize 节点降级作答**
   不带 tools 再问一次，逼模型就现有信息说话；真收敛不了才退回固定文案。
   手写版在这里是直接 return 兜底句——图里多一个节点几乎不要钱，就没必要
   把已经拿到的材料一起丢掉。

③ **同工具同参数的重复调用在 tools 节点拦下**
   SYSTEM_PROMPT 第 3 条早写了「不要重复同一个调用」，但那只是提示词，
   模型不听话就白烧 token。提示词归提示词，代码里拦才算数。

④ **空回答不再是 "(空)"**
   模型偶尔返回 content="" 且无 tool_calls，上一版直接把它当答案交出去。
"""

import json
import sys
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from context import trim_by_turn
from react_loop import (
    KEEP_MSG,
    MAX_STEPS,
    SYSTEM_PROMPT,
    call_llm,
    run_tools,
)
from tools_core import safe_args

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def append_messages(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """LangGraph reducer：将节点产生的新消息追加到共享状态。"""
    return left + right


class AgentState(TypedDict):
    """图的共享状态。

    steps 为什么不等于 len(messages)：
        消息流里混着 system / user / assistant / tool 四类，长度和
        「模型被调用了几次」没有对应关系。轮次必须单独计数，否则
        上限会随工具返回的条数漂移——一次并行调三个工具就吃掉三格。
    """

    messages: Annotated[list[dict[str, Any]], append_messages]
    steps: int
    max_steps: int
    stop_reason: str | None


def _signature(tool_call: dict[str, Any]) -> str:
    """工具调用签名：工具名 + 归一化后的参数。

    用 safe_args 把 JSON 字符串解析再按 key 排序，这样 model 写出
    `{"city":"西安"}` 和 `{"city": "西安"}`（空格差异）会被认成同一个调用。
    """
    fn = tool_call.get("function") or {}
    return "%s(%s)" % (
        fn.get("name"),
        json.dumps(safe_args(fn.get("arguments")), sort_keys=True, ensure_ascii=False),
    )


def agent_node(state: AgentState) -> dict[str, Any]:
    """调用模型，并返回需要追加的一条 assistant 消息。"""
    messages = trim_by_turn(state["messages"], KEEP_MSG)
    data = call_llm(messages)
    choice = data["choices"][0]
    message = dict(choice["message"])
    message.pop("reasoning_content", None)

    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    if not tool_calls:
        if choice.get("finish_reason") == "length":
            message["content"] = "【截断】回答超出长度上限，只生成了一部分。请把问题拆小一点。"
        elif not (content or "").strip():
            # 空 content + 无 tool_calls = 模型什么也没说就收手了。
            # 这不是答案，得让用户知道，而不是把空串当结果交出去。
            message["content"] = (
                "【空回答】模型这一轮没有返回内容，也没有请求工具。请重问一次或换个说法。"
            )

    print(f"  [agent] content={repr(message.get('content'))[:20]} tool_calls={len(tool_calls)}")
    return {"messages": [message], "steps": state.get("steps", 0) + 1}


def tools_node(state: AgentState) -> dict[str, list[dict[str, Any]]]:
    """执行最后一条 assistant 消息中声明的工具调用。

    分两批：签名在历史里出现过的直接拦下（回一条提示），剩下的才真执行。
    被拦的也必须回 tool 消息——每个 tool_call 都得有 tool_call_id 配对的
    回应，少一条下一次请求就 400。
    """
    last = state["messages"][-1]
    calls = last.get("tool_calls") or []
    seen = {
        _signature(call)
        for message in state["messages"][:-1]
        for call in (message.get("tool_calls") or [])
    }

    blocked: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    for call in calls:
        signature = _signature(call)
        if signature in seen:
            print(f"  [tools] 拦截重复调用 {signature}")
            blocked.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": (
                        f"这个调用（{signature}）前面已经执行过，结果不会变。"
                        "请换参数，或直接用已有信息回答。"
                    ),
                }
            )
        else:
            fresh.append(call)

    print(f"  [tools] 执行 {len(fresh)} 个工具（拦下 {len(blocked)} 个重复调用）")
    return {"messages": blocked + (run_tools(fresh) if fresh else [])}


def _drop_unanswered_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """删掉没有 tool 回应的 assistant(tool_calls)。

    OpenAI 兼容接口的硬约束：assistant 里声明了几个 tool_call，后面就必须有
    几条对应的 tool 消息（按 tool_call_id 配对），少一条整条请求 400。
    轮次用尽时最后一条正好是这种「还没人回应」的消息——直接发出去就会像
    README 里那个「孤立 tool 消息」一样炸，只是方向相反。
    """
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            pending = [c for c in message["tool_calls"] if c.get("id") not in answered]
            if pending:
                if (message.get("content") or "").strip():
                    message = {k: v for k, v in message.items() if k != "tool_calls"}
                else:
                    continue  # 空内容又没配上回应，这条没有信息量，丢掉
        cleaned.append(message)
    return cleaned


def finalize_node(state: AgentState) -> dict[str, Any]:
    """轮次用尽后的收尾：先把欠的工具补跑，再要一段自然语言。

    这是本版和手写版最主要的行为差异。手写版在 for 循环跑完后直接返回
    一句兜底提示，把已经拿到的工具结果全丢了；这里让模型基于这些结果
    尽量作答——大多数「超限」其实是模型想再查一遍，而不是真的没材料。

    补跑欠账这一步是踩出来的：max_steps=1 时模型刚发完工具请求就被掐断，
    此时手里一条材料都没有，收尾只能干说「已达上限」；而且那条
    assistant(tool_calls) 没有配对的 tool 消息，直接送去请求会 400。
    工具执行不消耗模型轮次，所以先跑完再收尾是划算的。
    """
    limit = state.get("max_steps", MAX_STEPS)
    messages = list(state["messages"])

    pending = messages[-1].get("tool_calls") if messages else None
    if pending:
        print(f"  [finalize] 补跑 {len(pending)} 个未执行的工具")
        messages = messages + run_tools(pending)
    messages = _drop_unanswered_tool_calls(messages)

    note = f"（已达到 {limit} 轮工具调用上限，以下基于已获得的信息作答）"
    hint = {
        "role": "user",
        "content": (
            "工具调用次数已达上限。请只根据上面已经拿到的信息回答问题，"
            "不要再请求任何工具。如果信息不足，就直接说明还缺什么。"
        ),
    }

    try:
        data = call_llm(messages + [hint], with_tools=False)
        message = dict(data["choices"][0]["message"])
        message.pop("reasoning_content", None)
        content = (message.get("content") or "").strip()
        if not content:
            content = f"已用满 {limit} 轮工具调用，模型未能给出结论。请把任务拆小一点再问。"
        answer = f"{note}\n\n{content}"
        print(f"  [finalize] 收尾成功，{len(content)} 字")
        return {
            "messages": [{"role": "assistant", "content": answer}],
            "stop_reason": "max_steps",
        }
    except Exception as exc:  # 收尾失败不能把整张图带崩
        print(f"  [finalize] 收尾失败 {type(exc).__name__}: {exc}")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"{note}\n\n收尾请求失败（{type(exc).__name__}），"
                    f"已执行 {state.get('steps', 0)} 轮工具调用。请把问题拆小一点再问。",
                }
            ],
            "stop_reason": "error",
        }


def should_continue(state: AgentState) -> str:
    """有工具调用时进入 tools；轮次用尽则去 finalize；否则结束。"""
    last = state["messages"][-1]
    if not last.get("tool_calls"):
        return END
    if state.get("steps", 0) >= state.get("max_steps", MAX_STEPS):
        return "finalize"
    return "tools"


def build_graph() -> Any:
    """构建 agent → tools ⇄ agent、超限 → finalize 的状态图。"""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


APP = build_graph()


def recursion_limit(max_steps: int) -> int:
    """把业务轮次换算成图执行步数上限。

    一轮 = agent + tools 两个超步，跑满后可能再进一次 finalize，
    再加边界的进出。留 5 步余量：上一版是 `* 2 + 1`，正好卡在边界上，
    图里多一个节点就会误伤正常流程。

    它和 max_steps 是两层保险，不要再混用：
        max_steps      业务层，触发后还能给用户一个答案
        recursion_limit 框架层，只在代码写错时兜底
    """
    return max_steps * 2 + 5


def run_graph(user_input: str, max_steps: int = MAX_STEPS) -> str:
    """运行状态图；超限由 finalize 收敛，不再依赖异常。"""
    initial_state = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "steps": 0,
        "max_steps": max_steps,
        "stop_reason": None,
    }
    try:
        result = APP.invoke(
            initial_state,
            {"recursion_limit": recursion_limit(max_steps)},
        )
    except GraphRecursionError:
        # 走到这里说明图结构或路由写错了（正常路径由 finalize 收敛）。
        # 留这条兜底是为了让错误可读，而不是让进程抛栈。
        return f"【兜底】图执行步数超过 {recursion_limit(max_steps)}，疑似路由未收敛"

    print(f"  [graph] 轮次 {result.get('steps', 0)}  终止 {result.get('stop_reason') or '正常'}")
    return result["messages"][-1].get("content") or "(空)"


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
