"""多智能体编排：Supervisor 派活，worker 各自带工具干活。

和前三条链路的关系
------------------
    react_loop   一个 Agent，五个工具全给它
    graph_agent  同一个 Agent，循环被拆成图
    mcp_agent    同一个 Agent，工具从远端发现
    team_agent   多个 Agent，谁来做由 Supervisor 决定

    前三条比的是「控制流放在哪」，这一条比的是「上下文和权限怎么切」。

为什么不是「一个 Agent 干所有事」
--------------------------------
多智能体的收益不在「更聪明」，在**隔离**。把五个工具的 schema 和十次工具
调用的结果全塞进一个上下文，模型到第十二条消息就开始丢重点——context.py
解决的正是这件事，但那只是**少喂一点**；这里是**根本不喂**：研究员看不到
计算器的工具，写作者看不到任何工具，它只拿到一份被压成一行一条的账本。

一句话：**裁剪是省，编排是不给它看。**

Supervisor 的职责也只有一条：决定下一步交给谁，不负责回答。它看账本，
不看工具调用流水——所以它的上下文不随子 Agent 干活变长。
"""

import re
import sys
from typing import Annotated, Any, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from react_loop import TOOLS_SCHEMA, call_llm, run_tools

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAX_ROUNDS = 4  # Supervisor 最多派几单；用尽后强制交给 writer 收口
WORKER_STEPS = 3  # 单个 worker 内部最多几次模型调用
NOTE_LIMIT = 600  # 一条结论进账本时的字数上限

_TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS_SCHEMA}

DEFAULT_WRITER_INSTRUCTION = "根据账本里已有的结论，直接回答总任务"
DEFAULT_WORKER_INSTRUCTION = "按总任务自行决定这一步做什么"

AGENTS: dict[str, dict[str, Any]] = {
    "researcher": {
        "desc": "查事实：天气、当前时间、读本地文件、列目录",
        "tools": ["get_weather", "get_current_time", "read_file", "list_dir"],
        "prompt": """你是研究员，只负责查事实。
规则：
1. 需要事实就调用工具，不要凭记忆回答。
2. 工具报错时换参数或换方式，不要重复同一个调用。
3. **只报工具返回的原始数值和事实，不要做任何算术、换算或推断**——
   计算是计算员的活。需要算的地方，把数字原样写出来交给下一步。
4. 一到两句话，不要解释过程。""",
    },
    "calculator": {
        "desc": "只做数学计算，不查资料",
        "tools": ["calculate"],
        "prompt": """你是计算员，只负责算数。
规则：
1. 必须调用 calculate 工具，不要自己心算。
2. 表达式里只能出现数字、括号和 + - * / % 。
3. 算完直接给结果，不要解释。""",
    },
    "writer": {
        "desc": "把已有结论整理成最终答案，不查也不算",
        "tools": [],
        "prompt": """你是写作者，负责把已有结论整理成给用户的最终答案。
规则：
1. 只用账本里出现过的结论，不要补充账本里没有的事实。
2. 账本里没有的信息，直接说没查到，不要猜。
3. 不要提账本、角色、调度这些内部过程，直接回答问题。
4. 简短作答，不要复述步骤。""",
    },
}

# 模型偶尔把角色名写成中文，或者把收尾写成 DONE。这些都不该让流程崩：
# 认不出来的名字一律当「该收尾了」，而不是报错或原地重试。
_ALIASES = {
    "研究员": "researcher",
    "查资料": "researcher",
    "搜索": "researcher",
    "计算员": "calculator",
    "计算器": "calculator",
    "calc": "calculator",
    "写作者": "writer",
    "作家": "writer",
    "撰写": "writer",
    "write": "writer",
    "done": "writer",
    "finish": "writer",
    "结束": "writer",
    "完成": "writer",
}

_MENU = "\n".join("- %s：%s" % (name, spec["desc"]) for name, spec in AGENTS.items())

SUPERVISOR_SYSTEM = f"""你是多智能体系统的调度员。你不回答问题，只决定下一步交给谁。

可选角色：
{_MENU}

规则：
1. 一次只派一个角色；指令要具体到「做什么」，不要写「继续」「再查一下」这种空话。
2. 不需要再查、再算时，派 writer 收尾。
3. 同一个角色、同一件事不要派第二次——账本里已有的结论直接用。
4. 账本里有数字但没有算出来的结果时，派 calculator，不要假设它已经算过。
5. 只输出一行「角色名:指令」，不要解释、不要编号、不要加粗。

例如：researcher:查北京今天的天气"""

SUPERVISOR_TEMPLATE = """总任务：{task}

账本（已完成的步骤）：
{ledger}

下一步派谁？只输出一行。"""


def _append(left: list[Any], right: list[Any]) -> list[Any]:
    """LangGraph reducer：把节点产生的新条目追加进列表。"""
    return left + right


class TeamState(TypedDict):
    """编排层的共享状态。

    和 AgentState 的区别：那里存的是**完整消息流**，这里存的是**账本**。
    账本一行一条（谁、要它做什么、结论是什么），子 Agent 的工具调用流水
    不进主状态——否则多智能体只是把上下文变大，不是变小。
    """

    task: str
    ledger: Annotated[list[dict[str, Any]], _append]
    next_agent: str
    instruction: str
    rounds: int
    max_rounds: int
    stop_reason: str | None


def _normalize(token: str) -> str | None:
    """角色名归一化；认不出来返回 None（由调用方当收尾处理）。"""
    key = token.strip().lower()
    if key in AGENTS:
        return key
    return _ALIASES.get(key)


def parse_route(raw: str) -> tuple[str, str]:
    """从 Supervisor 的输出里取「角色:指令」。

    为什么不能直接拿第一行按冒号切：模型会加粗、编号、写中文冒号，
    还会把角色名后面的话整个写成一段解释。所以两条路都留着——
    先按「角色:指令」匹配，不行再整行当角色名认。

    实在认不出来就返回 writer：**收敛到一个能出结果的节点，比抛异常好**。
    这和 graph_agent 里「超限去 finalize」是同一个取舍——流程卡住时，
    给用户一段基于现有材料的回答，永远好过一段报错。
    """
    text = (raw or "").strip()
    if not text:
        return "writer", DEFAULT_WRITER_INSTRUCTION

    line = next((item.strip() for item in text.splitlines() if item.strip()), "")

    match = re.match(r"^[-*\d.、\s]*([A-Za-z_\u4e00-\u9fff]+)\s*\**\s*[:：]\s*(.+)$", line)
    if match:
        name = _normalize(match.group(1))
        if name:
            return name, match.group(2).strip().strip("*` ") or DEFAULT_WRITER_INSTRUCTION

    token = _normalize(re.sub(r"[\s\-*`#]", "", line))
    if token:
        return token, DEFAULT_WORKER_INSTRUCTION

    return "writer", DEFAULT_WRITER_INSTRUCTION


def _ledger_text(ledger: list[dict[str, Any]], limit: int = 8) -> str:
    """账本转文本：只给最近几条，每条压成一行。

    截断是刻意的：调度员需要的是「做过什么、结论是什么」，
    不是子 Agent 的完整推理。给全了，Supervisor 的上下文会随轮次线性涨。
    """
    rows = []
    for index, item in enumerate(ledger[-limit:], 1):
        instruction = item.get("instruction", "")[:40]
        rows.append(
            "%d. %s — %s → %s" % (index, item["agent"], instruction, item.get("result", ""))
        )
    return "\n".join(rows) or "（还没开始）"


def run_scoped_tools(tool_calls: list[dict[str, Any]], allowed: list[str]) -> list[dict[str, Any]]:
    """只执行本角色允许的工具，越界的回一条提示而不是执行。

    为什么这层必须有：schema 里不给某个工具，只是「模型大概率不会调」，
    不是「调不到」。模型完全可能凭上下文里的印象写一个 get_weather 出来。
    真让它执行，权限隔离就成了摆设；而直接丢掉这条调用，又会破坏
    tool_call 必须配对 tool 消息的约束，下一次请求直接 400。
    """
    allowed_set = set(allowed)
    rejected = []
    permitted = []
    for call in tool_calls:
        name = (call.get("function") or {}).get("name")
        if name in allowed_set:
            permitted.append(call)
        else:
            print(f"  [scope] 拒绝越界工具 {name}")
            rejected.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": (
                        f"当前角色没有 {name} 这个工具，可用的是："
                        f"{'、'.join(allowed) or '（无，本角色不使用工具）'}。"
                        "请改用可用工具，或直接给出结论。"
                    ),
                }
            )
    return rejected + (run_tools(permitted) if permitted else [])


def _worker_system(name: str, task: str, ledger: list[dict[str, Any]]) -> str:
    """子 Agent 的系统提示：自己的规则 + 总任务 + 上游结论。

    注意没有别人的消息流。它看不到研究员调了几次工具、失败了几次，
    只看到「研究员 — 查北京天气 → 北京多云 18℃」。这就是隔离省下来的上下文。
    """
    return "%s\n\n总任务：%s\n\n已有进展：\n%s" % (
        AGENTS[name]["prompt"],
        task,
        _ledger_text(ledger),
    )


def _run_worker(name: str, state: TeamState) -> dict[str, Any]:
    """跑一个子 Agent 的小闭环，把结论记进账本。

    用尽步数时**不再额外调一次模型**，直接把工具结果交出去。
    这和 graph_agent.finalize 的做法不一样，理由也不同：那里的输出就是
    给用户的答案，值得再问一次把它说顺；这里输出是**材料**，材料不需要
    润色，润色是 writer 的活。多调一次只是在替 writer 干一半活。
    """
    spec = AGENTS[name]
    instruction = state.get("instruction") or DEFAULT_WORKER_INSTRUCTION
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _worker_system(name, state.get("task", ""), state.get("ledger") or []),
        },
        {"role": "user", "content": instruction},
    ]
    schemas = [_TOOL_BY_NAME[tool] for tool in spec["tools"]]
    collected: list[dict[str, Any]] = []

    print(f"  [{name}] 接单：{instruction[:40]}")
    for step in range(1, WORKER_STEPS + 1):
        data = call_llm(messages, with_tools=bool(schemas), tools=schemas or None)
        message = dict(data["choices"][0]["message"])
        message.pop("reasoning_content", None)
        calls = message.get("tool_calls") or []

        if not calls:
            result = (message.get("content") or "").strip()
            if not result:
                result = f"【{name} 空回答】这一步没有产出。"
            print(f"  [{name}] 结论：{result[:60]}")
            return {"ledger": [{"agent": name, "instruction": instruction, "result": result}]}

        print(f"  [{name}] 第 {step} 步调用 {len(calls)} 个工具")
        messages.append(message)
        results = run_scoped_tools(calls, spec["tools"])
        messages.extend(results)
        collected.extend(results)

    raw = "\n".join(str(item.get("content", "")) for item in collected)[-NOTE_LIMIT:]
    result = f"（已达 {WORKER_STEPS} 步上限，以下是原始工具结果）\n{raw}"
    print(f"  [{name}] 达到步数上限，交回 {len(collected)} 条原始结果")
    return {"ledger": [{"agent": name, "instruction": instruction, "result": result}]}


def researcher_node(state: TeamState) -> dict[str, Any]:
    """查事实的角色。"""
    return _run_worker("researcher", state)


def calculator_node(state: TeamState) -> dict[str, Any]:
    """只算数的角色。"""
    return _run_worker("calculator", state)


def writer_node(state: TeamState) -> dict[str, Any]:
    """收尾角色：无工具，只整理账本。"""
    return _run_worker("writer", state)


def supervisor_node(state: TeamState) -> dict[str, Any]:
    """决定下一步交给谁。

    轮次用尽时**不再问模型**：它只会再派一单，既费 token 又可能把同一件事
    重派一遍。直接转 writer 收口——上限不是崩溃点，而是「用现有材料收尾」
    的触发点，这一点和 graph_agent 的 finalize 是同一个取舍。
    """
    rounds = state.get("rounds", 0)
    max_rounds = state.get("max_rounds", MAX_ROUNDS)
    if rounds >= max_rounds:
        print(f"  [supervisor] 已达 {max_rounds} 轮调度上限，强制收口")
        return {
            "next_agent": "writer",
            "instruction": DEFAULT_WRITER_INSTRUCTION,
            "rounds": rounds,
            "stop_reason": "max_rounds",
        }

    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {
            "role": "user",
            "content": SUPERVISOR_TEMPLATE.format(
                task=state.get("task", ""), ledger=_ledger_text(state.get("ledger") or [])
            ),
        },
    ]
    data = call_llm(messages, with_tools=False)
    raw = (data["choices"][0]["message"].get("content") or "").strip()
    name, instruction = parse_route(raw)
    print(f"  [supervisor] 第 {rounds + 1} 单 → {name}：{instruction[:40]}")
    return {"next_agent": name, "instruction": instruction, "rounds": rounds + 1}


def route_from_supervisor(state: TeamState) -> str:
    """把 next_agent 映射成节点名；认不出来的名字一律走 writer。

    这层看似多余（parse_route 已经保证只返回三个名字之一），但它是
    最后一道防线：将来有人改了 parse_route、或者状态被外部写脏，
    图不会因为一个不存在的节点名直接抛 KeyError。
    """
    name = state.get("next_agent") or ""
    return name if name in AGENTS else "writer"


def build_graph() -> Any:
    """构建 supervisor ⇄ worker、writer 收尾的状态图。"""
    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "calculator": "calculator", "writer": "writer"},
    )
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("calculator", "supervisor")
    graph.add_edge("writer", END)
    return graph.compile()


APP = build_graph()


def recursion_limit(max_rounds: int) -> int:
    """业务轮次换算成图执行步数上限。

    一轮 = supervisor + 一个 worker，两个超步；跑满后还要再进一次 supervisor
    转去 writer，再加边界。留 6 步余量，理由同 graph_agent：卡在边界上，
    以后图里多一个节点就会误伤正常流程。
    """
    return max_rounds * 2 + 6


def run_team(task: str, max_rounds: int = MAX_ROUNDS) -> str:
    """跑一次多智能体协作，返回 writer 的最终答案。"""
    initial_state: TeamState = {
        "task": task,
        "ledger": [],
        "next_agent": "",
        "instruction": "",
        "rounds": 0,
        "max_rounds": max_rounds,
        "stop_reason": None,
    }
    try:
        result = APP.invoke(initial_state, {"recursion_limit": recursion_limit(max_rounds)})
    except GraphRecursionError:
        return f"【兜底】调度步数超过 {recursion_limit(max_rounds)}，疑似路由未收敛"

    ledger = result.get("ledger") or []
    notes = [item for item in ledger if item["agent"] == "writer"]
    if notes:
        return notes[-1]["result"]
    if ledger:
        return "（未经过写作者整理）\n" + ledger[-1]["result"]
    return "【空结果】调度没有产出任何结论。"


def main() -> None:
    """运行三个演示用例；需要有效 API Key。"""
    cases = [
        ("用例1 两步协作", "查一下北京的天气，再把温度和 10 相加", MAX_ROUNDS),
        ("用例2 单步", "现在几点了？", MAX_ROUNDS),
        ("用例3 调度上限", "分别查西安和上海的天气，各自换算成华氏度", 1),
    ]
    for title, question, limit in cases:
        print("\n" + "=" * 55)
        print(f"{title}：{question}")
        print("=" * 55)
        print("答案:", run_team(question, max_rounds=limit))


if __name__ == "__main__":
    main()
