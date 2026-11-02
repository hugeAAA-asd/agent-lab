# 架构说明

## 1. 共同的 Agent 闭环

三条实现都遵循同一个过程：

```text
用户输入
   ↓
模型生成回答或 tool_calls
   ↓ 有工具调用
执行工具并生成 role=tool 消息
   ↓
把工具结果回灌给模型
   ↓
模型给出最终回答
```

区别在于流程控制与工具所在位置，而不是提示词是否更复杂。

## 2. 手写 ReAct

`react_loop.py` 使用显式循环维护消息列表：

- `call_llm()` 只负责模型请求。
- `run_tools()` 负责参数解析、工具分发和异常回灌。
- `run()` 负责轮数上限与终止条件。
- `TOOLS_SCHEMA` 明确声明模型可见的工具。

这种实现适合观察消息协议：为什么 `assistant(tool_calls)` 必须先于对应的 `tool` 消息，以及为什么裁掉父消息会导致接口校验失败。

## 3. LangGraph 状态图

`graph_agent.py` 复用手写版的模型与工具函数，只替换控制流：

- `AgentState.messages` 是共享状态。
- `append_messages` 是 reducer，将节点返回的新消息追加到历史。
- `agent_node` 调模型，`tools_node` 执行工具。
- `should_continue` 决定进入工具节点还是结束。
- `tools → agent` 回边形成循环。

轮次上限由 `AgentState.steps` 承担，`recursion_limit` 退化为框架层兜底。两者不要再混用：

- `max_steps`：业务层。用尽后进入 `finalize`，用户仍能拿到答案。
- `recursion_limit = max_steps * 2 + 5`：superstep 上限，只在路由写错时兜底。留 5 步余量，是因为一轮 `agent → tools` 消耗两个 superstep，跑满后还要再进一次 `finalize`。

## 4. MCP 工具解耦

MCP 版本将工具移到独立进程：

```text
mcp_agent.py
   │ initialize / list_tools / call_tool
   │ JSON-RPC over stdio
   ↓
mcp_server.py
   ↓
tools_core.py
```

`to_openai_tools()` 将 MCP 工具 schema 转成模型所需的 function calling schema。MCP 负责工具发现与调用，function calling 负责模型表达调用意图，两者不是替代关系。

切换到官方 filesystem Server 时，Agent 循环不需要修改，只需替换 `StdioServerParameters`。这验证了工具能力可以通过协议插拔。

## 5. 上下文裁剪

`context.py` 以完整回合为最小单位：

- 始终保留首条 system 消息。
- `assistant(tool_calls)` 与后续连续的 tool 消息放在同一组。
- 从最近一组向前选择，直到达到消息条数预算。

与直接 `messages[-N:]` 相比，这避免了孤立 tool 消息；代价是实际保留条数可能超过预算。当前预算是消息条数，不是 token 数。

## 6. 错误边界

- 本地工具异常：转换为文本工具结果，让模型决定换参数、换工具或说明失败。
- MCP `is_error=True`：同样作为工具结果回灌。
- 达到轮数上限：进入 `finalize`——先补跑尚未执行的工具（工具不消耗模型轮次），再关掉 tools 请求一次自然语言；真的收敛不了才返回兜底文案。手写版在这里直接返回兜底句，已经拿到的材料会被丢掉。
- 未配对的 `assistant(tool_calls)`：请求前统一清理，否则 OpenAI 兼容接口直接 400。
- 重复的工具调用：同工具同参数第二次出现，直接拦截并回一条提示，不重复执行。
- 输出长度截断：没有后续工具调用时返回截断提示，不把半句话当作完整答案。

这些是教学项目中的最小防护，不代替生产环境中的权限控制、沙箱、token 预算和可观测性。
