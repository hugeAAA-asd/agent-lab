# agent-lab

用同一个工具集实现四条 Agent 执行链路：**手写 ReAct → LangGraph 状态图 → MCP 协议接入 → Supervisor 多智能体编排**。前三条比的是「循环控制和工具发现放在哪」，第四条比的是「上下文和权限怎么切」。

## 实现对照

| 实现 | 入口 | 核心问题 |
|---|---|---|
| 手写 ReAct | `react_loop.py` | 解析 `tool_calls`、执行工具、回灌结果，直到模型给出最终回答 |
| LangGraph | `graph_agent.py` | 将同一循环改写为 `agent` / `tools` 节点、条件边和共享状态 |
| MCP Agent | `mcp_agent.py` | 运行时发现并调用外部 MCP Server 的工具，Agent 不硬编码工具清单 |
| 多智能体编排 | `team_agent.py` | Supervisor 派活，每个角色只看自己那几个工具、只读压缩成一行一条的账本 |

`mcp_client.py` 用于单独观察 MCP 握手、工具发现和调用过程；`tools_core.py` 保存与模型、协议无关的本地工具实现；`context.py` 负责按完整回合裁剪消息。

## 工具与可靠性设计

本地 MCP Server 提供五个演示工具：

- `get_current_time`：获取当前时间。
- `calculate`：计算仅含数字和允许运算符的表达式，不开放任意代码执行。
- `get_weather`：返回固定演示数据，不请求真实天气服务。
- `read_file`：读取文本，最多返回 2000 个字符。
- `list_dir`：列出目录内容。

前三条链路（单 Agent）都包含以下边界处理：

- 工具异常转换为 `role=tool` 消息回灌，避免一次工具失败直接终止 Agent。
- 轮次上限由状态里的 `steps` 承担，用尽后进入 `finalize` 节点：先补跑尚未执行的工具，再基于已有信息作答；`recursion_limit` 只作为框架层兜底。
- 同工具同参数的重复调用在工具节点直接拦截，回一条提示而不是重复执行。
- 轮次掐断时会出现没有配对的 `assistant(tool_calls)`，请求前统一清理，避免 400。
- 按完整回合裁剪历史，避免孤立的 `tool` 消息找不到对应的 `assistant(tool_calls)`。
- 模型因 `finish_reason="length"` 停止且没有工具调用时，明确返回截断提示。

多智能体那条链路额外承担两层边界：

- **角色边界**：每个角色只拿到自己的工具 schema；模型若硬调越界工具，回一条提示而不是执行——既守住权限，也守住 `tool_call` 的配对。
- **调度收敛**：Supervisor 的输出只认「角色名:指令」，`DONE`、中文别名、整段解释一律按收尾处理；调度轮次用尽后不再问模型，直接转 writer 用账本收口。

这些机制只能降低失败风险，不能保证所有异常输入都能恢复。

## 快速开始

要求：Python 3.11+。仅 `fs` 模式额外需要 Node.js 和 `npx`。

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env  # 填入自己的 DEEPSEEK_API_KEY
```

运行四种实现：

```bash
python react_loop.py          # 手写 ReAct
python graph_agent.py         # LangGraph
python mcp_agent.py local     # 本地 MCP Server
python mcp_agent.py fs        # 官方 filesystem MCP Server，需要 Node.js
python team_agent.py          # Supervisor 多智能体编排
```

单独检查 MCP 协议交互：

```bash
python mcp_client.py
```

> `fs` 模式会根据操作系统自动选择 `npx.cmd`（Windows）或 `npx`（其他系统）。

## 自动化检查

离线测试使用 mock，不请求模型接口，也不需要 API Key：

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest
```

无需安装开发依赖的本地工具冒烟检查：

```bash
python -m scripts.smoke_test
```

验证范围和人工实验记录见 [`docs/verification.md`](docs/verification.md)，架构说明见 [`docs/architecture.md`](docs/architecture.md)。

## 项目结构

```text
agent-lab/
├── react_loop.py       # 手写 ReAct 循环
├── graph_agent.py      # LangGraph 状态图
├── mcp_agent.py        # 动态发现并调用 MCP 工具
├── team_agent.py       # Supervisor 多智能体编排
├── mcp_server.py       # 本地 MCP Server
├── mcp_client.py       # MCP 协议验证客户端
├── tools_core.py       # 纯工具实现与本地分发
├── context.py          # 按回合裁剪消息
├── docs/               # 架构与验证文档
├── scripts/            # 离线冒烟检查
└── tests/              # 不访问真实模型的自动化测试
```

## 已知局限

- 工具串行执行，没有并发、缓存或持久化记忆。
- 上下文按完整回合取整，消息数可能超过 `KEEP_MSG`；尚未按 token 数裁剪。
- 没有流式输出、权限确认和生产级沙箱；文件工具仅适合本地演示。
- 天气为固定数据；模型生成结果具有随机性，人工用例不代表稳定准确率。
- 仅在 Windows 与当前 Python 3.12 环境做过人工验证，其他平台主要依赖 CI 的离线检查。
- 多智能体是单机串行调度，没有并行、没有消息总线、没有共享工作区；角色边界靠提示词约束——实测中研究员会顺手替计算员算数，导致计算员一次都没被派到，加禁止越权的规则后才纠正。**工具白名单挡得住「调用别人的工具」，挡不住「用自然语言自己算」。**
