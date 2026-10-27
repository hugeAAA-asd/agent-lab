# agent-lab

从零手写 Agent 的最小闭环，写了三遍：**ReAct 循环 → LangGraph 状态图 → MCP 协议接入**。

不是调框架 API 的 demo。目的是搞清楚每一层到底在做什么、哪一层负责什么、换掉一层会发生什么。

## 三级实现

| 阶段 | 文件 | 做了什么 |
|---|---|---|
| 1. 手写循环 | `react_loop.py` | 不用任何框架，直接调 chat/completions：工具 schema、`tool_calls` 解析、结果回灌、步数上限、异常转文本回灌 |
| 2. 状态图 | `graph_agent.py` | 把同一个循环改写成 LangGraph 的 `StateGraph`：`agent` / `tools` 两个节点 + 条件边，对照「手写 while」和「图状态机」的差别 |
| 3. 协议解耦 | `mcp_server.py` / `mcp_client.py` / `mcp_agent.py` | 把工具从代码里抽出来做成独立 MCP Server，Agent 侧运行时发现工具清单，`local` / `fs` 两个 Server 可切换 |

辅助文件：`tools_core.py`（5 个工具的纯实现）、`warmup.py`（`dispatch` / `trim_messages` / `safe_args` 三块砖）、`rewrite.py`（重写练习）。

## 五个工具

`get_current_time` / `calculate` / `get_weather` / `read_file` / `list_dir`

`calculate` 用字符白名单替代 `eval`，`read_file` 在文件不存在时抛异常——故意留出错误路径，用来验证错误回灌。

## 验收：4 组用例

`验收记录.md` 里有完整实测输出，这里只列结论：

| 用例 | 验证什么 | 结果 |
|---|---|---|
| 多步任务 | 同一问题需要连调两个工具，顺序由模型决定而非写死 | 通过 |
| 错误回灌 | 读不存在的文件，工具抛异常 | 异常被捕获转文本回灌，模型自行换策略，Agent 不崩 |
| 单步 | 简单问题不应滥用工具 | 通过 |
| 防死循环 | `max_steps=1` 问一个需要两步的问题 | 返回兜底文案，不无限循环 |

## 踩过的坑

这些都是跑出来才发现的，不是查文档能查到的：

- **`role=tool` 消息必须排在它的 `assistant` 之后。** `tool_call_id` 是模型在那条 assistant 的 `tool_calls` 里生成的，服务端会校验每条 tool 消息的 `tool_call_id` 在上文出现过。顺序反了直接 400，而且模型根本没收到请求——是服务端在发出前退件。
- **DeepSeek 返回的 `reasoning_content` 不能原样回传。** 第二轮请求会报错，必须先从 message 里剔掉。
- **工具异常不能往外抛。** 抛出去整个 Agent 就崩了；转成文本回灌，模型拿到错误信息后能换参数或换工具。
- **按条数裁剪历史会把父子消息拆散。** 简单 `messages[-N:]` 可能切掉 assistant 只留下 tool，下一轮同样 400。正确做法是按「assistant + 它所有 tool」成组裁剪，见 `验收记录.md`。
- **MCP Python SDK 2.x 里 `FastMCP` 已改名为 `MCPServer`**，字段也蛇形化了（`server_info` / `input_schema` / `is_error`），网上大量 1.x 教程照抄会报错。
- **Windows 上 `npx` 必须写 `npx.cmd`。** `npx` 是脚本不是可执行文件，Python 的 `CreateProcess` 认不出，报 `WinError 2`。

## 一个对照实验

`mcp_agent.py` 支持 `local` / `fs` 两个 Server：

- `local`：接入本仓库的 5 个工具
- `fs`：接入官方 `@modelcontextprotocol/server-filesystem`（14 个工具，代码不在本仓库）

同一套循环、同一个模型，只换 Server，问「`not_exist.txt` 里写了什么」：

- `local` → 模型答「我没有列目录的工具，无法确认」
- `fs` → 模型先 `list_directory` 确认文件确实不存在，再回答

**Agent 的能力边界由工具决定，不由提示词决定。** 这是这个仓库最想说明的一件事。

## 已知局限

- 工具串行执行，没有并发
- 消息裁剪目前按条数，未实现成组裁剪（见上文坑）
- 无流式输出
- 天气工具是假数据，未接真实 API
- 无持久化记忆

## 运行

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env    # 填入 DEEPSEEK_API_KEY

python react_loop.py    # 手写循环，跑 4 组用例
python graph_agent.py   # LangGraph 版本
python mcp_agent.py     # 接 MCP Server（需要 Node，用于 fs 模式）
```
