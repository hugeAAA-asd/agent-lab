"""不依赖 Agent 框架的 ReAct 循环。运行：python react_loop.py"""

import logging
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

from context import trim_by_turn
from tools_core import BASE_DIR, dispatch, safe_args

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("DEEPSEEK_API_KEY")
URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-flash"
MAX_STEPS = 6
KEEP_MSG = 12

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个会使用工具的助手。
规则：
1. 需要实时信息或计算时必须调用工具，禁止凭记忆编造。
2. 一次只推进一步，拿到工具结果后再决定下一步。
3. 工具返回错误时，分析原因、换参数或换方式，不要重复同一个调用。
4. 信息齐了再给最终答案，最终答案要简短，不要复述中间过程。"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，例如 25*4+10",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "数学表达式"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的演示天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如西安"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文本文件，最多返回 2000 个字符",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件名或路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出指定目录中的直接子项",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "目录路径，默认当前目录"}},
                "required": [],
            },
        },
    },
]


def call_llm(messages: list[dict[str, Any]], with_tools: bool = True) -> dict[str, Any]:
    """请求 DeepSeek chat/completions；仅在实际调用时要求 API Key。

    with_tools=False 是给「收尾」用的：轮次用尽后只想要一段自然语言，
    这时再把 tools 递过去，模型很可能又发起一轮工具调用，等于白转。
    """
    if not API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 后填写")
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
    }
    if with_tools:
        payload["tools"] = TOOLS_SCHEMA
        payload["tool_choice"] = "auto"
    response = requests.post(
        URL,
        json=payload,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def run_tools(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """串行执行工具，并将异常文本回灌为对应的 tool 消息。"""
    results = []
    for tool_call in tool_calls:
        name = tool_call["function"]["name"]
        args = safe_args(tool_call["function"].get("arguments"))
        print(f"  → 调用 {name}({args})")
        try:
            output = str(dispatch(name, args))
        except Exception as exc:  # 工具错误是 Agent 的输入，不应直接终止循环
            logger.warning("工具 %s 执行失败: %s", name, exc)
            output = f"工具执行失败: {type(exc).__name__}: {exc}"
        print(f"  ← {output[:120]}")
        results.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": output,
            }
        )
    return results


def run(user_input: str, max_steps: int = MAX_STEPS) -> str:
    """执行至多 max_steps 次模型调用，返回最终回答或明确的兜底提示。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for step in range(1, max_steps + 1):
        messages = trim_by_turn(messages, KEEP_MSG)
        data = call_llm(messages)
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = dict(choice["message"])
        message.pop("reasoning_content", None)  # DeepSeek 扩展字段不能原样回传

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            print(f"[step {step}] finish_reason = {finish_reason}")
            if finish_reason == "length":
                return "【截断】回答超出长度上限，只生成了一部分。请把问题拆小一点。"
            return message.get("content") or "(空)"

        print(f"[step {step}] 模型要求调用 {len(tool_calls)} 个工具")
        messages.append(message)
        messages.extend(run_tools(tool_calls))

    return f"【兜底】已用满 {max_steps} 步仍未得到最终答案。请换个问法，或把任务拆小一点。"


def main() -> None:
    """运行四个人工演示用例；需要有效 API Key。"""
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
        print("答案:", run(question, max_steps=limit))


if __name__ == "__main__":
    main()
