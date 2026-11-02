"""与模型和传输协议无关的本地工具实现。"""

import ast
import json
import operator
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "demo.db"
SQL_MAX_ROWS = 50

# 工具的说明文字只有一份：React/LangGraph 的 tools schema 和 MCP Server 的
# description 都引它，避免「模型看到的表结构」和「真实表结构」各写一遍。
SQL_TOOL_DOC = (
    "在只读演示电商库（SQLite）上执行一条查询，只允许单条 SELECT / WITH，最多返回 50 行。\n"
    "表 products(id, name, category, price, stock)：商品、品类、单价、库存\n"
    "表 orders(id, order_no, city, product_id, qty, amount, status, created_at)：订单\n"
    "  - product_id 关联 products.id，问「品类」要 JOIN products\n"
    "  - status 取值 paid 已付款 / refunded 已退款 / pending 待付款\n"
    "  - amount 是订单实付金额，qty 是件数\n"
    "问哪个品类卖得最好、退款多少单、哪些商品库存不足时用它，不要凭记忆编数字。"
)

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _evaluate_node(node: ast.AST) -> float:
    """递归计算受限 AST 节点，不执行名称、属性或函数调用。"""
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("表达式不能包含布尔值")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate_node(node.left), _evaluate_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise ValueError("只支持数字、括号和 + - * / % 运算符")


def calculate(expr: str) -> str:
    """计算受限数学表达式，通过 AST 白名单避免执行任意 Python 代码。"""
    if not expr:
        raise ValueError("expr 不能为空")
    if len(expr) > 200:
        raise ValueError("表达式不能超过 200 个字符")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("表达式格式不正确") from exc
    return str(_evaluate_node(tree))


def get_weather(city: str) -> str:
    """返回演示天气数据；未连接真实天气服务。"""
    if not city:
        raise ValueError("city 不能为空")
    fake = {"西安": "晴 26℃", "北京": "多云 18℃", "上海": "小雨 22℃"}
    return fake.get(city, f"{city}: 暂无数据")


def _resolve_path(path: str) -> Path:
    """将相对路径解析到项目目录；绝对路径保持原样。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else BASE_DIR / candidate


def read_file(path: str) -> str:
    """读取本地文本文件，最多返回 2000 个字符。"""
    if not path:
        raise ValueError("path 不能为空")
    candidate = _resolve_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"文件不存在: {candidate}")
    return candidate.read_text(encoding="utf-8", errors="replace")[:2000]


def list_dir(path: str = ".") -> str:
    """列出目录内的直接子项名称，用顿号连接。"""
    candidate = _resolve_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"目录不存在: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"不是目录: {candidate}")
    return "、".join(item.name for item in candidate.iterdir())


_READ_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),  # WITH RECURSIVE 需要
}


def _only_read(action: int, *_args: Any) -> int:
    """authorizer 回调：只放行「读表」和「调函数」，其余一律拒绝。

    `mode=ro` 已经在连接层挡住了写入，这一层是第二道闸：
    PRAGMA、ATTACH、CREATE 这类不写数据但能改会话状态的操作也一起挡掉。
    """
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def _cell(value: Any) -> str:
    """把单元格转成短文本；整数形态的浮点不显示 .0。"""
    if value is None:
        return "NULL"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def sql_query(sql: str) -> str:
    """在只读演示库上执行单条 SELECT，最多返回 50 行。"""
    if not sql or not sql.strip():
        raise ValueError("sql 不能为空")
    if not DB_PATH.exists():
        # clone 下来第一次调用就建库，省掉一步手工初始化
        from scripts.init_db import build

        build()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        conn.set_authorizer(_only_read)
        cursor = conn.execute(sql)
        columns = [item[0] for item in cursor.description or []]
        if not columns:
            raise ValueError("这条语句没有返回结果集，只支持 SELECT / WITH 查询")
        rows = cursor.fetchmany(SQL_MAX_ROWS + 1)
    except sqlite3.Error as exc:
        # 语法错、表名错、越权写：都当成可回灌的输入错误，让模型自己改 SQL
        raise ValueError(f"SQL 执行失败：{exc}") from exc
    finally:
        conn.close()

    if not rows:
        return "查询成功，但没有匹配的行。"

    lines = [" | ".join(columns)]
    lines.extend(" | ".join(_cell(value) for value in row) for row in rows[:SQL_MAX_ROWS])
    if len(rows) > SQL_MAX_ROWS:
        lines.append(f"（结果超过 {SQL_MAX_ROWS} 行，只返回前 {SQL_MAX_ROWS} 行）")
    return "\n".join(lines)


def dispatch(name: str, args: dict[str, Any]) -> str:
    """按工具名调用本地实现；未知工具抛出 ValueError。"""
    table = {
        "get_current_time": lambda a: get_current_time(),
        "calculate": lambda a: calculate(a.get("expr", "")),
        "get_weather": lambda a: get_weather(a.get("city", "")),
        "read_file": lambda a: read_file(a.get("path", "")),
        "list_dir": lambda a: list_dir(a.get("path", ".")),
        "sql_query": lambda a: sql_query(a.get("sql", "")),
    }
    function = table.get(name)
    if function is None:
        raise ValueError(f"未知工具: {name}")
    return function(args)


def safe_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """将模型返回的 arguments 解析为字典；无效输入返回空字典。"""
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return args if isinstance(args, dict) else {}
