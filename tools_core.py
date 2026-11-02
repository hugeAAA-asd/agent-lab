"""与模型和传输协议无关的本地工具实现。"""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).parent

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


def dispatch(name: str, args: dict[str, Any]) -> str:
    """按工具名调用本地实现；未知工具抛出 ValueError。"""
    table = {
        "get_current_time": lambda a: get_current_time(),
        "calculate": lambda a: calculate(a.get("expr", "")),
        "get_weather": lambda a: get_weather(a.get("city", "")),
        "read_file": lambda a: read_file(a.get("path", "")),
        "list_dir": lambda a: list_dir(a.get("path", ".")),
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
