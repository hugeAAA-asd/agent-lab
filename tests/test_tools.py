"""本地工具与参数解析测试，不访问网络。"""

import pytest

from tools_core import calculate, dispatch, list_dir, read_file, safe_args


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("25*4+10", "110"), ("(2+3)*4", "20"), ("-2+5", "3"), ("10/4", "2.5")],
)
def test_calculate(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "2**8", "open('x')", "1+", ""])
def test_calculate_rejects_unsupported_expression(expression: str) -> None:
    with pytest.raises(ValueError):
        calculate(expression)


def test_safe_args() -> None:
    assert safe_args('{"expr": "1+1"}') == {"expr": "1+1"}
    assert safe_args({"city": "西安"}) == {"city": "西安"}
    assert safe_args("坏 JSON") == {}
    assert safe_args("[1, 2]") == {}
    assert safe_args(None) == {}


def test_dispatch_supports_all_five_tools(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")

    assert dispatch("calculate", {"expr": "2*10"}) == "20"
    assert "note.txt" in dispatch("list_dir", {"path": str(tmp_path)})
    assert dispatch("read_file", {"path": str(tmp_path / "note.txt")}) == "hello"
    assert dispatch("get_weather", {"city": "西安"}) == "晴 26℃"
    assert dispatch("get_current_time", {})


def test_unknown_tool_and_file_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="未知工具"):
        dispatch("not_a_tool", {})
    with pytest.raises(FileNotFoundError):
        read_file(str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        list_dir(str(tmp_path / "missing"))
