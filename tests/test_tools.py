"""本地工具与参数解析测试，不访问网络。"""

import sqlite3

import pytest

from tools_core import (
    DB_PATH,
    SQL_TOOL_DOC,
    calculate,
    dispatch,
    list_dir,
    read_file,
    safe_args,
    sql_query,
)


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


def test_dispatch_supports_all_six_tools(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")

    assert dispatch("calculate", {"expr": "2*10"}) == "20"
    assert "note.txt" in dispatch("list_dir", {"path": str(tmp_path)})
    assert dispatch("read_file", {"path": str(tmp_path / "note.txt")}) == "hello"
    assert dispatch("get_weather", {"city": "西安"}) == "晴 26℃"
    assert dispatch("get_current_time", {})
    assert "机械键盘" in dispatch("sql_query", {"sql": "SELECT name FROM products LIMIT 1"})


def test_unknown_tool_and_file_errors(tmp_path) -> None:
    with pytest.raises(ValueError, match="未知工具"):
        dispatch("not_a_tool", {})
    with pytest.raises(FileNotFoundError):
        read_file(str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        list_dir(str(tmp_path / "missing"))


def test_sql_query_returns_a_table() -> None:
    out = sql_query("SELECT city, COUNT(*) AS n FROM orders GROUP BY city ORDER BY n DESC")
    head, *rows = out.splitlines()
    assert head == "city | n"
    assert len(rows) == 5


def test_sql_query_supports_join() -> None:
    """问「哪个品类最贵」需要 JOIN，证明这不是单表桩。"""
    out = sql_query(
        "SELECT p.category, SUM(o.amount) AS total FROM orders o "
        "JOIN products p ON p.id = o.product_id GROUP BY p.category ORDER BY total DESC"
    )
    assert out.splitlines()[0] == "category | total"
    assert "图书" in out


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET amount = 0",
        "INSERT INTO products VALUES (99, 'x', 'y', 1, 1)",
        "DROP TABLE orders",
        "CREATE TABLE hack (id INTEGER)",
        "ATTACH DATABASE ':memory:' AS side",
        "PRAGMA table_info(orders)",
    ],
)
def test_sql_query_rejects_anything_but_read(sql: str) -> None:
    """写数据、改结构、改会话状态都被拒；错误信息要能回灌给模型。"""
    with pytest.raises(ValueError, match="SQL 执行失败"):
        sql_query(sql)


def test_sql_query_rejects_multiple_statements() -> None:
    with pytest.raises(ValueError, match="SQL 执行失败"):
        sql_query("SELECT 1; SELECT 2")


def test_sql_query_reports_unknown_table() -> None:
    """表名写错要原样报错，模型才能自己改 SQL 重试。"""
    with pytest.raises(ValueError, match="no such table"):
        sql_query("SELECT * FROM not_a_table")


def test_sql_query_caps_rows_at_fifty() -> None:
    out = sql_query("SELECT * FROM orders")
    lines = out.splitlines()
    assert len(lines) == 1 + 50 + 1
    assert "只返回前 50 行" in lines[-1]


def test_sql_query_handles_empty_result() -> None:
    assert sql_query("SELECT * FROM orders WHERE city = '拉萨'") == "查询成功，但没有匹配的行。"


def test_sql_tool_doc_matches_real_schema() -> None:
    """说明文字里必须包含真表的每一列；改了表结构忘了改文档，这条会红。"""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        for table in ("products", "orders"):
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            assert columns, f"{table} 不存在"
            missing = [column for column in columns if column not in SQL_TOOL_DOC]
            assert not missing, f"{table} 的这些列没写进 SQL_TOOL_DOC: {missing}"
    finally:
        conn.close()
