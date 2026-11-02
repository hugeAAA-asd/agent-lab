"""不访问模型接口的本地工具冒烟检查。运行：python -m scripts.smoke_test"""

from tools_core import calculate, dispatch, get_weather, safe_args


def main() -> None:
    """检查工具分发、参数解析、非法表达式拦截与只读 SQL。"""
    assert calculate("25*4+10") == "110"
    assert dispatch("calculate", {"expr": "2*10"}) == "20"
    assert get_weather("西安") == "晴 26℃"
    assert safe_args('{"expr": "1+1"}') == {"expr": "1+1"}
    assert safe_args("不是 JSON") == {}
    assert "机械键盘" in dispatch("sql_query", {"sql": "SELECT name FROM products LIMIT 1"})

    try:
        calculate("__import__('os')")
    except ValueError:
        pass
    else:
        raise AssertionError("非法表达式应被拦截")

    try:
        dispatch("sql_query", {"sql": "DELETE FROM orders"})
    except ValueError:
        pass
    else:
        raise AssertionError("只读库不应接受写操作")

    print("本地工具冒烟检查通过。")


if __name__ == "__main__":
    main()
