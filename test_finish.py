"""test_finish.py —— 测一个正常提问几乎触发不到的分支：finish_reason == "length"

思路：不改业务代码，用 unittest.mock 把 call_llm 换成一个假的，
让它返回我们指定的响应。这样"被截断"这件事可以在本地稳定复现。

这就是"怎么测一个线上难复现的分支"的标准做法。
"""

from unittest.mock import patch
import react_loop


def 造假(finish, content):
    """造一个和真实响应结构一样的假数据。注意 finish_reason 和 message 同级。"""
    return {
        "choices": [{
            "index": 0,
            "finish_reason": finish,
            "message": {"role": "assistant", "content": content},
        }]
    }


# ---------- 情况 1：被截断 ----------
截断 = 造假("length", "西安今天天气晴朗，气温大约在 26 度左右，适合出")
with patch.object(react_loop, "call_llm", return_value=截断):
    print("=== 伪造 finish_reason = 'length' ===")
    print("返回：", react_loop.run("写一篇 5000 字的西安天气分析"))

print()

# ---------- 情况 2：正常说完 ----------
正常 = 造假("stop", "西安今天晴，26 度。")
with patch.object(react_loop, "call_llm", return_value=正常):
    print("=== 伪造 finish_reason = 'stop' ===")
    print("返回：", react_loop.run("查一下西安天气"))
