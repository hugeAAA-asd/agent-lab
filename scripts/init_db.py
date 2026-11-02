"""生成演示电商库 `data/demo.db`（固定种子，重复运行结果一致）。

运行：python -m scripts.init_db

库很小（12 个商品 / 60 笔订单），目的是让 Agent 有一张**真表**可查，
而不是把「查数据库」做成一个返回假数据的桩。
"""

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "demo.db"

SCHEMA = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,
    price    REAL    NOT NULL,
    stock    INTEGER NOT NULL
);

CREATE TABLE orders (
    id         INTEGER PRIMARY KEY,
    order_no   TEXT    NOT NULL UNIQUE,
    city       TEXT    NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    qty        INTEGER NOT NULL,
    amount     REAL    NOT NULL,
    status     TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
"""

# (id, 名称, 品类, 单价, 库存)
PRODUCTS = [
    (1, "机械键盘", "数码", 399.0, 42),
    (2, "无线鼠标", "数码", 129.0, 130),
    (3, "27 寸显示器", "数码", 1299.0, 8),
    (4, "电饭煲", "家电", 299.0, 26),
    (5, "空气净化器", "家电", 899.0, 5),
    (6, "护眼台灯", "家电", 159.0, 61),
    (7, "Python 编程", "图书", 89.0, 200),
    (8, "算法导论", "图书", 128.0, 74),
    (9, "数据库系统概念", "图书", 96.0, 3),
    (10, "冲锋衣", "服饰", 459.0, 33),
    (11, "运动鞋", "服饰", 329.0, 58),
    (12, "双肩包", "服饰", 259.0, 12),
]

CITIES = ["西安", "北京", "上海", "成都", "广州"]
STATUS = ["paid", "paid", "paid", "paid", "refunded", "pending", "paid"]
BASE_DAY = datetime(2026, 9, 1, 9, 0, 0)


def _build_orders() -> list[tuple]:
    """60 笔订单，固定种子 + 固定基准日，保证每次生成同一份数据。"""
    rng = random.Random(20261102)
    rows = []
    for index in range(1, 61):
        product = rng.choice(PRODUCTS)
        qty = rng.randint(1, 3)
        amount = round(product[3] * qty, 2)
        rows.append(
            (
                index,
                f"SO2026{index:04d}",
                rng.choice(CITIES),
                product[0],
                qty,
                amount,
                rng.choice(STATUS),
                (BASE_DAY + timedelta(days=rng.randint(0, 60))).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    return rows


def build(path: Path = DB_PATH) -> Path:
    """重建演示库，返回文件路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", PRODUCTS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _build_orders())
        conn.commit()
    finally:
        conn.close()
    return path


if __name__ == "__main__":
    target = build()
    size = target.stat().st_size
    print(f"已生成 {target}（{size} 字节）")
    print(f"商品 {len(PRODUCTS)} 条，订单 {len(_build_orders())} 条")
