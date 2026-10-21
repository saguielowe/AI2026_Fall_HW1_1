# 4x4 数独搜索（15分）

## 问题规则

在 4x4 棋盘中填入数字 1-4，使得：

1. 每行包含 1-4，且不能重复；
2. 每列包含 1-4，且不能重复；
3. 每个 2x2 小宫格包含 1-4，且不能重复；
4. 不提供任何初始数字。

## 目录结构

| 文件 | 作用 |
|---|---|
| `common.py` | 棋盘尺寸、数字集合、创建/复制/打印棋盘及结果验证 |
| `backtracking.py` | 回溯搜索 |
| `local_search.py` | 最小冲突局部搜索及分步事件 |
| `visualizer.py` | 只负责展示分步事件的本地浏览器界面 |
| `test_sudoku.py` | 约束、回溯和局部搜索测试 |
| `test_visualizer_model.py` | 分步事件和可视化数据连接测试 |

## 小问1：回溯搜索

初始状态为全空棋盘，`0` 表示空格：

```text
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
```

代码中已提供：

- `is_valid(board, row, col, value)`：检查行、列和 2x2 小宫格约束；
- `find_empty(board)`：按行优先顺序找到第一个空格；
- `backtrack(board, solutions)` 的终止条件、解的深拷贝和候选数字循环。

你只需要补全 `backtrack(board, solutions)` 中标记为 `TODO` 的关键回溯步骤：判断候选数字是否合法、填入后递归，以及递归返回后恢复空格。

不考虑对称变换和数字重命名的等价关系时，4x4 数独共有 288 个完整解。

## 小问2：最小冲突局部搜索

局部搜索需要完成：

- `get_conflicts(board, row, col)`：计算当前格子在同列和同宫格中的冲突数；
- `min_conflicts_steps(max_steps, max_restarts, rng)`：执行最小冲突搜索，并产生分步事件。

`min_conflicts_steps()` 的重启循环、步数循环和事件字典已经搭好；你只需补全
标注为 `TODO 1` 至 `TODO 4` 的冲突格选择、候选交换评估、最优候选选择和执行交换。

搜索规则：

1. 初始化时，每一行都是 `1, 2, 3, 4` 的随机排列；
2. 从所有有冲突的格子中随机选择一个；
3. 唯一允许的修正操作是交换同一行中的两个数字；
4. 分别计算三个候选交换后的总冲突数；
5. 在总冲突最少的候选中随机选择一个执行；
6. 达到步数上限仍未求解时，重新随机初始化。

这里的“总冲突数”是棋盘中不同冲突对的数量。同一对格子不能重复计数。由于
`get_conflicts(board, row, col)` 会分别从冲突对的两个格子各统计一次，可以用：

```python
total_conflicts = sum(
    get_conflicts(board, row, col)
    for row in range(SIZE)
    for col in range(SIZE)
) // 2
```

`min_conflicts()` 是已提供的包装函数，它与可视化界面共同使用 `min_conflicts_steps()` 产生的结果。

### 分步事件

`min_conflicts_steps()` 使用 `yield` 产生字典，其中三个核心阶段为：

| `phase` | 含义 |
|---|---|
| `select` | 选中一个冲突格 |
| `evaluate` | 计算三个候选交换的总冲突 |
| `execute` | 执行一个最优交换 |

每个事件至少应包含 `phase`、`board`、`total_conflicts` 和 `message`。为了让可视化界面完整显示选择、评估和交换过程，请按下列格式增加对应字段。

**每次 `yield` 时都必须保存棋盘的独立快照。** 请使用已经提供的 `copy_board(board)`，不要直接将后续还会修改的 `board` 对象放入事件，否则之前产生的事件也会随着后续交换而改变。

```python
# 初始化或重启
{
    "phase": "initialize", "board": copy_board(board),
    "total_conflicts": total_conflicts, "message": message,
    "restart": restart, "swaps": swaps,
}

# 选中冲突格
{
    "phase": "select", "board": copy_board(board),
    "total_conflicts": total_conflicts, "message": message,
    "selected": [row, col], "restart": restart, "swaps": swaps,
}

# 评估三个同行交换候选
{
    "phase": "evaluate", "board": copy_board(board),
    "total_conflicts": total_conflicts, "message": message,
    "selected": [row, col],
    "candidates": [
        {"target_col": other_col, "total_conflicts": value, "is_best": is_best},
        # 共三项
    ],
    "chosen_target_col": chosen_col,
    "restart": restart, "swaps": swaps,
}

# 执行交换
{
    "phase": "execute", "board": copy_board(board),
    "total_conflicts": total_conflicts, "message": message,
    "selected": [row, col], "chosen_target_col": chosen_col,
    "last_swap": [[row, col], [row, chosen_col]],
    "restart": restart, "swaps": swaps,
}
```

找到解时，任意阶段的事件均可增加 `"solved": True`。达到所有重启上限仍未找到解时，应产生 `phase` 为 `failed` 的事件。可视化界面忠实显示算法输出，不会修正计算结果。

## 运行

在项目根目录执行：

```bash
python code/sudoku/backtracking.py
python code/sudoku/local_search.py
python code/sudoku/visualizer.py
```

可视化程序会自动打开浏览器。如果没有自动打开，请访问终端中显示的 `http://127.0.0.1:端口`。

## 可视化操作

1. 先完成 `get_conflicts()`，启动 `visualizer.py` 后即可查看随机初始棋盘及其冲突数；完成 `min_conflicts_steps()` 后，才能继续查看选择、评估和交换过程。页面直接读取你的 `get_conflicts()` 和 `min_conflicts_steps()` 输出，不会自动修正结果。
2. 点击“前进一步”依次查看 `select`、`evaluate`、`execute` 事件；棋盘颜色、候选表与执行日志会展示当前事件。
3. 点击“自动前进”可每秒播放一个事件，再次点击可暂停；“新建随机棋盘”会放弃当前过程并重新开始。
4. 若程序因代码异常无法继续，页面右侧会显示 traceback；请根据最后一条异常信息定位。

## 测试

如已安装 pytest：

```bash
python -m pytest code/sudoku/test_sudoku.py code/sudoku/test_visualizer_model.py
```

运行测试前，请先完成[项目根目录中的环境配置](../../README.md)。
