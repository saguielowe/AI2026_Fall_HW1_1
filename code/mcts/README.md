# Homework 1：围棋树搜索

本题只使用围棋环境。开始前请按照[项目根目录的环境配置说明](../../README.md)安装依赖并编译围棋环境。

## 需要完成的算法

1. 补全 `mcts/uct_mcts.py` 中标有 `TODO` 的 UCT MCTS 关键步骤。
2. 补全 `other_algo/alpha_beta_search.py` 中的 3 处 `TODO`：negamax 递归、`value/alpha` 更新和剪枝条件。该实现为带启发式评估的深度限制 Alpha-Beta，<strong>只允许在 5×5 围棋上使用</strong>。

Alpha-Beta 中的“深度”指从根节点选择一次动作后，向下最多搜索的总落子层数；设为 1 时，比较各个根节点动作之后的启发式评估。建议从 1 或 2 开始，深度每增加 1，搜索量会按合法动作数成倍增长。

## 围棋环境约定

- 棋盘从上到下、从左到右编号，落子动作为 `0` 至 `n*n-1`，“停一手”为 `n*n`。
- `action_mask[i] == 1` 表示动作 `i` 合法。`step(action)` 落子后会自动切换行动方。
- `step()` 返回（局面、奖励、是否结束）。胜负的终局奖励属于<strong>刚刚落子的玩家</strong>；和棋始终返回中性标记 `EPS`，未结束时为 `NOTEND`。
- 搜索后继必须用 `fork()` 复制局面，不能用普通赋值共享同一环境。不需要观察时可调用 `step(action, return_obs=False)`。

## MCTS 实现约定

1. `rollout(node)` 返回从 `node.env.current_player`（叶节点当前行动方）视角计算的价值。
2. 终局 `step()` 的胜负 `reward` 属于刚落子的玩家，转换到叶节点当前行动方视角时需取反；和棋的 `EPS` 不取反。
3. `backup(node, value)` 沿叶节点到根节点逐边更新，每向上经过一条边转换一次玩家视角，每条边的访问数每轮恰好加 1。
4. 非终局叶节点独立运行 `n_rollout` 次随机模拟，取算术平均后只执行一次 `backup`。
5. `get_policy()` 将合法动作的访问数归一化，非法动作概率必须为 0。本题不要求温度或噪声。

因此，完成 `n_search` 轮后，根节点各动作访问数之和应为 `n_search`，而不是 `n_search*n_rollout`。

## 对局与实验

作业中的常规对局和对比实验建议直接在可视化界面中完成：

```bash
python code/mcts/visualizer.py
```

页面下方的“批量对局实验”支持：

- 算法 A/B 分别选择 Random、MCTS、启发式 MCTS 或 Alpha-Beta；
- 分别设置双方的 `C`、`n_search`、`n_rollout`、启发式权重/缩放或 Alpha-Beta 深度；
- 固定随机种子，并自动交换执黑、执白；
- 实时显示完成局数、用时、A/B 胜率、不败率、最近一局终局棋盘和落子记录；
- 一键复制结果摘要，便于整理实验报告。

批量实验中的搜索次数 64、模拟次数 7、Alpha-Beta 深度 1 和每种执棋
顺序 10 局，已与作业第 3 小问的基本实验设置对齐；做其他小问时再修改相应参数。

`pit.py` 仍保留为可选的脚本化入口：只有在需要更大规模、自定义循环或额外数据处理时，才需要修改并运行 `python code/mcts/pit.py`。

### 界面操作顺序

**单局交互/调试：**

1. 在页面顶部设置棋盘大小、黑白方算法和参数，然后点击“新对局”。已开始的对局不会因直接修改输入框而改变参数。
2. 人工回合直接点击棋盘合法交叉点，或点击“停一手”。算法回合点击“开始本回合搜索”。
3. 搜索计算完成后，点击“下一搜索事件”逐步调试，或用“自动回放”观察。只有回放到最后的“执行落子”事件后，棋盘才会落子并切换行动方。
4. 若只需要算法完成本回合，点击“跳过本轮搜索细节”。点击“新对局”可在任意时刻取消旧搜索并重新开局。

**批量对比实验：**

1. 在页面下方设置算法 A/B、各自超参、随机种子和每种执棋顺序的对局数。选择算法后，该算法不使用的参数会自动变灰且不可修改。勾选“交换执棋”时，总局数为输入局数的两倍。
2. 点击“开始批量实验”，通过进度条、分执棋表格、最近一局棋盘和落子记录检查结果。
3. 点击“取消实验”后，界面需等待当前搜索返回才会停止，这段时间进度栏会显示“正在等待当前搜索停止”。
4. 完成后可点击“复制结果摘要”。如果算法抛出异常或返回非法动作，界面会中止相关过程并显示异常信息。

### 带评估函数的 MCTS

`other_algo/heuristic.py` 中的 `go_heuristic_evaluation` 返回当前行动方视角的围棋局面评分。`HeuristicUCTPlayer` 已封装好：

```python
heuristic_config = HeuristicUCTMCTSConfig(
    C=1.0, n_search=64, n_rollout=1,
    heuristic_weight=1.0, heuristic_scale=1.0,
)
player = HeuristicUCTPlayer(heuristic_config, deterministic=True)
```

`heuristic_weight=1.0` 时完全替代 rollout；评估值是确定的，因此每个被选中的叶节点只评估一次，
此时 `n_rollout` 不会导致同一评估被重复计算。`0<w<1` 时使用
`w*启发式价值+(1-w)*rollout价值`，`n_rollout` 仍用于控制随机模拟次数。
建议固定棋盘、对手、随机种子和其他 MCTS 参数，分别执黑、执白比较时间、胜率和不败率。

## 单局交互与树搜索调试

```bash
python code/mcts/visualizer.py
```

单局界面可选人工、MCTS、启发式 MCTS 或 Alpha-Beta 棋手。Alpha-Beta 选项在 7×7 和 9×9 上会自动禁用。MCTS 回放依次显示 UCT 选择、扩展、rollout、backup 和根节点策略；Alpha-Beta 回放显示局面展开、启发式评估、边界更新和剪枝。可以逐事件前进、自动回放或跳过本轮细节。任意时刻点击“新对局”都会取消旧搜索，不会覆盖新棋局。

可视化器会调用你在 `mcts/uct_mcts.py` 和 `other_algo/alpha_beta_search.py` 中完成的实现。它不会修正搜索树、节点统计或返回动作；如果代码抛出异常，请查看搜索树右侧或批量实验结果区的 traceback。

## 测试

完成 TODO 并编译围棋环境后，在项目根目录执行：

```bash
python -m pytest code/mcts/test_uct_mcts.py code/mcts/test_heuristic_mcts.py code/mcts/test_alpha_beta.py code/mcts/test_visualizer.py
```

对应 TODO 未完成时，相关测试失败属于正常现象。
