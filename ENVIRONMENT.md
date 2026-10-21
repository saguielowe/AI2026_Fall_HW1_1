# 本地运行环境

已在项目目录内配置 Python 3.11 独立环境 `.venv`，基础 Python 位于 `.environment/python`。
不要单独移动或删除其中一个目录。原有 Anaconda 环境没有安装或修改任何包。

## 启动

在项目根目录的 PowerShell 中执行，不需要激活环境或修改执行策略：

```powershell
.\start-homework.cmd calculation
.\start-homework.cmd sudoku
.\start-homework.cmd mcts
```

每条命令启动对应的本地网页并自动打开浏览器。停止时在终端按 Ctrl+C。
启动脚本为计算题使用端口 18767，数独和围棋会自动选择空闲端口。
本机在 2026-10-14 的 Windows TCP 保留范围为 8756–8855，包含原默认端口 8767，
绑定时会出现 WinError 10013，因此改用已验证可绑定的端口。
若直接运行计算题源文件，请同时指定端口：

```powershell
.\.venv\Scripts\python.exe code/calculation/visualizer.py --port 18767
```

也可以直接使用项目解释器，例如：

```powershell
.\.venv\Scripts\python.exe code/mcts/visualizer.py
```

## 检查与重新编译

```powershell
.\start-homework.cmd check
.\start-homework.cmd build
```

`check` 只检查依赖、5×5/7×7 棋盘的基础操作以及三个网页入口和初始状态接口；
临时网页服务会在检查后关闭。它不运行作答、搜索求解或棋手对比实验。

`build` 调用原有 `setup.py`，为编译进程设置当前 Visual Studio 和 Windows SDK 的工具路径。
这解决了本机编译时 `cl.exe` / `rc.exe` 搜索路径问题，不改变全局 PATH。
编译会自动更新 Cython 生成的 `goboard.cpp` 并生成 `.pyd`、`build` 等产物。
手写代码及所有 TODO 均保持原样。

依赖版本记录在 `.environment/requirements-lock.txt`，本次验收结果在 `.environment/verification.txt`。

数独和围棋的网页只展示你的算法输出。TODO 尚未完成时，报错、算法测试失败或无法继续搜索
属于待完成作业的行为，不能据此判定 Python 环境安装失败。

## 提交时

按作业说明打包代码和统一 PDF 报告；不要把 `.venv`、`.environment`、`__pycache__`、
`build`、`.pyd` 或 `.so` 放入提交包。本文件及启动脚本是本地辅助文件，也无需提交。
