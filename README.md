# AI2026 第一次作业：Python 环境配置

请在开始作业前完成本文档中的所有步骤。**MCTS 部分必须编译棋类环境**，否则无法正常运行棋局、测试搜索算法或使用可视化调试。

## 1. 准备 Python 环境

推荐使用 **Python 3.10 或 Python 3.11**，并为本次作业创建独立环境。下面两种方式选择一种即可。

### 方式一：使用 Anaconda/Miniconda（推荐）

在终端中进入本项目根目录，执行：

```bash
conda create -n ai2026_hw1 python=3.11 -y
conda activate ai2026_hw1
python -m pip install --upgrade pip
python -m pip install -r code/mcts/requirements.txt
python -m pip install pytest
```

### 方式二：使用 Python 自带的 venv

在项目根目录执行：

```bash
python -m venv .venv
```

激活环境：

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

如果 Windows PowerShell 提示“禁止运行脚本”，可仅为当前终端临时放行后再激活：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r code/mcts/requirements.txt
python -m pip install pytest
```

建议始终使用 `python -m pip` 安装依赖，避免将依赖安装到其他 Python 环境中。

## 2. 编译 MCTS 棋类环境（必须）

MCTS 部分的围棋底层棋盘由 C++/Cython 实现，因此需要先安装 C/C++ 编译器：

- Windows：安装 Visual Studio Build Tools，并选择“使用 C++ 的桌面开发”。
- macOS：在终端执行 `xcode-select --install`。
- Linux：安装 GCC/G++。Ubuntu 可执行 `sudo apt install build-essential python3-dev`。

确保已激活上一节创建的 Python 环境，然后从项目根目录执行：

```bash
cd code/mcts/env/go
python setup.py build_ext --inplace

cd ../../../..
```

编译成功后，相应目录中会生成 `.so` 文件（Windows 下为 `.pyd` 文件）。编译过程中出现普通 warning 通常不影响使用；若编译失败，请将完整错误信息发给助教。

> 项目中即使已经存在 `.so` 或 `.pyd` 文件，它也可能是在其他操作系统或 Python 版本下编译的，不一定能在你的电脑上使用。请在当前 Python 环境中完成编译；如遇到扩展模块无法导入，请删除原有的 `.so`/`.pyd` 文件和 `build` 目录后重新编译。

## 3. 检查环境

如果电脑上同时安装了多个 Python，请执行：

```bash
python --version
python -m pip --version
```

两条命令显示的 Python 路径应当属于同一个虚拟环境。各题的启动和测试命令请查看对应 `code` 子目录中的 README。
