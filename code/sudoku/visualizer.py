"""4x4 数独最小冲突搜索的本地 Web 可视化界面。

本文件只负责展示 `local_search.min_conflicts_steps` 产生的事件，
不会自行计算冲突、选择候选交换或修改棋盘。算法给出的
棋盘、候选分数和执行交换会被原样展示。

运行：
    python code/sudoku/visualizer.py
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional
from urllib.parse import urlparse

try:
    from .local_search import get_conflicts, min_conflicts_steps
except ImportError:
    from local_search import get_conflicts, min_conflicts_steps


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>4x4 数独最小冲突可视化</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef2f7;
      --panel: #ffffff;
      --text: #172033;
      --muted: #58657a;
      --border: #cbd4e1;
      --strong: #27364d;
      --normal: #f8fafc;
      --conflict: #ffd9d9;
      --selected: #ff9696;
      --candidate: #fff0ad;
      --best: #bdebc6;
      --swapped: #bad9ff;
      --primary: #315ee8;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
    }
    main { max-width: 1050px; margin: 24px auto; padding: 0 18px 30px; }
    h1 { margin: 0 0 16px; font-size: 25px; }
    .guide { margin: 0 0 18px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    .guide summary { padding: 13px 16px; cursor: pointer; color: var(--strong); font-weight: 700; }
    .guide ol { margin: 0; padding: 0 22px 14px 39px; color: var(--muted); line-height: 1.65; font-size: 14px; }
    .guide li + li { margin-top: 4px; }
    .layout { display: grid; grid-template-columns: 430px 1fr; gap: 18px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
    .board {
      width: 398px; height: 398px; display: grid;
      grid-template-columns: repeat(4, 1fr); grid-template-rows: repeat(4, 1fr);
      border: 3px solid var(--strong); background: var(--strong);
    }
    .cell {
      position: relative; display: flex; align-items: center; justify-content: center;
      background: var(--normal); border: 1px solid var(--border);
      color: var(--text); font-size: 34px; font-weight: 700;
    }
    .cell:nth-child(2), .cell:nth-child(6), .cell:nth-child(10), .cell:nth-child(14) { border-right: 3px solid var(--strong); }
    .cell:nth-child(n+5):nth-child(-n+8) { border-bottom: 3px solid var(--strong); }
    .cell.conflict { background: var(--conflict); }
    .cell.selected { background: var(--selected); outline: 4px solid var(--danger); outline-offset: -5px; }
    .cell.candidate { background: var(--candidate); }
    .cell.best { background: var(--best); }
    .cell.swapped { background: var(--swapped); }
    .conflict-count { position: absolute; bottom: 7px; font-size: 12px; font-weight: 500; color: var(--muted); }
    .controls { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 14px; }
    button {
      border: 1px solid var(--border); border-radius: 8px; background: var(--panel);
      color: var(--text); padding: 9px 14px; font-size: 14px; cursor: pointer;
    }
    button.primary { background: var(--primary); border-color: var(--primary); color: white; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .status { margin-top: 13px; min-height: 48px; line-height: 1.55; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 14px; }
    .metric { background: #f6f8fb; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 3px; }
    .metric strong { font-size: 18px; }
    h2 { font-size: 16px; margin: 16px 0 8px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--border); text-align: center; font-size: 14px; }
    th { color: var(--muted); font-weight: 600; }
    tr.chosen { background: var(--best); }
    .log { height: 190px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; padding: 9px; background: #fbfcfe; }
    .log div { padding: 5px 2px; border-bottom: 1px dashed var(--border); font-size: 13px; line-height: 1.45; }
    .error { color: var(--danger); white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12px; }
    .legend { margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.8; }
    .dot { display: inline-block; width: 12px; height: 12px; margin: 0 4px 0 10px; border: 1px solid var(--border); vertical-align: -1px; }
    @media (max-width: 850px) {
      .layout { grid-template-columns: 1fr; }
      .board { width: min(398px, 90vw); height: min(398px, 90vw); }
    }
  </style>
</head>
<body>
<main>
  <h1>4x4 数独最小冲突搜索</h1>
  <details class="guide" open>
    <summary>操作指引</summary>
    <ol>
      <li>启动时会先生成每行均为 1-4 随机排列的初始棋盘；点击“新建随机棋盘”可随时重新开始。</li>
      <li>“前进一步”每次只读取一个算法事件，通常按“选中冲突格 → 评估三个同行交换 → 执行交换”循环；可结合颜色、候选表和日志对照代码。</li>
      <li>“自动前进”每秒读取一个事件，再次点击即暂停；找到解或发生异常时会自动停止。</li>
      <li>界面直接展示 <code>local_search.py</code> 产生的棋盘、冲突数和候选评估，不会自动改正算法结果；若代码抛出异常，详细信息会显示在右侧。</li>
    </ol>
  </details>
  <div class="layout">
    <section class="panel">
      <div id="board" class="board" aria-label="4x4 数独棋盘"></div>
      <div class="controls">
        <button id="reset">新建随机棋盘</button>
        <button id="step" class="primary">前进一步</button>
        <button id="auto">自动前进（1秒/步）</button>
      </div>
      <div id="status" class="status"></div>
      <div class="legend">
        <span class="dot" style="background:var(--conflict)"></span>冲突格
        <span class="dot" style="background:var(--selected)"></span>选中格
        <span class="dot" style="background:var(--candidate)"></span>候选
        <span class="dot" style="background:var(--best)"></span>最优
        <span class="dot" style="background:var(--swapped)"></span>已交换
      </div>
    </section>
    <section class="panel">
      <div class="metrics">
        <div class="metric"><span>当前阶段</span><strong id="phase">-</strong></div>
        <div class="metric"><span>当前总冲突</span><strong id="total">-</strong></div>
        <div class="metric"><span>可视化微步骤</span><strong id="micro">0</strong></div>
        <div class="metric"><span>交换 / 重启</span><strong id="counts">0 / 0</strong></div>
      </div>
      <h2>候选交换</h2>
      <table>
        <thead><tr><th>同行交换</th><th>交换后总冲突</th><th>判定</th></tr></thead>
        <tbody id="candidates"><tr><td colspan="3">尚未评估</td></tr></tbody>
      </table>
      <h2>执行日志</h2>
      <div id="log" class="log"></div>
      <div id="error" class="error"></div>
    </section>
  </div>
</main>
<script>
  let state = null;
  let timer = null;

  async function request(path, method = 'GET') {
    const response = await fetch(path, { method });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  }

  function sameCoord(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a[0] === b[0] && a[1] === b[1];
  }

  function candidateFor(row, col) {
    if (!state.selected || row !== state.selected[0]) return null;
    return (state.candidates || []).find(item => item.target_col === col) || null;
  }

  function renderBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';
    const values = state.board || Array.from({length: 4}, () => [0,0,0,0]);
    const conflicts = state.cell_conflicts || Array.from({length: 4}, () => [0,0,0,0]);
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        const coord = [row, col];
        const candidate = candidateFor(row, col);
        if ((state.last_swap || []).some(item => sameCoord(item, coord))) cell.classList.add('swapped');
        else if (sameCoord(state.selected, coord)) cell.classList.add('selected');
        else if (candidate) cell.classList.add(candidate.is_best ? 'best' : 'candidate');
        else if (conflicts[row] && conflicts[row][col] > 0) cell.classList.add('conflict');
        cell.textContent = values[row][col];
        const count = document.createElement('span');
        count.className = 'conflict-count';
        count.textContent = `冲突 ${conflicts[row] ? conflicts[row][col] : '?'}`;
        cell.appendChild(count);
        board.appendChild(cell);
      }
    }
  }

  function renderCandidates() {
    const body = document.getElementById('candidates');
    body.innerHTML = '';
    const candidates = state.candidates || [];
    if (!candidates.length || !state.selected) {
      body.innerHTML = '<tr><td colspan="3">尚未评估</td></tr>';
      return;
    }
    for (const candidate of candidates) {
      const row = document.createElement('tr');
      if (candidate.target_col === state.chosen_target_col) row.className = 'chosen';
      const judgment = candidate.target_col === state.chosen_target_col
        ? '将执行'
        : (candidate.is_best ? '最优' : '');
      row.innerHTML = `<td>列 ${state.selected[1] + 1} ↔ ${candidate.target_col + 1}</td>`
        + `<td>${candidate.total_conflicts}</td><td>${judgment}</td>`;
      body.appendChild(row);
    }
  }

  function render() {
    renderBoard();
    renderCandidates();
    document.getElementById('phase').textContent = state.phase || '-';
    document.getElementById('total').textContent = state.total_conflicts ?? '?';
    document.getElementById('micro').textContent = state.micro_steps ?? 0;
    document.getElementById('counts').textContent = `${state.swaps ?? 0} / ${state.restart ?? 0}`;
    document.getElementById('status').textContent = state.message || '';
    document.getElementById('error').textContent = state.error || '';
    document.getElementById('step').disabled = state.finished || state.phase === 'error';
    if (state.finished || state.phase === 'error') stopAuto();
  }

  function appendLog() {
    const log = document.getElementById('log');
    const line = document.createElement('div');
    line.textContent = `${state.micro_steps}. [${state.phase}] ${state.message}`;
    log.appendChild(line);
    log.scrollTop = log.scrollHeight;
  }

  async function step() {
    try {
      state = await request('/api/step', 'POST');
      appendLog();
      render();
    } catch (error) {
      stopAuto();
      document.getElementById('error').textContent = error.message;
    }
  }

  async function reset() {
    stopAuto();
    document.getElementById('log').innerHTML = '';
    state = await request('/api/reset', 'POST');
    appendLog();
    render();
  }

  function stopAuto() {
    if (timer) clearInterval(timer);
    timer = null;
    document.getElementById('auto').textContent = '自动前进（1秒/步）';
  }

  function toggleAuto() {
    if (timer) { stopAuto(); return; }
    document.getElementById('auto').textContent = '暂停自动前进';
    step();
    timer = setInterval(step, 1000);
  }

  document.getElementById('step').addEventListener('click', step);
  document.getElementById('reset').addEventListener('click', reset);
  document.getElementById('auto').addEventListener('click', toggleAuto);
  reset();
</script>
</body>
</html>
"""


class SearchSession:
    """每次只向分步生成器请求一个事件。"""

    def __init__(
        self,
        generator_factory: Callable[..., Iterator[Dict]] = min_conflicts_steps,
        seed: Optional[int] = None,
    ) -> None:
        self.generator_factory = generator_factory
        self.seed = seed
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> Dict:
        with getattr(self, "lock", threading.Lock()):
            self.rng = random.Random(self.seed)
            self.iterator = iter(self.generator_factory(max_steps=200, max_restarts=20, rng=self.rng))
            self.micro_steps = 0
            self.state = self._advance_unlocked(initial=True)
            return self.state

    def _validate(self, event: Dict) -> Dict:
        # 不对算法结果做正确性判断；缺少的展示字段沿用上一步状态。
        previous = getattr(self, "state", {})
        if isinstance(event, dict):
            merged = dict(previous)
            merged.update(event)
            event = merged
        else:
            raw_output = event
            event = dict(previous)
            event["phase"] = "output"
            event["message"] = str(raw_output)

        board = event.get("board", [[0] * 4 for _ in range(4)])

        # 单格冲突同样直接使用 get_conflicts 的输出，界面不自行修正。
        event = dict(event)
        event["cell_conflicts"] = [
            [get_conflicts(board, row, col) for col in range(4)]
            for row in range(4)
        ]
        event.setdefault("selected", None)
        event.setdefault("candidates", [])
        event.setdefault("chosen_target_col", None)
        event.setdefault("last_swap", None)
        event.setdefault("swaps", 0)
        event.setdefault("restart", 0)
        event.setdefault("solved", False)
        event["micro_steps"] = self.micro_steps
        event["finished"] = bool(event.get("solved")) or event.get("phase") in {"failed", "error"}
        return event

    def _error_state(self) -> Dict:
        detail = traceback.format_exc()
        return {
            "phase": "error",
            "board": [[0] * 4 for _ in range(4)],
            "cell_conflicts": [[0] * 4 for _ in range(4)],
            "total_conflicts": "?",
            "message": "代码运行时抛出异常，程序无法继续执行。",
            "error": detail,
            "selected": None,
            "candidates": [],
            "chosen_target_col": None,
            "last_swap": None,
            "swaps": 0,
            "restart": 0,
            "solved": False,
            "finished": True,
            "micro_steps": self.micro_steps,
        }

    def _advance_unlocked(self, initial: bool = False) -> Dict:
        try:
            event = next(self.iterator)
            if not initial:
                self.micro_steps += 1
            return self._validate(event)
        except StopIteration:
            return {
                "phase": "finished",
                "board": self.state.get("board", [[0] * 4 for _ in range(4)]) if hasattr(self, "state") else [[0] * 4 for _ in range(4)],
                "cell_conflicts": self.state.get("cell_conflicts", [[0] * 4 for _ in range(4)]) if hasattr(self, "state") else [[0] * 4 for _ in range(4)],
                "total_conflicts": self.state.get("total_conflicts", "?") if hasattr(self, "state") else "?",
                "message": "分步运行已结束。",
                "error": "",
                "selected": None,
                "candidates": [],
                "chosen_target_col": None,
                "last_swap": None,
                "swaps": 0,
                "restart": 0,
                "solved": False,
                "finished": True,
                "micro_steps": self.micro_steps,
            }
        except Exception:
            return self._error_state()

    def advance(self) -> Dict:
        with self.lock:
            if self.state.get("finished"):
                return self.state
            self.state = self._advance_unlocked()
            return self.state


def make_handler(session: SearchSession):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._json(session.state)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/step":
                self._json(session.advance())
            elif path == "/api/reset":
                self._json(session.reset())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="4x4 数独最小冲突可视化")
    parser.add_argument("--port", type=int, default=0, help="监听端口，0 表示自动选择")
    parser.add_argument("--seed", type=int, default=None, help="可选随机种子")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    session = SearchSession(seed=args.seed)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"可视化已启动：{url}")
    print("按 Ctrl+C 停止服务。")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
