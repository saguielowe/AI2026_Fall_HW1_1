"""Check dependencies, the Go extension, and HTTP entry points without solving homework."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
for name in ("numpy", "tqdm", "cython", "matplotlib", "pytest", "setuptools", "pip"):
    print(f"{name}: {importlib.metadata.version(name)}", flush=True)
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

sys.path.insert(0, str(root / "code" / "mcts"))
from env.go.go_env import GoGame
from env.go import goboard

for size in (5, 7):
    game = GoGame(size)
    observation = game.reset()
    assert observation.shape == (size, size)
    assert game.action_mask.shape == (size * size + 1,)
    child = game.fork()
    child.step(0)
    assert game.observation[0, 0] == 0
    assert child.observation[0, 0] != 0
    assert child.current_player == -game.current_player
print(f"Go extension OK: {goboard.__file__}", flush=True)
print("5x5 / 7x7 reset, legal mask, step and independent fork OK", flush=True)

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for module in ("calculation", "sudoku", "mcts"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    log_path = root / ".environment" / f"check-{module}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(root / "code" / module / "visualizer.py"),
             "--no-browser", "--port", str(port)],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{module} startup failed; see {log_path}")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/", timeout=1) as response:
                        body = response.read().decode("utf-8")
                        assert response.status == 200 and "<html" in body.lower()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"{module} HTTP check timed out; see {log_path}")
                    time.sleep(0.1)
            print(f"{module}: HTTP 200, HTML served", flush=True)
            if module in ("sudoku", "mcts"):
                with opener.open(f"http://127.0.0.1:{port}/api/state", timeout=3) as response:
                    state = json.load(response)
                print(f"{module}: initial state API OK", flush=True)
                if module == "sudoku" and state.get("phase") == "error":
                    print("Sudoku reports an algorithm error; TODOs are intentionally incomplete.", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

originals = json.loads((root / ".environment" / "original-code-hashes.json").read_text(encoding="utf-8-sig"))
changed = []
for entry in originals:
    path = Path(entry["Path"])
    if hashlib.sha256(path.read_bytes()).hexdigest().upper() != entry["Hash"]:
        changed.append(path.relative_to(root).as_posix())
allowed = {"code/mcts/env/go/goboard.cpp"}
assert set(changed) <= allowed, f"Unexpected source changes: {changed}"
print("Original Python / Cython / handwritten C++ files unchanged.", flush=True)
print(f"Regenerated build source: {changed}", flush=True)
print("Environment checks passed; no homework answers or algorithm experiments were run.", flush=True)
