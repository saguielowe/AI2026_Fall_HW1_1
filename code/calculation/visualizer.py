"""Browser interface for a no-programming search calculation problem."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from generator import check_answer, generate_practice_question, generate_question, seed_from_student_id


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"


class QuestionStore:
    def __init__(self) -> None:
        self._answers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        mode = payload.get("mode", "practice")
        if mode == "student":
            student_id = str(payload.get("student_id", "")).strip()
            public, answer = generate_question(seed_from_student_id(student_id), f"学号 {student_id}")
        elif mode == "practice":
            public, answer = generate_practice_question()
        else:
            raise ValueError("未知出题方式")
        with self._lock:
            self._answers[public["question_id"]] = answer
            if len(self._answers) > 500:
                oldest = next(iter(self._answers))
                self._answers.pop(oldest, None)
        return public

    def check(self, payload: dict[str, Any]) -> dict[str, Any]:
        question_id = str(payload.get("question_id", ""))
        with self._lock:
            answer = self._answers.get(question_id)
        if answer is None:
            raise ValueError("题目已失效，请重新生成")
        return check_answer(payload.get("answers", {}), answer)


STORE = QuestionStore()


class Handler(BaseHTTPRequestHandler):
    server_version = "SearchCalculationLab/1.0"

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/":
            data = INDEX_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/question":
                self._json(200, STORE.create(payload))
            elif path == "/api/check":
                self._json(200, STORE.check(payload))
            else:
                self._json(404, {"error": "not found"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # keep the page usable and expose a concise message
            self._json(500, {"error": f"服务器异常：{exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="搜索算法计算题交互界面")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"计算题界面已启动：{url}")
    print("按 Ctrl+C 退出。")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

