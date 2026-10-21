"""Interactive browser visualizer for Go, UCT MCTS, and Alpha-Beta.

This file deliberately does not contain a second MCTS implementation.  It calls
``UCTPlayer.play`` and temporarily wraps the methods of the current
``UCTMCTS`` object so that the real selection, rollout, backup, and policy
results can be replayed in the browser.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
import types
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from env.base_env import BLACK, DRAW, EPS, WHITE
from env.go.go_env import GoGame
from mcts.heuristic_mcts import HeuristicUCTMCTSConfig
from mcts.uct_mcts import UCTMCTSConfig
from players.alpha_beta_player import AlphaBetaPlayer
from players.heuristic_uct_player import HeuristicUCTPlayer
from players.random_player import RandomPlayer
from players.uct_player import UCTPlayer


MAX_TREE_NODES = 180
MAX_TREE_DEPTH = 6
MAX_ALPHA_BETA_EVENTS = 160
MAX_ALPHA_BETA_TREE_NODES = 100


class SearchCancelled(Exception):
    """Internal control-flow signal for a search invalidated by a new game."""


def _plain(value: Any) -> Any:
    """Convert NumPy-rich values into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _action_label(game_kind: str, size: int, action: int | None) -> str:
    if action is None:
        return "根节点"
    if game_kind == "go" and action == size * size:
        return "停一手"
    row, col = divmod(int(action), size)
    letters = "ABCDEFGHJKLMNOPQRSTUV"
    return f"{letters[col]}{size - row}"


class TraceRecorder:
    """Record actual calls made by one UCTMCTS.search invocation."""

    def __init__(
        self,
        mcts: Any,
        game_kind: str,
        size: int,
        progress_callback: Any = None,
        cancel_event: threading.Event | None = None,
    ):
        self.mcts = mcts
        self.game_kind = game_kind
        self.size = size
        self.events: list[dict[str, Any]] = []
        self._node_ids: dict[int, int] = {}
        self._next_node_id = 0
        self._originals: dict[str, Any] = {}
        self._last_backup_stats: dict[int, tuple[int, float]] = {}
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event

    def check_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise SearchCancelled

    def report_progress(self, phase: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(phase)

    def node_id(self, node: Any) -> int | None:
        if node is None:
            return None
        key = id(node)
        if key not in self._node_ids:
            self._node_ids[key] = self._next_node_id
            self._next_node_id += 1
        return self._node_ids[key]

    def path(self, node: Any) -> list[int]:
        path: list[int] = []
        seen: set[int] = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            node_id = self.node_id(node)
            if node_id is not None:
                path.append(node_id)
            node = getattr(node, "parent", None)
        return list(reversed(path))

    def candidates(self, node: Any) -> list[dict[str, Any]]:
        if node is None:
            return []
        visits = np.asarray(getattr(node, "child_N_visit", []))
        totals = np.asarray(getattr(node, "child_V_total", []))
        mask = np.asarray(getattr(node, "action_mask", []))
        rows = []
        for action in range(min(len(visits), len(totals), len(mask))):
            if not bool(mask[action]):
                continue
            n_visit = int(visits[action])
            value_total = float(totals[action])
            rows.append(
                {
                    "action": action,
                    "label": _action_label(self.game_kind, self.size, action),
                    "n": n_visit,
                    "w": value_total,
                    "q": value_total / n_visit if n_visit else None,
                    "expanded": bool(getattr(node, "has_child", lambda _a: False)(action)),
                }
            )
        return rows

    def tree(self, focus: Any = None) -> dict[str, Any]:
        root = getattr(self.mcts, "root", None)
        if root is None:
            return {"nodes": [], "edges": [], "path": []}

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, int]] = []
        queue = deque([(root, 0)])
        seen: set[int] = set()
        while queue and len(nodes) < MAX_TREE_NODES:
            node, depth = queue.popleft()
            if id(node) in seen:
                continue
            seen.add(id(node))
            node_id = self.node_id(node)
            parent = getattr(node, "parent", None)
            action = getattr(node, "action", None)
            if parent is None or action is None:
                n_visit = int(np.asarray(getattr(node, "child_N_visit", [])).sum())
                # Root has no incoming edge.  Its aggregate statistics are the
                # sum of all outgoing root-edge statistics.
                value_total = float(np.asarray(getattr(node, "child_V_total", [])).sum())
            else:
                parent_visits = np.asarray(getattr(parent, "child_N_visit", []))
                parent_values = np.asarray(getattr(parent, "child_V_total", []))
                n_visit = int(parent_visits[action]) if action < len(parent_visits) else 0
                value_total = float(parent_values[action]) if action < len(parent_values) else 0.0
            nodes.append(
                {
                    "id": node_id,
                    "depth": depth,
                    "action": action,
                    "label": _action_label(self.game_kind, self.size, action),
                    "n": n_visit,
                    "w": value_total,
                    "q": value_total / n_visit if n_visit else None,
                    "done": bool(getattr(node, "done", False)),
                }
            )
            if depth >= MAX_TREE_DEPTH:
                continue
            children = getattr(node, "children", {})
            for child_action, child in sorted(children.items(), key=lambda item: int(item[0])):
                child_id = self.node_id(child)
                edges.append({"source": node_id, "target": child_id})
                queue.append((child, depth + 1))

        kept = {node["id"] for node in nodes}
        edges = [edge for edge in edges if edge["source"] in kept and edge["target"] in kept]
        return {
            "nodes": nodes,
            "edges": edges,
            "path": [node_id for node_id in self.path(focus) if node_id in kept],
            "truncated": bool(queue),
        }

    def add_event(
        self,
        phase: str,
        title: str,
        message: str,
        *,
        focus: Any = None,
        tree: bool = False,
        candidates: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> None:
        event: dict[str, Any] = {
            "phase": phase,
            "title": title,
            "message": message,
            "focus": self.node_id(focus),
            "path": self.path(focus),
        }
        if tree:
            event["tree"] = self.tree(focus)
        if candidates is not None:
            event["candidates"] = candidates
        event.update(_plain(extra))
        self.events.append(event)

    def _wrap(self, name: str, after: Any) -> None:
        if not hasattr(self.mcts, name):
            return
        original = getattr(self.mcts, name)
        self._originals[name] = original

        def wrapped(_instance: Any, *args: Any, **kwargs: Any) -> Any:
            self.check_cancelled()
            result = original(*args, **kwargs)
            self.check_cancelled()
            after(args, kwargs, result)
            return result

        setattr(self.mcts, name, types.MethodType(wrapped, self.mcts))

    def install(self) -> None:
        self.add_event(
            "start",
            "开始搜索",
            "调用现有 UCTMCTS.search()。",
            focus=getattr(self.mcts, "root", None),
            tree=True,
        )

        def after_select(args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
            node = args[0] if args else None
            self.report_progress("selection")
            self.add_event(
                "select-action",
                "UCT 选择动作",
                f"当前节点返回动作 {_action_label(self.game_kind, self.size, int(result))}。",
                focus=node,
                candidates=self.candidates(node),
                selected_action=int(result),
            )

        def after_pick(_args: tuple[Any, ...], _kwargs: dict[str, Any], leaf: Any) -> None:
            self.report_progress("expansion")
            self.add_event(
                "select-leaf",
                "选择 / 扩展叶节点",
                f"本轮到达 {_action_label(self.game_kind, self.size, getattr(leaf, 'action', None))}。",
                focus=leaf,
                tree=True,
                board=np.asarray(leaf.env.observation),
            )

        def after_rollout(args: tuple[Any, ...], _kwargs: dict[str, Any], value: Any) -> None:
            node = args[0] if args else None
            self.report_progress("rollout")
            self.add_event(
                "rollout",
                "随机模拟",
                f"rollout 返回价值 {float(value):.4f}。",
                focus=node,
                value=float(value),
            )

        def after_backup(args: tuple[Any, ...], _kwargs: dict[str, Any], _result: Any) -> None:
            node = args[0] if args else None
            value = float(args[1]) if len(args) > 1 else None
            self.report_progress("backup")
            snapshot = self.tree(node)
            path_ids = set(self.path(node))
            changes = []
            current_stats: dict[int, tuple[int, float]] = {}
            for item in snapshot["nodes"]:
                node_id = int(item["id"])
                now = (int(item["n"]), float(item["w"]))
                current_stats[node_id] = now
                if node_id not in path_ids:
                    continue
                previous = self._last_backup_stats.get(node_id, (0, 0.0))
                changes.append(
                    {
                        "id": node_id,
                        "delta_n": now[0] - previous[0],
                        "delta_w": now[1] - previous[1],
                    }
                )
            self.add_event(
                "backup",
                "回传更新",
                f"沿高亮路径更新 {len(path_ids)} 个节点的访问次数与累计价值。",
                focus=node,
                value=value,
                backup_changes=changes,
            )
            self.events[-1]["tree"] = snapshot
            self._last_backup_stats = current_stats

        def after_policy(args: tuple[Any, ...], _kwargs: dict[str, Any], result: Any) -> None:
            node = args[0] if args and args[0] is not None else getattr(self.mcts, "root", None)
            self.add_event(
                "policy",
                "生成落子策略",
                "根据根节点各动作的访问次数生成策略。",
                focus=node,
                tree=True,
                candidates=self.candidates(node),
                policy=np.asarray(result),
            )

        self._wrap("uct_action_select", after_select)
        self._wrap("pick_leaf", after_pick)
        self._wrap("rollout", after_rollout)
        self._wrap("backup", after_backup)
        self._wrap("get_policy", after_policy)

    def restore(self) -> None:
        for name, original in self._originals.items():
            setattr(self.mcts, name, original)
        self._originals.clear()


def _bound_label(value: Any) -> str:
    value = float(value)
    if np.isneginf(value):
        return "−∞"
    if np.isposinf(value):
        return "+∞"
    return f"{value:.3f}"


class AlphaBetaTraceRecorder:
    """Turn events emitted by the actual Alpha-Beta code into replay frames."""

    def __init__(
        self,
        game_kind: str,
        size: int,
        algorithm: str,
        cancel_event: threading.Event | None = None,
    ):
        self.game_kind = game_kind
        self.size = size
        self.algorithm = algorithm
        self.events: list[dict[str, Any]] = []
        self.nodes: dict[int, dict[str, Any]] = {}
        self.edges: list[dict[str, int]] = []
        self.parents: dict[int, int | None] = {}
        self.candidates: dict[int, dict[int, dict[str, Any]]] = {}
        self.omitted_events = 0
        self.cancel_event = cancel_event

    def _path(self, node_id: int | None) -> list[int]:
        path: list[int] = []
        seen: set[int] = set()
        while node_id is not None and node_id not in seen:
            seen.add(node_id)
            path.append(node_id)
            node_id = self.parents.get(node_id)
        return list(reversed(path))

    def _tree(self, focus: int | None) -> dict[str, Any]:
        visible = list(self.nodes.values())[:MAX_ALPHA_BETA_TREE_NODES]
        kept = {node["id"] for node in visible}
        return {
            "nodes": [dict(node) for node in visible],
            "edges": [
                dict(edge)
                for edge in self.edges
                if edge["source"] in kept and edge["target"] in kept
            ],
            "path": [node for node in self._path(focus) if node in kept],
            "truncated": len(self.nodes) > len(visible),
            "algorithm": "alpha-beta",
        }

    def _append(self, phase: str, title: str, message: str, node_id: int | None, **extra: Any) -> None:
        if len(self.events) >= MAX_ALPHA_BETA_EVENTS and phase not in {"search-end", "move", "error"}:
            self.omitted_events += 1
            return
        event = {
            "phase": phase,
            "title": title,
            "message": message,
            "focus": node_id,
            "path": self._path(node_id),
            "tree": self._tree(node_id),
            "algorithm": "alpha-beta",
        }
        event.update(_plain(extra))
        self.events.append(event)

    def record(self, raw: dict[str, Any]) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise SearchCancelled
        phase = raw["phase"]
        node_id = int(raw.get("node_id", 0))
        if phase == "node":
            parent_id = raw.get("parent_id")
            action = raw.get("action")
            self.parents[node_id] = None if parent_id is None else int(parent_id)
            self.nodes[node_id] = {
                "id": node_id,
                "depth": int(raw.get("depth", 0)),
                "action": action,
                "label": _action_label(self.game_kind, self.size, action),
                "algorithm": "alpha-beta",
                "value": None,
                "alpha": _bound_label(raw.get("alpha", -np.inf)),
                "beta": _bound_label(raw.get("beta", np.inf)),
                "pruned": False,
            }
            if parent_id is not None:
                self.edges.append({"source": int(parent_id), "target": node_id})
            self._append(
                "expand",
                "展开局面",
                f"进入 {_action_label(self.game_kind, self.size, action)}，读取当前 α、β 边界。",
                node_id,
                board=raw.get("board"),
            )
            return

        node = self.nodes.get(node_id)
        if node is None:
            return
        if phase in {"leaf", "terminal"}:
            node["value"] = float(raw["value"])
            node["alpha"] = _bound_label(raw.get("alpha", -np.inf))
            node["beta"] = _bound_label(raw.get("beta", np.inf))
            title = "启发式评估" if phase == "leaf" else "终局返回"
            message = f"该节点返回价值 {float(raw['value']):.3f}。"
            self._append(phase, title, message, node_id, value=float(raw["value"]))
            return

        if phase == "update":
            action = int(raw["action"])
            value = float(raw["action_value"])
            node["value"] = float(raw["value"])
            node["alpha"] = _bound_label(raw["alpha"])
            node["beta"] = _bound_label(raw["beta"])
            rows = self.candidates.setdefault(node_id, {})
            rows[action] = {
                "algorithm": "alpha-beta",
                "action": action,
                "label": _action_label(self.game_kind, self.size, action),
                "value": value,
                "status": "已检查",
            }
            self._append(
                "bound-update",
                "更新 α / β",
                f"动作 {_action_label(self.game_kind, self.size, action)} 的价值为 {value:.3f}；当前 α={node['alpha']}，β={node['beta']}。",
                node_id,
                candidates=list(rows.values()),
                selected_action=action,
            )
            return

        if phase == "prune":
            node["pruned"] = True
            skipped = [int(action) for action in raw.get("skipped", [])]
            rows = self.candidates.setdefault(node_id, {})
            for action in skipped:
                rows[action] = {
                    "algorithm": "alpha-beta",
                    "action": action,
                    "label": _action_label(self.game_kind, self.size, action),
                    "value": None,
                    "status": "已剪枝",
                }
            labels = "、".join(_action_label(self.game_kind, self.size, action) for action in skipped)
            message = (
                f"满足剪枝条件，跳过后续动作：{labels}。" if skipped else "已达到当前可取的最大价值，结束该节点搜索。"
            )
            self._append(
                "prune",
                "Alpha-Beta 剪枝",
                message,
                node_id,
                candidates=list(rows.values()),
            )
            return

        if phase == "finish":
            action = int(raw["action"])
            suffix = f"（另有 {self.omitted_events} 个中间事件未加入回放）" if self.omitted_events else ""
            self._append(
                "search-end",
                "搜索完成",
                f"{self.algorithm} 选择动作 {_action_label(self.game_kind, self.size, action)}，价值 {float(raw['value']):.3f}。{suffix}",
                node_id,
                selected_action=action,
            )

    def add_move_event(self, action: int) -> None:
        self._append(
            "move",
            "执行落子",
            f"把 {_action_label(self.game_kind, self.size, action)} 应用于真实棋盘。",
            0,
            selected_action=action,
        )


class GameSession:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.progress_lock = threading.Lock()
        self._active_searches: set[threading.Event] = set()
        self.generation = 0
        self._search_progress = {
            "running": False,
            "completed": 0,
            "total": 0,
            "current_rollout": 0,
            "rollouts_per_search": 1,
            "phase": "idle",
            "started_at": None,
        }
        self.game_kind = "go"
        self.size = 5
        self.player_types = {BLACK: "human", WHITE: "mcts"}
        self.config_values = {
            "C": 1.0, "n_search": 32, "n_rollout": 3, "ab_depth": 2,
            "heuristic_weight": 1.0, "heuristic_scale": 1.0,
        }
        self.players: dict[int, Any | None] = {BLACK: None, WHITE: None}
        self.game: Any = None
        self.history: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.trace_index = -1
        self.pending_action: int | None = None
        self.error: str | None = None
        self.new_game({})

    def _set_search_progress(self, generation: int | None = None, **updates: Any) -> None:
        # Uses a separate lock so /api/progress stays responsive while the
        # main game lock is held by a long-running search.
        with self.progress_lock:
            if generation is not None and generation != self.generation:
                return
            self._search_progress.update(updates)

    def search_progress(self) -> dict[str, Any]:
        with self.progress_lock:
            progress = dict(self._search_progress)
        started_at = progress.pop("started_at", None)
        progress["elapsed"] = max(0.0, time.monotonic() - started_at) if started_at else 0.0
        total = int(progress.get("total", 0))
        completed = int(progress.get("completed", 0))
        current_rollout = int(progress.get("current_rollout", 0))
        rollouts_per_search = max(1, int(progress.get("rollouts_per_search", 1)))
        fractional_completed = completed + min(0.99, current_rollout / rollouts_per_search)
        progress["percent"] = min(100.0, 100.0 * fractional_completed / total) if total else 0.0
        return progress

    def _make_player(self, player_type: str) -> Any | None:
        if player_type == "human":
            return None
        if player_type == "alphabeta":
            return AlphaBetaPlayer(depth=int(self.config_values["ab_depth"]))
        if player_type == "heuristic_mcts":
            config = HeuristicUCTMCTSConfig(
                C=float(self.config_values["C"]),
                n_search=int(self.config_values["n_search"]),
                n_rollout=int(self.config_values["n_rollout"]),
                heuristic_weight=float(self.config_values["heuristic_weight"]),
                heuristic_scale=float(self.config_values["heuristic_scale"]),
            )
            return HeuristicUCTPlayer(config, deterministic=True)
        config = UCTMCTSConfig(
            C=float(self.config_values["C"]),
            n_search=int(self.config_values["n_search"]),
            n_rollout=int(self.config_values["n_rollout"]),
        )
        return UCTPlayer(config, deterministic=True)

    def new_game(self, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            game_kind = "go"
            size = int(options.get("size", 5))
            if size not in {5, 7, 9}:
                raise ValueError("围棋棋盘大小必须是 5、7 或 9")

            black_type = str(options.get("black", self.player_types[BLACK]))
            white_type = str(options.get("white", self.player_types[WHITE]))
            allowed_players = {"human", "mcts", "heuristic_mcts", "alphabeta"}
            if black_type not in allowed_players or white_type not in allowed_players:
                raise ValueError("未知的玩家类型")
            if size != 5 and "alphabeta" in {black_type, white_type}:
                raise ValueError("Alpha-Beta 仅允许在 5×5 围棋棋盘上使用")

            config_values = {
                "C": float(options.get("C", self.config_values["C"])),
                "n_search": int(options.get("n_search", self.config_values["n_search"])),
                "n_rollout": int(options.get("n_rollout", self.config_values["n_rollout"])),
                "ab_depth": int(options.get("ab_depth", self.config_values["ab_depth"])),
                "heuristic_weight": float(options.get("heuristic_weight", self.config_values["heuristic_weight"])),
                "heuristic_scale": float(options.get("heuristic_scale", self.config_values["heuristic_scale"])),
            }
            if config_values["n_search"] < 1 or config_values["n_rollout"] < 1:
                raise ValueError("搜索次数和 rollout 次数必须至少为 1")
            if not 1 <= config_values["ab_depth"] <= 3:
                raise ValueError("Alpha-Beta 深度必须在 1 到 3 之间")
            if not 0 <= config_values["heuristic_weight"] <= 1:
                raise ValueError("启发式权重必须在 0 到 1 之间")
            if config_values["heuristic_scale"] <= 0:
                raise ValueError("启发式缩放必须大于 0")

            # Actively stop every older search.  The generation check below is
            # still retained as a second guard against stale result commits.
            for cancel_event in self._active_searches:
                cancel_event.set()
            self._active_searches.clear()
            self.generation += 1
            self.game_kind = game_kind
            self.size = size
            self.player_types = {BLACK: black_type, WHITE: white_type}
            self.config_values = config_values
            self.players = {
                BLACK: self._make_player(black_type),
                WHITE: self._make_player(white_type),
            }
            self.game = GoGame(size)
            self.game.reset()
            self.history = []
            self.trace = []
            self.trace_index = -1
            self.pending_action = None
            self.error = None
            self._set_search_progress(
                running=False,
                completed=0,
                total=config_values["n_search"],
                current_rollout=0,
                rollouts_per_search=config_values["n_rollout"],
                phase="idle",
                started_at=None,
            )
            return self.state()

    def _result_text(self, reward: float, mover: int) -> str:
        if abs(float(reward)) < EPS * 10:
            return "和棋"
        winner = mover if reward > 0 else -mover
        return "黑棋获胜" if winner == BLACK else "白棋获胜"

    def _apply_action(self, action: int, source: str) -> dict[str, Any]:
        mover = int(self.game.current_player)
        _obs, reward, done = self.game.step(int(action))
        self.history.append(
            {
                "player": mover,
                "action": int(action),
                "label": _action_label(self.game_kind, self.size, int(action)),
                "source": source,
            }
        )
        opponent = self.players.get(-mover)
        if opponent is not None and hasattr(opponent, "opp_play"):
            opponent.opp_play(int(action))
        if done:
            self.error = None
            self.history[-1]["result"] = self._result_text(float(reward), mover)
        return self.state()

    def human_move(self, action: int) -> dict[str, Any]:
        with self.lock:
            if self.pending_action is not None:
                raise ValueError("请先完成当前搜索过程的回放")
            if self.game.ended:
                raise ValueError("对局已经结束，请开始新对局")
            current = int(self.game.current_player)
            if self.player_types[current] != "human":
                raise ValueError("当前轮到算法玩家，请先开始搜索")
            if action < 0 or action >= self.game.action_space_size or not self.game.action_mask[action]:
                raise ValueError("该位置当前不能落子")
            self.error = None
            return self._apply_action(action, "human")

    def prepare_search(self) -> dict[str, Any]:
        with self.lock:
            current_type = self.player_types[int(self.game.current_player)]
        if current_type == "alphabeta":
            return self._prepare_alpha_beta_search(current_type)

        # Only state capture and result commit need the game lock.  The costly
        # search runs outside it so a new game can replace this one immediately.
        with self.lock:
            if self.pending_action is not None:
                return self.payload()
            if self.game.ended:
                raise ValueError("对局已经结束，请开始新对局")
            current = int(self.game.current_player)
            player = self.players[current]
            if player is None:
                raise ValueError("当前轮到人工玩家，请直接点击棋盘落子")
            if player.mcts is None:
                player.init(self.game)

            generation = self.generation
            cancel_event = threading.Event()
            self._active_searches.add(cancel_event)
            search_game = self.game
            traced_mcts = player.mcts
            total_searches = int(self.config_values["n_search"])
            completed_searches = 0
            current_rollouts = 0
            self._set_search_progress(
                generation=generation,
                running=True,
                completed=0,
                total=total_searches,
                current_rollout=0,
                rollouts_per_search=int(self.config_values["n_rollout"]),
                phase="starting",
                started_at=time.monotonic(),
            )

            def update_progress(phase: str) -> None:
                nonlocal completed_searches, current_rollouts
                if phase == "rollout":
                    current_rollouts += 1
                if phase == "backup":
                    completed_searches += 1
                    current_rollouts = 0
                self._set_search_progress(
                    generation=generation,
                    completed=completed_searches,
                    current_rollout=current_rollouts,
                    phase=phase,
                )

            recorder = TraceRecorder(
                traced_mcts,
                self.game_kind,
                self.size,
                progress_callback=update_progress,
                cancel_event=cancel_event,
            )
            recorder.install()

        action: int | None = None
        search_error: str | None = None
        search_cancelled = False
        try:
            action = int(player.play(search_game))
            recorder.add_event(
                "search-end",
                "搜索完成",
                f"策略选择动作 {_action_label(self.game_kind, self.size, action)}。",
                focus=getattr(traced_mcts, "root", None),
                tree=True,
                selected_action=action,
            )
            recorder.add_event(
                "move",
                "执行落子",
                f"把 {_action_label(self.game_kind, self.size, action)} 应用于真实棋盘。",
                selected_action=action,
            )
            self._set_search_progress(
                generation=generation,
                running=False,
                completed=total_searches,
                current_rollout=0,
                phase="complete",
            )
        except SearchCancelled:
            search_cancelled = True
        except Exception:
            search_error = traceback.format_exc()
            recorder.add_event(
                "error",
                "搜索异常退出",
                "现有算法在搜索过程中抛出了异常。",
                tree=True,
                traceback=search_error,
            )
            self._set_search_progress(generation=generation, running=False, phase="error")
        finally:
            recorder.restore()
            with self.lock:
                self._active_searches.discard(cancel_event)

        with self.lock:
            if search_cancelled or generation != self.generation:
                return self.payload()
            self.pending_action = action if search_error is None else None
            self.error = search_error
            self.trace = recorder.events
            self.trace_index = 0 if self.trace else -1
            return self.payload()

    def _prepare_alpha_beta_search(self, player_type: str) -> dict[str, Any]:
        with self.lock:
            if self.pending_action is not None:
                return self.payload()
            if self.game.ended:
                raise ValueError("对局已经结束，请开始新对局")
            current = int(self.game.current_player)
            if self.player_types[current] != player_type:
                raise ValueError("当前玩家已经发生变化，请重新开始搜索")
            player = self.players[current]
            generation = self.generation
            cancel_event = threading.Event()
            self._active_searches.add(cancel_event)
            search_game = self.game
            algorithm = "Alpha-Beta"
            recorder = AlphaBetaTraceRecorder(
                self.game_kind, self.size, algorithm, cancel_event=cancel_event
            )
            player.policy.trace_callback = recorder.record
            self._set_search_progress(
                generation=generation,
                running=True,
                completed=0,
                total=1,
                current_rollout=0,
                rollouts_per_search=1,
                phase="alpha-beta",
                started_at=time.monotonic(),
            )

        action: int | None = None
        search_error: str | None = None
        search_cancelled = False
        try:
            action = int(player.play(search_game))
            if not recorder.events:
                recorder._append(
                    "search-end",
                    "读取缓存结果",
                    f"该局面已有搜索结果，直接选择 {_action_label(self.game_kind, self.size, action)}。",
                    0,
                    selected_action=action,
                )
            recorder.add_move_event(action)
            self._set_search_progress(
                generation=generation,
                running=False,
                completed=1,
                phase="complete",
            )
        except SearchCancelled:
            search_cancelled = True
        except Exception:
            search_error = traceback.format_exc()
            recorder._append(
                "error",
                "搜索异常退出",
                "现有 Alpha-Beta 算法在搜索过程中抛出了异常。",
                0,
                traceback=search_error,
            )
            self._set_search_progress(generation=generation, running=False, phase="error")
        finally:
            player.policy.trace_callback = None
            with self.lock:
                self._active_searches.discard(cancel_event)

        with self.lock:
            if search_cancelled or generation != self.generation:
                return self.payload()
            self.pending_action = action if search_error is None else None
            self.error = search_error
            self.trace = recorder.events
            self.trace_index = 0 if self.trace else -1
            return self.payload()

    def next_event(self) -> dict[str, Any]:
        with self.lock:
            if not self.trace:
                raise ValueError("当前没有可回放的搜索过程")
            if self.trace_index + 1 < len(self.trace):
                self.trace_index += 1
                event = self.trace[self.trace_index]
                if event.get("phase") == "move" and self.pending_action is not None:
                    action = self.pending_action
                    self.pending_action = None
                    try:
                        source = self.player_types[int(self.game.current_player)]
                        self._apply_action(action, source)
                    except Exception:
                        self.error = traceback.format_exc()
                        event["phase"] = "error"
                        event["title"] = "落子异常退出"
                        event["message"] = "搜索返回的动作无法应用到真实棋盘。"
                        event["traceback"] = self.error
            return self.payload()

    def skip_search(self) -> dict[str, Any]:
        """Run or finish this algorithm turn and apply its selected action."""
        with self.lock:
            if self.game.ended:
                raise ValueError("对局已经结束，请开始新对局")
            current = int(self.game.current_player)
            if self.player_types[current] == "human":
                raise ValueError("当前是人工回合")
            if self.error:
                return self.payload()
            generation = self.generation
            needs_search = self.pending_action is None

        if needs_search:
            self.prepare_search()

        with self.lock:
            if generation != self.generation:
                return self.payload()
            while self.pending_action is not None and self.trace_index + 1 < len(self.trace):
                self.next_event()
            return self.payload()

    def state(self) -> dict[str, Any]:
        current = int(self.game.current_player)
        result = self.history[-1].get("result") if self.game.ended and self.history else None
        return {
            "game": self.game_kind,
            "size": self.size,
            "board": _plain(np.asarray(self.game.observation)),
            "valid_actions": [
                int(index) for index, allowed in enumerate(np.asarray(self.game.action_mask)) if allowed
            ],
            "current_player": current,
            "current_player_type": self.player_types[current],
            "players": {"black": self.player_types[BLACK], "white": self.player_types[WHITE]},
            "config": dict(self.config_values),
            "ended": bool(self.game.ended),
            "result": result,
            "history": list(self.history),
            "pending": self.pending_action is not None,
            "error": self.error,
        }

    def payload(self) -> dict[str, Any]:
        event = None
        if 0 <= self.trace_index < len(self.trace):
            event = self.trace[self.trace_index]
        return {
            "state": self.state(),
            "event": event,
            "trace_index": self.trace_index,
            "trace_count": len(self.trace),
            "has_next": self.trace_index + 1 < len(self.trace),
        }


class ExperimentManager:
    """Run repeatable head-to-head experiments without blocking the web UI."""

    PLAYER_NAMES = {
        "random": "Random", "mcts": "MCTS",
        "heuristic_mcts": "启发式 MCTS", "alphabeta": "Alpha-Beta",
    }

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.data = self._empty()

    def _empty(self) -> dict[str, Any]:
        return {
            "running": False, "cancelled": False, "completed": 0, "total": 0,
            "percent": 0.0, "elapsed": 0.0, "phase": "idle", "error": None,
            "rows": [], "last_game": None, "config": None,
        }

    @staticmethod
    def _number(spec: dict[str, Any], key: str, default: Any, cast: Any) -> Any:
        try:
            return cast(spec.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 参数无效") from exc

    def _validate_player(self, raw: dict[str, Any], size: int) -> dict[str, Any]:
        algorithm = str(raw.get("algorithm", "random"))
        if algorithm not in self.PLAYER_NAMES:
            raise ValueError("未知的实验棋手类型")
        spec = {
            "algorithm": algorithm,
            "C": self._number(raw, "C", 1.0, float),
            "n_search": self._number(raw, "n_search", 32, int),
            "n_rollout": self._number(raw, "n_rollout", 3, int),
            "heuristic_weight": self._number(raw, "heuristic_weight", 1.0, float),
            "heuristic_scale": self._number(raw, "heuristic_scale", 1.0, float),
            "ab_depth": self._number(raw, "ab_depth", 2, int),
        }
        if spec["n_search"] < 1 or spec["n_search"] > 2000 or spec["n_rollout"] < 1 or spec["n_rollout"] > 100:
            raise ValueError("搜索次数必须为 1–2000，模拟次数必须为 1–100")
        if not 0 <= spec["heuristic_weight"] <= 1 or spec["heuristic_scale"] <= 0:
            raise ValueError("启发式权重必须为 0–1，缩放必须大于 0")
        if not 1 <= spec["ab_depth"] <= 3:
            raise ValueError("Alpha-Beta 深度必须为 1–3")
        if algorithm == "alphabeta" and size != 5:
            raise ValueError("Alpha-Beta 仅允许在 5×5 围棋棋盘上使用")
        return spec

    def _make_player(self, spec: dict[str, Any]) -> Any:
        if spec["algorithm"] == "random":
            return RandomPlayer()
        if spec["algorithm"] == "alphabeta":
            return AlphaBetaPlayer(depth=spec["ab_depth"])
        if spec["algorithm"] == "heuristic_mcts":
            config = HeuristicUCTMCTSConfig(
                C=spec["C"], n_search=spec["n_search"], n_rollout=spec["n_rollout"],
                heuristic_weight=spec["heuristic_weight"], heuristic_scale=spec["heuristic_scale"],
            )
            return HeuristicUCTPlayer(config, deterministic=True)
        config = UCTMCTSConfig(C=spec["C"], n_search=spec["n_search"], n_rollout=spec["n_rollout"])
        return UCTPlayer(config, deterministic=True)

    def start(self, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.data["running"]:
                raise ValueError("已有批量实验正在运行")
        size = int(options.get("size", 5))
        if size not in {5, 7, 9}:
            raise ValueError("围棋棋盘大小必须是 5、7 或 9")
        games = int(options.get("games", 5))
        if not 1 <= games <= 50:
            raise ValueError("每种执棋顺序的对局数必须为 1–50")
        swap = bool(options.get("swap", True))
        seed = int(options.get("seed", 0))
        player_a = self._validate_player(dict(options.get("player_a", {})), size)
        player_b = self._validate_player(dict(options.get("player_b", {})), size)
        config = {"size": size, "games": games, "swap": swap, "seed": seed, "player_a": player_a, "player_b": player_b}
        total = games * (2 if swap else 1)
        self.cancel_event = threading.Event()
        with self.lock:
            if self.data["running"]:
                raise ValueError("已有批量实验正在运行")
            self.data = self._empty()
            self.data.update({"running": True, "total": total, "phase": "starting", "config": config})
        self.worker = threading.Thread(target=self._run, args=(config,), daemon=True)
        self.worker.start()
        return self.status()

    def cancel(self) -> dict[str, Any]:
        self.cancel_event.set()
        with self.lock:
            if self.data["running"]:
                self.data["phase"] = "cancelling"
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            return _plain(dict(self.data))

    def _play_one(self, size: int, black_spec: dict[str, Any], white_spec: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        game = GoGame(size); game.reset()
        black, white = self._make_player(black_spec), self._make_player(white_spec)
        history: list[dict[str, Any]] = []
        while not game.ended:
            if self.cancel_event.is_set():
                raise SearchCancelled
            mover = int(game.current_player)
            player = black if mover == BLACK else white
            action = int(player.play(game))
            if action < 0 or action >= game.action_space_size or not game.action_mask[action]:
                raise ValueError(f"{self.PLAYER_NAMES[(black_spec if mover == BLACK else white_spec)['algorithm']]} 返回了非法动作 {action}")
            _, reward, done = game.step(action, return_obs=False)
            history.append({"player": mover, "action": action, "label": _action_label("go", size, action)})
            opponent = white if mover == BLACK else black
            if hasattr(opponent, "opp_play"):
                opponent.opp_play(action)
            if done:
                black_value = float(reward) if mover == BLACK else -float(reward)
                result = "和棋" if abs(black_value) < EPS * 10 else ("黑棋获胜" if black_value > 0 else "白棋获胜")
                return black_value, {"size": size, "board": _plain(game.observation), "history": history, "result": result}
        raise RuntimeError("对局未正常结束")

    def _run(self, config: dict[str, Any]) -> None:
        started = time.perf_counter()
        np.random.seed(config["seed"])
        orientations = [("A 执黑 / B 执白", True)]
        if config["swap"]:
            orientations.append(("B 执黑 / A 执白", False))
        rows = []
        completed = 0
        try:
            for label, a_is_black in orientations:
                row = {"label": label, "games": 0, "a_wins": 0, "b_wins": 0, "draws": 0, "elapsed": 0.0}
                rows.append(row)
                orientation_started = time.perf_counter()
                for _ in range(config["games"]):
                    if self.cancel_event.is_set():
                        raise SearchCancelled
                    black_spec = config["player_a"] if a_is_black else config["player_b"]
                    white_spec = config["player_b"] if a_is_black else config["player_a"]
                    black_value, last_game = self._play_one(config["size"], black_spec, white_spec)
                    a_value = black_value if a_is_black else -black_value
                    row["games"] += 1
                    if abs(a_value) < EPS * 10:
                        row["draws"] += 1
                    elif a_value > 0:
                        row["a_wins"] += 1
                    else:
                        row["b_wins"] += 1
                    completed += 1
                    row["elapsed"] = time.perf_counter() - orientation_started
                    with self.lock:
                        self.data.update({
                            "completed": completed, "percent": 100.0 * completed / self.data["total"],
                            "elapsed": time.perf_counter() - started, "phase": "playing",
                            "rows": [dict(item) for item in rows], "last_game": last_game,
                        })
            with self.lock:
                self.data.update({"running": False, "percent": 100.0, "phase": "complete", "elapsed": time.perf_counter() - started})
        except SearchCancelled:
            with self.lock:
                self.data.update({"running": False, "cancelled": True, "phase": "cancelled", "elapsed": time.perf_counter() - started})
        except Exception:
            with self.lock:
                self.data.update({"running": False, "phase": "error", "error": traceback.format_exc(), "elapsed": time.perf_counter() - started})


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>棋类树搜索可视化调试器</title>
<style>
  :root { color-scheme: light; --ink:#152238; --muted:#637083; --line:#d7deea; --panel:#fff; --bg:#f4f7fb; --blue:#2563eb; --blue-soft:#dbeafe; --amber:#b45309; --amber-soft:#fef3c7; --red:#b91c1c; --green:#047857; }
  * { box-sizing: border-box; }
  body { margin:0; color:var(--ink); background:var(--bg); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
  button, select, input { font:inherit; }
  button { cursor:pointer; }
  button:disabled { cursor:not-allowed; opacity:.48; }
  .app { max-width:1440px; margin:0 auto; padding:20px; }
  .topbar { display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap; padding:16px; background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 6px 24px rgba(24,42,72,.06); }
  .guide { margin-top:12px; overflow:hidden; background:var(--panel); border:1px solid var(--line); border-radius:12px; box-shadow:0 4px 18px rgba(24,42,72,.05); }
  .guide summary { padding:13px 16px; cursor:pointer; color:var(--ink); font-weight:650; }
  .guide-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; padding:0 18px 15px; color:var(--muted); }
  .guide-grid h2 { margin:0 0 5px; color:var(--ink); font-size:14px; }
  .guide-grid ol { margin:0; padding-left:20px; }
  .guide-grid li + li { margin-top:4px; }
  .calculation-progress { margin-top:12px; padding:12px 16px; background:var(--panel); border:1px solid var(--line); border-radius:12px; box-shadow:0 4px 18px rgba(24,42,72,.05); }
  .calculation-progress[hidden] { display:none; }
  .calculation-meta { display:flex; justify-content:space-between; gap:12px; margin-bottom:7px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .calculation-meta strong { color:var(--ink); font-weight:650; }
  .progress-track { height:12px; overflow:hidden; border-radius:999px; background:#e5eaf2; }
  .progress-fill { position:relative; width:0; height:100%; border-radius:inherit; background:var(--blue); transition:width .18s ease; }
  .progress-fill.running::after { content:""; position:absolute; inset:0; background:linear-gradient(100deg,transparent 20%,rgba(255,255,255,.42) 48%,transparent 76%); animation:progress-shimmer 1.15s linear infinite; }
  @keyframes progress-shimmer { from { transform:translateX(-100%); } to { transform:translateX(100%); } }
  @media (prefers-reduced-motion:reduce) { .progress-fill { transition:none; } .progress-fill.running::after { animation:none; } }
  .brand { margin-right:auto; min-width:210px; }
  h1 { margin:0; font-size:20px; }
  .subtitle { color:var(--muted); margin-top:3px; }
  .field { display:grid; gap:5px; color:var(--muted); font-size:12px; }
  select, input { min-height:36px; border:1px solid #bac5d6; border-radius:8px; background:#fff; color:var(--ink); padding:6px 9px; }
  input[type=number] { width:76px; }
  .primary, .secondary, .danger { min-height:38px; border-radius:9px; padding:7px 14px; border:1px solid transparent; }
  .primary { background:var(--blue); color:white; }
  .secondary { background:#fff; border-color:#b8c4d6; color:var(--ink); }
  .danger { background:#fff; border-color:#fecaca; color:var(--red); }
  .layout { display:grid; grid-template-columns:minmax(350px, .78fr) minmax(520px, 1.35fr); gap:16px; margin-top:16px; align-items:start; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 6px 24px rgba(24,42,72,.05); overflow:hidden; }
  .panel-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line); }
  .panel-title { font-size:16px; font-weight:650; }
  .status { color:var(--muted); }
  .status strong { color:var(--ink); }
  .board-wrap { padding:20px; display:grid; place-items:center; min-height:430px; }
  .go-board { width:min(100%,520px); aspect-ratio:1; display:block; border-radius:8px; box-shadow:0 2px 10px rgba(67,45,16,.14); }
  .board-actions { padding:0 16px 16px; display:flex; flex-wrap:wrap; gap:9px; justify-content:center; }
  .history { padding:12px 16px 16px; border-top:1px solid var(--line); color:var(--muted); max-height:118px; overflow:auto; }
  .history span { display:inline-block; margin:3px 8px 3px 0; color:var(--ink); }
  .trace-controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); }
  .trace-progress { margin-left:auto; color:var(--muted); }
  .event { margin:14px 16px 0; padding:12px 14px; border-radius:10px; background:#f7f9fc; border:1px solid var(--line); }
  .event-title { font-weight:650; }
  .event-message { color:var(--muted); margin-top:3px; }
  .phase { display:inline-block; margin-right:8px; padding:2px 7px; border-radius:999px; background:var(--blue-soft); color:#1d4ed8; font-size:12px; }
  .search-grid { display:grid; grid-template-columns:minmax(0,1fr) 250px; min-height:520px; }
  .tree-wrap { padding:12px; overflow:auto; border-right:1px solid var(--line); }
  #tree { display:block; width:100%; min-width:520px; height:auto; min-height:450px; }
  .side-data { padding:12px; overflow:auto; }
  .side-data h3 { margin:2px 0 9px; font-size:14px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th, td { padding:7px 5px; text-align:right; border-bottom:1px solid #e7ebf1; }
  th:first-child, td:first-child { text-align:left; }
  tr.selected { background:var(--amber-soft); }
  .empty { color:var(--muted); padding:14px 4px; }
  pre { max-height:180px; overflow:auto; white-space:pre-wrap; color:var(--red); background:#fff1f2; border:1px solid #fecdd3; border-radius:8px; padding:10px; font-size:12px; }
  .legend { display:flex; gap:14px; color:var(--muted); font-size:12px; padding:0 16px 14px; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:5px; }
  .experiment { margin-top:16px; }
  .experiment-intro { padding:12px 16px; color:var(--muted); border-bottom:1px solid var(--line); }
  .experiment-controls { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; padding:16px; }
  .experiment-common { grid-column:1/-1; display:flex; flex-wrap:wrap; gap:12px; align-items:flex-end; padding-bottom:14px; border-bottom:1px solid var(--line); }
  .player-card { padding:14px; border:1px solid var(--line); border-radius:12px; background:#fafcff; }
  .player-card h3 { margin:0 0 10px; font-size:15px; }
  .parameter-grid { display:grid; grid-template-columns:repeat(3,minmax(90px,1fr)); gap:9px; }
  .parameter-grid input { width:100%; }
  .parameter-grid .param-disabled { opacity:.42; filter:grayscale(.8); }
  .parameter-grid .param-disabled input { background:#e9edf3; color:#7a8494; cursor:not-allowed; }
  .check-field { display:flex; align-items:center; gap:7px; min-height:36px; color:var(--ink); font-size:13px; }
  .check-field input { min-height:0; width:auto; }
  .experiment-actions { grid-column:1/-1; display:flex; gap:9px; align-items:center; flex-wrap:wrap; }
  .experiment-progress { flex:1; min-width:260px; }
  .experiment-results { padding:0 16px 18px; }
  .experiment-results table th,.experiment-results table td { text-align:center; }
  .experiment-summary { margin:0 0 10px; color:var(--muted); }
  .experiment-error { color:var(--red); white-space:pre-wrap; max-height:180px; overflow:auto; }
  .last-game { display:grid; grid-template-columns:minmax(260px,420px) 1fr; gap:16px; margin-top:16px; align-items:start; }
  .last-board { width:100%; aspect-ratio:1; }
  .last-history { max-height:310px; overflow:auto; padding:10px; border:1px solid var(--line); border-radius:10px; }
  .last-history span { display:inline-block; margin:3px 8px 3px 0; }
  @media (max-width:980px) { .layout { grid-template-columns:1fr; } .search-grid { grid-template-columns:1fr; } .tree-wrap { border-right:0; border-bottom:1px solid var(--line); } }
  @media (max-width:760px) { .guide-grid,.experiment-controls,.last-game { grid-template-columns:1fr; } .parameter-grid { grid-template-columns:repeat(2,minmax(90px,1fr)); } }
  @media (max-width:600px) { .app { padding:10px; } .board-wrap { min-height:330px; padding:12px; } .topbar { align-items:stretch; } .brand { width:100%; } .search-grid { display:block; } .side-data { max-height:300px; } }
</style>
</head>
<body>
<main class="app">
  <section class="topbar">
    <div class="brand"><h1>围棋算法实验与可视化调试</h1><div class="subtitle">交互下棋、逐事件调试，并直接运行批量对比实验</div></div>
    <input id="game" type="hidden" value="go">
    <label class="field" id="sizeField">围棋大小<select id="size"><option selected>5</option><option>7</option><option>9</option></select></label>
    <label class="field">黑方<select id="black"><option value="human">人工</option><option value="mcts">MCTS</option><option value="heuristic_mcts">启发式 MCTS</option><option value="alphabeta">Alpha-Beta（仅 5×5）</option></select></label>
    <label class="field">白方<select id="white"><option value="mcts">MCTS</option><option value="heuristic_mcts">启发式 MCTS</option><option value="human">人工</option><option value="alphabeta">Alpha-Beta（仅 5×5）</option></select></label>
    <label class="field">C<input id="cValue" type="number" min="0" step="0.1" value="1"></label>
    <label class="field">搜索次数<input id="searches" type="number" min="1" max="2000" value="32"></label>
    <label class="field">模拟次数<input id="rollouts" type="number" min="1" max="100" value="3"></label>
    <label class="field">Alpha-Beta 深度<input id="abDepth" type="number" min="1" max="3" value="2"></label>
    <label class="field">启发式权重<input id="heuristicWeight" type="number" min="0" max="1" step="0.1" value="1"></label>
    <label class="field">启发式缩放<input id="heuristicScale" type="number" min="0.01" step="0.1" value="1"></label>
    <button class="primary" id="newGame">新对局</button>
  </section>

  <details class="guide" open>
    <summary>操作指引（参数设置、单局调试与批量实验）</summary>
    <div class="guide-grid">
      <div>
        <h2>单局交互与搜索回放</h2>
        <ol>
          <li>先选棋盘、黑白方和搜索参数，再点击“新对局”；修改顶部参数后，只有新对局才会应用它们。</li>
          <li>人工回合直接点击棋盘交叉点或“停一手”；算法回合点击“开始本回合搜索”。</li>
          <li>搜索计算完成后，用“下一搜索事件”或“自动回放”查看选择、扩展、模拟和回传；回放到“执行落子”时才会切换玩家。</li>
          <li>“跳过本轮搜索细节”会在必要时完成搜索并直接落子。任意时刻都可点击“新对局”，旧搜索会被取消。</li>
        </ol>
      </div>
      <div>
        <h2>批量对局实验</h2>
        <ol>
          <li>在页面下方分别选择算法 A/B 及各自参数；Alpha-Beta 只能用于 5×5 棋盘。</li>
          <li>“每种执棋顺序对局数”是单个黑白顺序的局数；勾选“交换执棋”后，实际总局数为该数值的两倍。</li>
          <li>点击“开始批量实验”后查看进度和分执棋结果；“取消实验”会等待当前一次搜索结束后停止。</li>
          <li>实验后可检查最近一局终局与落子记录，并用“复制结果摘要”整理报告。算法抛出的异常会显示在结果区。</li>
        </ol>
      </div>
    </div>
  </details>

  <section class="calculation-progress" id="calculationProgress" aria-live="polite" hidden>
    <div class="calculation-meta"><strong id="calculationLabel">正在准备树搜索…</strong><span id="calculationNumbers">0 / 0 · 0%</span></div>
    <div class="progress-track" role="progressbar" id="progressTrack" aria-label="树搜索进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="progress-fill running" id="progressFill"></div></div>
  </section>

  <div class="layout">
    <section class="panel">
      <div class="panel-head"><div class="panel-title" id="boardTitle">围棋 5×5</div><div class="status" id="status"></div></div>
      <div class="board-wrap" id="board"></div>
      <div class="board-actions"><button class="secondary" id="pass">停一手</button><button class="primary" id="startSearch">开始本回合搜索</button><button class="secondary" id="skipSearch">跳过本轮搜索细节</button></div>
      <div class="history" id="history"></div>
    </section>

    <section class="panel">
      <div class="panel-head"><div class="panel-title">搜索树</div><div class="status" id="treeStats">尚未搜索</div></div>
      <div class="trace-controls">
        <button class="primary" id="nextEvent">下一搜索事件</button>
        <button class="secondary" id="auto">自动回放（1 秒）</button>
        <button class="secondary" id="stopAuto" disabled>停止</button>
        <span class="trace-progress" id="progress">0 / 0</span>
      </div>
      <div class="event" id="event"><div class="event-title">等待搜索</div><div class="event-message">轮到算法玩家时，点击“开始本回合搜索”。</div></div>
      <div class="search-grid">
        <div class="tree-wrap"><svg id="tree" viewBox="0 0 900 460" role="img" aria-label="树搜索过程"></svg></div>
        <aside class="side-data"><h3>当前节点的动作</h3><div id="candidates" class="empty">搜索过程中会显示动作统计或评估值。</div><div id="error"></div></aside>
      </div>
      <div class="legend"><span><i class="dot" style="background:#2563eb"></i>本次路径</span><span><i class="dot" style="background:#f59e0b"></i>当前节点</span><span id="treeLegend">MCTS：N 访问次数 · W 累计价值 · Q 平均价值</span></div>
    </section>
  </div>

  <section class="panel experiment">
    <div class="panel-head"><div class="panel-title">批量对局实验</div><div class="status" id="experimentStatus">尚未运行</div></div>
    <div class="experiment-intro">用于完成 Random / MCTS / 启发式 MCTS / Alpha-Beta 的速度与棋力对比。实验在后台运行，页面会持续显示进度；选择“交换执棋”可自动各运行一组 A 执黑和 B 执黑的对局。
    </div>
    <div class="experiment-controls">
      <div class="experiment-common">
        <label class="field">棋盘<select id="expSize"><option selected>5</option><option>7</option><option>9</option></select></label>
        <label class="field">每种执棋顺序对局数<input id="expGames" type="number" min="1" max="50" value="10"></label>
        <label class="field">随机种子<input id="expSeed" type="number" value="0"></label>
        <label class="check-field"><input id="expSwap" type="checkbox" checked>交换执棋（推荐）</label>
      </div>
      <div class="player-card" data-exp-player="a"><h3>算法 A</h3><div class="parameter-grid"></div></div>
      <div class="player-card" data-exp-player="b"><h3>算法 B</h3><div class="parameter-grid"></div></div>
      <div class="experiment-actions">
        <button class="primary" id="startExperiment">开始批量实验</button><button class="danger" id="cancelExperiment" disabled>取消实验</button><button class="secondary" id="copyExperiment" disabled>复制结果摘要</button>
        <div class="experiment-progress"><div class="calculation-meta"><strong id="experimentPhase">等待开始</strong><span id="experimentNumbers">0 / 0 · 0%</span></div><div class="progress-track"><div class="progress-fill" id="experimentFill"></div></div></div>
      </div>
    </div>
    <div class="experiment-results">
      <p class="experiment-summary" id="experimentSummary">结果将在每局结束后更新。</p>
      <div class="experiment-error" id="experimentError"></div>
      <table><thead><tr><th>执棋顺序</th><th>已完成</th><th>A 获胜</th><th>B 获胜</th><th>和棋</th><th>A 胜率</th><th>B 不败率</th><th>用时</th></tr></thead><tbody id="experimentRows"><tr><td colspan="8" class="empty">尚无数据</td></tr></tbody></table>
      <div class="last-game" id="lastGame" hidden><div><h3>最近一局终局</h3><div id="lastBoard" class="last-board"></div></div><div><h3 id="lastGameResult">—</h3><div id="lastHistory" class="last-history"></div></div></div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let payload = null;
let latestTree = {nodes:[],edges:[],path:[]};
let latestCandidates = [];
let autoTimer = null;
let searchProgressTimer = null;
let busy = false;
let operationToken = 0;

async function api(path, body={}) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

const phaseLabels = {
  idle:'等待搜索', starting:'正在准备搜索树…', selection:'正在执行 UCT 选择…',
  expansion:'正在选择或扩展叶节点…', rollout:'正在进行随机模拟…',
  backup:'正在回传更新…', 'alpha-beta':'正在执行 Alpha-Beta 搜索…',
  complete:'搜索完成', error:'搜索异常结束'
};

function renderCalculationProgress(progress) {
  const total=Number(progress.total||Number($('searches').value)||0);
  const completed=Math.min(total,Number(progress.completed||0));
  const percent=Math.min(100,Number(progress.percent??(total?100*completed/total:0)));
  const rollout=Number(progress.current_rollout||0), rolloutTotal=Number(progress.rollouts_per_search||0);
  $('calculationProgress').hidden=false;
  const rolloutDetail=progress.phase==='rollout'&&rolloutTotal?`（本轮模拟 ${rollout} / ${rolloutTotal}）`:'';
  $('calculationLabel').textContent=(phaseLabels[progress.phase]||'MCTS 正在计算…')+rolloutDetail;
  $('calculationNumbers').textContent=`${completed} / ${total} · ${percent.toFixed(0)}% · ${Number(progress.elapsed||0).toFixed(1)} 秒`;
  $('progressFill').style.width=`${percent}%`;
  $('progressFill').classList.toggle('running',!!progress.running);
  $('progressTrack').setAttribute('aria-valuenow',String(Math.round(percent)));
}

async function pollSearchProgress() {
  try {
    const response=await fetch('/api/progress',{cache:'no-store'});
    if(response.ok) renderCalculationProgress(await response.json());
  } catch(_error) {
    // The search request still reports the actual error when it finishes.
  }
}

async function runMCTS(path) {
  if(busy)return;
  const token=++operationToken;
  busy=true;
  stopAuto();
  latestTree={nodes:[],edges:[],path:[]}; latestCandidates=[];
  renderTree(latestTree,null); renderCandidates([],null);
  renderCalculationProgress({running:true,completed:0,total:Number($('searches').value),phase:'starting',elapsed:0});
  for(const id of ['startSearch','skipSearch','pass','nextEvent','auto']) $(id).disabled=true;
  const progressTimer=setInterval(()=>{ if(token===operationToken)pollSearchProgress(); },200);
  searchProgressTimer=progressTimer;
  await new Promise(resolve=>requestAnimationFrame(()=>resolve()));
  try {
    const result=await api(path);
    if(token===operationToken){ payload=result; render(payload); }
  } catch(e) {
    if(token===operationToken)alert(e.message);
  } finally {
    clearInterval(progressTimer);
    if(searchProgressTimer===progressTimer)searchProgressTimer=null;
    if(token===operationToken){
      await pollSearchProgress();
      busy=false;
      if(payload)render(payload);
      setTimeout(()=>{ if(!busy&&token===operationToken)$('calculationProgress').hidden=true; },900);
    }
  }
}

async function startNewGame() {
  const token=++operationToken;
  busy=true;
  stopAuto();
  if(searchProgressTimer){ clearInterval(searchProgressTimer); searchProgressTimer=null; }
  $('calculationProgress').hidden=true;
  latestTree={nodes:[],edges:[],path:[]}; latestCandidates=[];
  try {
    const result=await api('/api/new',{game:$('game').value,size:Number($('size').value),black:$('black').value,white:$('white').value,C:Number($('cValue').value),n_search:Number($('searches').value),n_rollout:Number($('rollouts').value),ab_depth:Number($('abDepth').value),heuristic_weight:Number($('heuristicWeight').value),heuristic_scale:Number($('heuristicScale').value)});
    if(token===operationToken){ payload=result; render(result); }
  } catch(e) {
    if(token===operationToken)alert(e.message);
  } finally {
    if(token===operationToken){ busy=false; $('newGame').disabled=false; if(payload)render(payload); }
  }
}

function playerName(value, game) {
  return value === 1 ? '黑棋' : '白棋';
}

function algorithmName(type) {
  return {human:'人工',random:'Random',mcts:'MCTS',heuristic_mcts:'启发式 MCTS',alphabeta:'Alpha-Beta'}[type] || type;
}

function renderBoard(state) {
  const host = $('board'); host.innerHTML = '';
  $('boardTitle').textContent = `围棋 ${state.size}×${state.size}`;
  $('pass').style.display = '';
  const canHuman = !state.ended && !state.pending && state.current_player_type === 'human';
  {
    const n=state.size, pad=34, span=532, step=(span-2*pad)/(n-1);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.classList.add('go-board'); svg.setAttribute('viewBox','0 0 532 532'); svg.style.background='#d6a85f';
    for(let i=0;i<n;i++){
      const p=pad+i*step;
      for(const [x1,y1,x2,y2] of [[pad,p,span-pad,p],[p,pad,p,span-pad]]){
        const line=document.createElementNS(svg.namespaceURI,'line');
        for(const [k,v] of Object.entries({x1,y1,x2,y2,stroke:'#3b2b18','stroke-width':2})) line.setAttribute(k,v); svg.appendChild(line);
      }
    }
    state.board.flat().forEach((stone, action)=>{
      const row=Math.floor(action/n), col=action%n, cx=pad+col*step, cy=pad+row*step;
      const hit=document.createElementNS(svg.namespaceURI,'circle');
      hit.setAttribute('cx',cx); hit.setAttribute('cy',cy); hit.setAttribute('r',Math.min(18,step*.38));
      hit.setAttribute('fill',stone===1?'#151515':stone===-1?'#fafafa':'transparent');
      if(stone!==0){ hit.setAttribute('stroke',stone===1?'#000':'#777'); hit.setAttribute('stroke-width','1.5'); }
      const legal=canHuman && state.valid_actions.includes(action);
      if(legal){ hit.style.cursor='pointer'; hit.onmouseenter=()=>hit.setAttribute('fill','rgba(37,99,235,.28)'); hit.onmouseleave=()=>hit.setAttribute('fill','transparent'); hit.onclick=()=>humanMove(action); }
      svg.appendChild(hit);
    }); host.appendChild(svg);
  }
  const turn=playerName(state.current_player,state.game);
  const algorithm=algorithmName(state.current_player_type);
  $('status').innerHTML=state.ended?`<strong>${state.result||'对局结束'}</strong>`:`轮到 <strong>${turn}</strong> · ${algorithm}`;
  $('pass').disabled=!(canHuman && state.valid_actions.includes(state.size*state.size));
  $('startSearch').textContent=`开始本回合${state.current_player_type==='human'?'搜索':algorithm}`;
  $('skipSearch').textContent=`跳过本轮${state.current_player_type.includes('mcts')?' MCTS':''}细节`;
  $('startSearch').disabled=state.ended || state.pending || state.current_player_type==='human';
  $('skipSearch').disabled=state.ended || state.current_player_type==='human' || !!state.error;
  $('history').innerHTML=state.history.length?state.history.map((m,i)=>`<span>${i+1}. ${playerName(m.player,state.game)} ${m.label}</span>`).join(''):'尚未落子';
}

function renderTree(tree, event) {
  if (tree) latestTree=tree;
  const data=latestTree, svg=$('tree'); svg.innerHTML='';
  if(!data.nodes.length){ svg.setAttribute('viewBox','0 0 900 460'); return; }
  const groups={}; data.nodes.forEach(n=>(groups[n.depth]??=[]).push(n));
  const depths=Object.keys(groups).map(Number), maxDepth=Math.max(...depths), widestRow=Math.max(...Object.values(groups).map(row=>row.length)), width=Math.max(900,widestRow*78), rowH=105, height=Math.max(460,95+maxDepth*rowH);
  svg.style.width=`${width}px`; svg.style.minWidth=`${width}px`;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const positions={};
  depths.forEach(depth=>{ const row=groups[depth]; row.forEach((node,i)=>{positions[node.id]={x:(i+1)*width/(row.length+1),y:52+depth*rowH};}); });
  const path=new Set((event&&event.path)||data.path||[]), focus=event&&event.focus;
  const backupChanges=new Map(((event&&event.backup_changes)||[]).map(change=>[change.id,change]));
  data.edges.forEach(edge=>{ const a=positions[edge.source],b=positions[edge.target]; if(!a||!b)return; const line=document.createElementNS(svg.namespaceURI,'line'); line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);line.setAttribute('stroke',path.has(edge.source)&&path.has(edge.target)?'#2563eb':'#cbd5e1');line.setAttribute('stroke-width',path.has(edge.source)&&path.has(edge.target)?'3':'1.5');svg.appendChild(line); });
  data.nodes.forEach(node=>{ const p=positions[node.id],g=document.createElementNS(svg.namespaceURI,'g');
    const circle=document.createElementNS(svg.namespaceURI,'circle'); circle.setAttribute('cx',p.x);circle.setAttribute('cy',p.y);circle.setAttribute('r','28');circle.setAttribute('fill',node.id===focus?'#fef3c7':path.has(node.id)?'#dbeafe':node.pruned?'#fee2e2':'#fff');circle.setAttribute('stroke',node.id===focus?'#f59e0b':path.has(node.id)?'#2563eb':node.pruned?'#b91c1c':'#94a3b8');circle.setAttribute('stroke-width',node.id===focus?'4':'2');g.appendChild(circle);
    const label=document.createElementNS(svg.namespaceURI,'text');label.setAttribute('x',p.x);label.setAttribute('y',p.y-4);label.setAttribute('text-anchor','middle');label.setAttribute('font-size','12');label.setAttribute('font-weight','600');label.textContent=node.label;g.appendChild(label);
    const stats=document.createElementNS(svg.namespaceURI,'text');stats.setAttribute('x',p.x);stats.setAttribute('y',p.y+12);stats.setAttribute('text-anchor','middle');stats.setAttribute('font-size','10');stats.setAttribute('fill','#475569');stats.textContent=node.algorithm==='alpha-beta'?`V=${node.value==null?'—':Number(node.value).toFixed(2)} α=${node.alpha}`:`N=${node.n} Q=${node.q==null?'—':Number(node.q).toFixed(2)}`;g.appendChild(stats);svg.appendChild(g);
    const change=backupChanges.get(node.id);
    if(change){ const delta=document.createElementNS(svg.namespaceURI,'text');delta.setAttribute('x',p.x);delta.setAttribute('y',p.y+43);delta.setAttribute('text-anchor','middle');delta.setAttribute('font-size','11');delta.setAttribute('font-weight','700');delta.setAttribute('fill',change.delta_w<0?'#b91c1c':'#047857');delta.textContent=`ΔN=${change.delta_n>=0?'+':''}${change.delta_n}  ΔW=${change.delta_w>=0?'+':''}${Number(change.delta_w).toFixed(2)}`;g.appendChild(delta); }
  });
  $('treeStats').textContent=`${data.nodes.length} 个可见节点${data.truncated?'（已截断）':''}`;
  $('treeLegend').textContent=data.algorithm==='alpha-beta'?'Alpha-Beta：V 当前价值 · α/β 搜索边界 · 红色表示发生剪枝':'MCTS：N 访问次数 · W 累计价值 · Q 平均价值';
}

function renderCandidates(candidates, selected) {
  if(candidates) latestCandidates=candidates;
  if(!latestCandidates.length){ $('candidates').innerHTML='<div class="empty">当前事件没有动作统计。</div>'; return; }
  if(latestCandidates[0].algorithm==='alpha-beta'){
    $('candidates').innerHTML=`<table><thead><tr><th>动作</th><th>价值</th><th>状态</th></tr></thead><tbody>${latestCandidates.map(c=>`<tr class="${c.action===selected?'selected':''}"><td>${c.label}</td><td>${c.value==null?'—':Number(c.value).toFixed(3)}</td><td>${c.status}</td></tr>`).join('')}</tbody></table>`;
  }else{
    $('candidates').innerHTML=`<table><thead><tr><th>动作</th><th>N</th><th>W</th><th>Q</th></tr></thead><tbody>${latestCandidates.map(c=>`<tr class="${c.action===selected?'selected':''}"><td>${c.label}</td><td>${c.n}</td><td>${Number(c.w).toFixed(2)}</td><td>${c.q==null?'—':Number(c.q).toFixed(3)}</td></tr>`).join('')}</tbody></table>`;
  }
}

function render(data) {
  payload=data; const state=data.state, event=data.event;
  renderBoard(state);
  if(!event&&!latestTree.nodes.length)$('treeLegend').textContent=state.current_player_type.startsWith('alphabeta')?'Alpha-Beta：V 当前价值 · α/β 搜索边界 · 红色表示发生剪枝':'MCTS：N 访问次数 · W 累计价值 · Q 平均价值';
  $('progress').textContent=`${data.trace_index<0?0:data.trace_index+1} / ${data.trace_count}`;
  $('nextEvent').disabled=!data.has_next || busy;
  $('auto').disabled=!data.has_next || !!autoTimer;
  $('stopAuto').disabled=!autoTimer;
  if(event){
    $('event').innerHTML=`<div class="event-title"><span class="phase">${event.phase}</span>${event.title}</div><div class="event-message">${event.message}</div>`;
    renderTree(event.tree,event); renderCandidates(event.candidates,event.selected_action);
    $('error').innerHTML=event.traceback?`<pre>${escapeHtml(event.traceback)}</pre>`:'';
  } else {
    // 新对局或人工落子后不应继续显示上一轮的搜索树和事件。
    renderTree({nodes:[],edges:[],path:[]},null);
    renderCandidates([],null);
    $('treeStats').textContent='尚未搜索';
    $('event').innerHTML='<div class="event-title"><span class="phase">等待搜索</span></div><div class="event-message">轮到算法玩家时，点击“开始本回合搜索”。</div>';
    $('error').innerHTML=state.error?`<pre>${escapeHtml(state.error)}</pre>`:'';
  }
  if(!data.has_next) stopAuto();
}

function escapeHtml(s){ const d=document.createElement('div');d.textContent=s;return d.innerHTML; }
async function guarded(fn){ if(busy)return;busy=true;try{await fn();}catch(e){alert(e.message);}finally{busy=false;if(payload)render(payload);} }
async function humanMove(action){ await guarded(async()=>render(await api('/api/move',{action}))); }

function setupExperimentCard(side, defaultAlgorithm){
  const card=document.querySelector(`[data-exp-player="${side}"]`), prefix=`exp${side.toUpperCase()}`;
  card.querySelector('.parameter-grid').innerHTML=`
    <label class="field">算法<select id="${prefix}Algorithm"><option value="random">Random</option><option value="mcts">MCTS</option><option value="heuristic_mcts">启发式 MCTS</option><option value="alphabeta">Alpha-Beta（仅 5×5）</option></select></label>
    <label class="field" data-for="mcts heuristic_mcts">C<input id="${prefix}C" type="number" min="0" step="0.1" value="1"></label>
    <label class="field" data-for="mcts heuristic_mcts">搜索次数<input id="${prefix}Searches" type="number" min="1" max="2000" value="64"></label>
    <label class="field" data-for="mcts heuristic_mcts">模拟次数<input id="${prefix}Rollouts" type="number" min="1" max="100" value="7"></label>
    <label class="field" data-for="heuristic_mcts">启发式权重<input id="${prefix}Weight" type="number" min="0" max="1" step="0.1" value="1"></label>
    <label class="field" data-for="heuristic_mcts">启发式缩放<input id="${prefix}Scale" type="number" min="0.01" step="0.1" value="1"></label>
    <label class="field" data-for="alphabeta">Alpha-Beta 深度<input id="${prefix}Depth" type="number" min="1" max="3" value="1"></label>`;
  $(`${prefix}Algorithm`).value=defaultAlgorithm;
  $(`${prefix}Algorithm`).onchange=()=>syncExperimentCard(side);
  syncExperimentCard(side);
}

function syncExperimentCard(side){
  const prefix=`exp${side.toUpperCase()}`, algorithm=$(`${prefix}Algorithm`).value;
  document.querySelectorAll(`[data-exp-player="${side}"] [data-for]`).forEach(field=>{
    const enabled=field.dataset.for.split(' ').includes(algorithm);
    field.hidden=false;
    field.classList.toggle('param-disabled',!enabled);
    const input=field.querySelector('input');
    if(input)input.disabled=!enabled;
  });
}

function syncExperimentSize(){
  const allowAlpha=$('expSize').value==='5';
  for(const side of ['a','b']){
    const prefix=`exp${side.toUpperCase()}`, select=$(`${prefix}Algorithm`), option=select.querySelector('option[value="alphabeta"]');
    option.disabled=!allowAlpha;
    if(!allowAlpha&&select.value==='alphabeta'){select.value='mcts';syncExperimentCard(side);}
  }
}

function experimentSpec(side){
  const prefix=`exp${side.toUpperCase()}`;
  return {algorithm:$(`${prefix}Algorithm`).value,C:Number($(`${prefix}C`).value),n_search:Number($(`${prefix}Searches`).value),n_rollout:Number($(`${prefix}Rollouts`).value),heuristic_weight:Number($(`${prefix}Weight`).value),heuristic_scale:Number($(`${prefix}Scale`).value),ab_depth:Number($(`${prefix}Depth`).value)};
}

function renderLastExperimentGame(game){
  $('lastGame').hidden=!game;
  if(!game)return;
  const host=$('lastBoard');host.innerHTML='';
  const n=game.size,pad=34,span=532,step=(span-2*pad)/(n-1),svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.classList.add('go-board');svg.setAttribute('viewBox','0 0 532 532');svg.style.background='#d6a85f';
  for(let i=0;i<n;i++){const p=pad+i*step;for(const [x1,y1,x2,y2] of [[pad,p,span-pad,p],[p,pad,p,span-pad]]){const line=document.createElementNS(svg.namespaceURI,'line');for(const [k,v] of Object.entries({x1,y1,x2,y2,stroke:'#3b2b18','stroke-width':2}))line.setAttribute(k,v);svg.appendChild(line);}}
  game.board.flat().forEach((stone,action)=>{if(!stone)return;const row=Math.floor(action/n),col=action%n,circle=document.createElementNS(svg.namespaceURI,'circle');circle.setAttribute('cx',pad+col*step);circle.setAttribute('cy',pad+row*step);circle.setAttribute('r',Math.min(18,step*.38));circle.setAttribute('fill',stone===1?'#151515':'#fafafa');circle.setAttribute('stroke',stone===1?'#000':'#777');svg.appendChild(circle);});
  host.appendChild(svg);$('lastGameResult').textContent=`${game.result} · ${game.history.length} 手`;$('lastHistory').innerHTML=game.history.map((move,index)=>`<span>${index+1}. ${move.player===1?'黑':'白'} ${move.label}</span>`).join('');
}

function renderExperiment(data){
  const percent=Number(data.percent||0),running=!!data.running;
  const phaseText={idle:'等待开始',starting:'正在准备对局…',playing:'正在运行批量对局…',cancelling:'正在等待当前搜索停止…',cancelled:'实验已取消',complete:'实验完成',error:'实验异常退出'}[data.phase]||data.phase;
  $('experimentStatus').textContent=phaseText;$('experimentPhase').textContent=phaseText;$('experimentNumbers').textContent=`${data.completed||0} / ${data.total||0} · ${percent.toFixed(0)}% · ${Number(data.elapsed||0).toFixed(1)} 秒`;
  $('experimentFill').style.width=`${percent}%`;$('experimentFill').classList.toggle('running',running);$('startExperiment').disabled=running;$('cancelExperiment').disabled=!running;$('copyExperiment').disabled=!(data.rows&&data.rows.length);
  document.querySelectorAll('.experiment-controls input,.experiment-controls select').forEach(control=>control.disabled=running);
  if(!running){syncExperimentCard('a');syncExperimentCard('b');syncExperimentSize();}
  $('experimentError').textContent=data.error||'';
  const rows=data.rows||[];$('experimentRows').innerHTML=rows.length?rows.map(row=>{const games=Math.max(1,row.games);return `<tr><td>${row.label}</td><td>${row.games}</td><td>${row.a_wins}</td><td>${row.b_wins}</td><td>${row.draws}</td><td>${(100*row.a_wins/games).toFixed(1)}%</td><td>${(100*(row.b_wins+row.draws)/games).toFixed(1)}%</td><td>${Number(row.elapsed).toFixed(2)} 秒</td></tr>`;}).join(''):'<tr><td colspan="8" class="empty">尚无数据</td></tr>';
  if(data.config){const a=data.config.player_a,b=data.config.player_b;$('experimentSummary').textContent=`A=${algorithmName(a.algorithm)}，B=${algorithmName(b.algorithm)}，${data.config.size}×${data.config.size}，种子 ${data.config.seed}。已完成 ${data.completed}/${data.total} 局，总用时 ${Number(data.elapsed||0).toFixed(2)} 秒。`;}
  renderLastExperimentGame(data.last_game);
}

let experimentTimer=null,latestExperiment=null;
async function pollExperiment(){try{const response=await fetch('/api/experiment',{cache:'no-store'});if(!response.ok)return;latestExperiment=await response.json();renderExperiment(latestExperiment);if(!latestExperiment.running&&experimentTimer){clearInterval(experimentTimer);experimentTimer=null;}}catch(_error){}}
async function startExperiment(){
  try{latestExperiment=await api('/api/experiment/start',{size:Number($('expSize').value),games:Number($('expGames').value),seed:Number($('expSeed').value),swap:$('expSwap').checked,player_a:experimentSpec('a'),player_b:experimentSpec('b')});renderExperiment(latestExperiment);if(experimentTimer)clearInterval(experimentTimer);experimentTimer=setInterval(pollExperiment,300);}
  catch(error){alert(error.message);}
}

$('startExperiment').onclick=startExperiment;
$('cancelExperiment').onclick=async()=>{try{renderExperiment(await api('/api/experiment/cancel'));if(!experimentTimer)experimentTimer=setInterval(pollExperiment,300);}catch(error){alert(error.message);}};
$('copyExperiment').onclick=async()=>{if(!latestExperiment)return;const lines=[$('experimentSummary').textContent,...(latestExperiment.rows||[]).map(row=>`${row.label}：${row.games} 局，A 胜 ${row.a_wins}，B 胜 ${row.b_wins}，和棋 ${row.draws}，用时 ${Number(row.elapsed).toFixed(2)} 秒`)];try{await navigator.clipboard.writeText(lines.join('\n'));$('copyExperiment').textContent='已复制';setTimeout(()=>$('copyExperiment').textContent='复制结果摘要',1200);}catch(error){alert('无法复制：'+error.message);}};
$('expSize').onchange=syncExperimentSize;
setupExperimentCard('a','mcts');setupExperimentCard('b','random');syncExperimentSize();pollExperiment();

$('size').onchange=()=>{
  const allowed=$('size').value==='5';
  for(const select of [$('black'),$('white')]){
    const option=select.querySelector('option[value="alphabeta"]');
    option.disabled=!allowed;
    if(!allowed&&select.value==='alphabeta')select.value='mcts';
  }
};
$('newGame').onclick=startNewGame;
$('startSearch').onclick=()=>runMCTS('/api/search');
$('skipSearch').onclick=()=>runMCTS('/api/skip');
$('nextEvent').onclick=()=>guarded(async()=>render(await api('/api/next')));
$('pass').onclick=()=>{ if(payload)humanMove(payload.state.size*payload.state.size); };
$('auto').onclick=()=>{ if(autoTimer)return; autoTimer=setInterval(()=>{ if(!busy&&payload&&payload.has_next)$('nextEvent').click(); else if(payload&&!payload.has_next)stopAuto(); },1000); $('auto').disabled=true;$('stopAuto').disabled=false; };
$('stopAuto').onclick=stopAuto;
function stopAuto(){ if(autoTimer){clearInterval(autoTimer);autoTimer=null;} $('stopAuto').disabled=true;if(payload)$('auto').disabled=!payload.has_next; }

fetch('/api/state').then(r=>r.json()).then(data=>{ $('size').value=String(data.state.size);$('black').value=data.state.players.black;$('white').value=data.state.players.white;$('cValue').value=data.state.config.C;$('searches').value=data.state.config.n_search;$('rollouts').value=data.state.config.n_rollout;$('abDepth').value=data.state.config.ab_depth;$('heuristicWeight').value=data.state.config.heuristic_weight;$('heuristicScale').value=data.state.config.heuristic_scale;$('size').onchange();render(data); });
</script>
</body>
</html>'''


class VisualizerHandler(BaseHTTPRequestHandler):
    session = GameSession()
    experiments = ExperimentManager()

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(_plain(payload), ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A new-game request intentionally abandons the older search HTTP
            # response.  The cancelled worker must exit quietly in that case.
            return

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
            self._send_json(self.session.payload())
        elif path == "/api/progress":
            self._send_json(self.session.search_progress())
        elif path == "/api/experiment":
            self._send_json(self.experiments.status())
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/new":
                result = {"state": self.session.new_game(data), "event": None, "trace_index": -1, "trace_count": 0, "has_next": False}
            elif path == "/api/move":
                result = {"state": self.session.human_move(int(data["action"])), "event": None, "trace_index": -1, "trace_count": 0, "has_next": False}
            elif path == "/api/search":
                result = self.session.prepare_search()
            elif path == "/api/next":
                result = self.session.next_event()
            elif path == "/api/skip":
                result = self.session.skip_search()
            elif path == "/api/experiment/start":
                result = self.experiments.start(data)
            elif path == "/api/experiment/cancel":
                result = self.experiments.cancel()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc), "traceback": traceback.format_exc()}, status=400)


def main() -> None:
    parser = argparse.ArgumentParser(description="围棋树搜索可视化调试器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="端口；0 表示自动选择")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"棋类树搜索可视化已启动：{url}")
    print("按 Ctrl+C 停止。")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
