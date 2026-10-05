"""Generate a self-contained interactive multi-game match viewer."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import chess

from .evaluation import EvaluationWeights, StaticEvaluator
from .measure import Style, move_distribution
from .search import AdaptiveExpectedValue


PIECE_GLYPHS = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def _style_from_dict(values: Dict[str, float]) -> Style:
    names = {field.name for field in fields(Style)}
    kwargs = {name: values[name] for name in names if name in values}
    return Style(**kwargs)


def _weights_from_dict(values: Dict[str, float]) -> EvaluationWeights:
    names = {field.name for field in fields(EvaluationWeights)}
    kwargs = {name: values[name] for name in names if name in values}
    return EvaluationWeights(**kwargs)


def _board_matrix(fen: str) -> List[List[Dict[str, str]]]:
    rows = []
    for rank_index, rank in enumerate(fen.split()[0].split("/")):
        row = []
        file_index = 0
        for char in rank:
            if char.isdigit():
                for _ in range(int(char)):
                    row.append(
                        {
                            "square": chess.square_name(
                                chess.square(file_index, 7 - rank_index)
                            ),
                            "piece": "",
                        }
                    )
                    file_index += 1
            else:
                row.append(
                    {
                        "square": chess.square_name(
                            chess.square(file_index, 7 - rank_index)
                        ),
                        "piece": PIECE_GLYPHS[char],
                    }
                )
                file_index += 1
        rows.append(row)
    return rows


def _landscape_payload(
    board: chess.Board,
    style: Style,
    beta: float,
    evaluator: StaticEvaluator,
    player_color: chess.Color,
    depth: int = 1,
    adaptive_c: float = 0.3,
    search_mode: str = "accurate",
) -> Dict[str, Any]:
    search = AdaptiveExpectedValue(style, beta, evaluator, depth=depth, adaptive_c=adaptive_c, search_mode=search_mode)
    landscape = search.landscape(board)
    records_by_probability = sorted(
        landscape.records,
        key=lambda item: (item.probability, item.static_after),
        reverse=True,
    )
    probability_rank = {record.uci: rank for rank, record in enumerate(records_by_probability, start=1)}

    current_u = search.expected_value(board)
    selection = search.node_selection(board)
    branches = {branch.uci: branch for branch in selection.branches} if selection else {}
    moves = []
    for record in landscape.records:
        branch = branches.get(record.uci)
        reply_expected = branch.observable_value if branch else record.static_after
        g_tilde = reply_expected - current_u
        moves.append(
            {
                **record.as_dict(),
                "probability_rank": probability_rank[record.uci],
                "reply_expected_value": reply_expected,
                "display_reply_expected": reply_expected,
                "g_tilde": g_tilde,
                "branch_value": reply_expected,
                "selected_for_refinement": branch.was_deepened if branch else False,
                "response_neff": branch.response_neff if branch else None,
                "response_k": branch.response_k if branch else None,
                "player_advantage": g_tilde,
            }
        )

    if board.turn == chess.WHITE:
        moves.sort(
            key=lambda item: (item["g_tilde"], item["probability"], item["display_reply_expected"]),
            reverse=True,
        )
    else:
        moves.sort(
            key=lambda item: (item["g_tilde"], -item["probability"], item["display_reply_expected"]),
        )
    best = moves[0] if moves else None
    for rank, move in enumerate(moves, start=1):
        move["advantage_rank"] = rank

    return {
        "expected_value": current_u,
        "display_expected_value": current_u,
        "entropy": landscape.entropy,
        "effective_moves": landscape.effective_moves,
        "best_advantage_move": best,
        "sort_direction": "max" if board.turn == chess.WHITE else "min",
        "depth": depth,
        "search_mode": search_mode,
        "moves": moves,
    }


def _reply_update_series(
    plies: List[Dict[str, Any]],
    side: str,
    style: Style,
    beta: float,
    evaluator: StaticEvaluator,
) -> List[Dict[str, Any]]:
    updates = []
    for index, item in enumerate(plies[:-1]):
        row = item["row"]
        if row.get("side") != side:
            continue
        expected = row.get("predicted_reply_expected_value")
        if expected is None:
            continue
        reply_row = plies[index + 1]["row"]
        board_after_reply = chess.Board(reply_row["fen_after"])
        recomputed = move_distribution(board_after_reply, style, beta, evaluator).expected_value
        updates.append(
            {
                "ply": int(row["ply"]),
                "reply_ply": int(reply_row["ply"]),
                "move": row["san"],
                "opponent_reply": reply_row["san"],
                "expected_before_reply": float(expected),
                "recomputed_after_reply": recomputed,
                "delta_u": recomputed - float(expected),
                "realized_static": evaluator.evaluate(board_after_reply),
            }
        )
    return updates


def build_game_payload(match_json: Path) -> Dict[str, Any]:
    source = json.loads(match_json.read_text(encoding="utf-8"))
    config = source.get("config", {})
    evaluator = StaticEvaluator(_weights_from_dict(source.get("evaluation_weights", {})))
    white_style = _style_from_dict(source.get("white_style", {}))
    black_style = _style_from_dict(source.get("black_style", {}))
    beta_white = float(config.get("beta_white", 4.0))
    beta_black = float(config.get("beta_black", 4.0))
    depth = int(config.get("depth", config.get("depth", 1)))
    search_mode = str(config.get("search_mode", "accurate"))
    adaptive_c = float(config.get("adaptive_c", 0.3))
    plies = source.get("plies", [])
    initial_fen = plies[0]["row"]["fen_before"] if plies else chess.STARTING_FEN

    state_specs = [
        {
            "index": 0,
            "ply": 0,
            "move_label": "Start",
            "side": "",
            "san": "",
            "uci": "",
            "fen": initial_fen,
            "prediction_error": None,
        }
    ]
    for item in plies:
        row = item["row"]
        state_specs.append(
            {
                "index": int(row["ply"]),
                "ply": int(row["ply"]),
                "move_label": f"{row['ply']}. {row['side']} {row['san']}",
                "side": row["side"],
                "san": row["san"],
                "uci": row["uci"],
                "fen": row["fen_after"],
                "prediction_error": row.get("prediction_error"),
                "played_row": row,
            }
        )

    states = []
    for spec in state_specs:
        board = chess.Board(spec["fen"])
        states.append(
            {
                **spec,
                "turn": "white" if board.turn == chess.WHITE else "black",
                "fullmove_number": board.fullmove_number,
                "is_game_over": board.is_game_over(claim_draw=True),
                "result": board.result(claim_draw=True)
                if board.is_game_over(claim_draw=True)
                else "*",
                "board": _board_matrix(spec["fen"]),
                "static_evaluation": evaluator.evaluate(board),
                "white_panel": _landscape_payload(board, white_style, beta_white, evaluator, chess.WHITE, depth, adaptive_c, search_mode),
                "black_panel": _landscape_payload(board, black_style, beta_black, evaluator, chess.BLACK, depth, adaptive_c, search_mode),
            }
        )

    return {
        "id": match_json.stem,
        "source_file": str(match_json),
        "label": match_json.stem.replace("_", " "),
        "result": source.get("result", "*"),
        "final_fen": source.get("final_fen", ""),
        "white_style": white_style.as_dict(),
        "black_style": black_style.as_dict(),
        "evaluation_weights": evaluator.weights.as_dict(),
        "white_reply_updates": _reply_update_series(
            plies, "white", white_style, beta_white, evaluator
        ),
        "black_reply_updates": _reply_update_series(
            plies, "black", black_style, beta_black, evaluator
        ),
        "states": states,
    }


def build_platform_payload(results_dir: Path = Path("data/games/json")) -> Dict[str, Any]:
    games = []
    for match_json in sorted(results_dir.glob("*.json")):
        try:
            games.append(build_game_payload(match_json))
        except Exception as exc:  # keep one malformed result from breaking the platform
            games.append(
                {
                    "id": match_json.stem,
                    "source_file": str(match_json),
                    "label": f"{match_json.stem} (failed to load)",
                    "error": str(exc),
                    "states": [],
                }
            )
    return {"results_dir": str(results_dir), "games": games}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thermodynamic Chess Match Viewer</title>
  <style>
    :root { --bg: #f4f1ea; --ink: #161616; --muted: #66615a; --line: #c8c1b5; --panel: #fffdfa; --light: #e9d8b7; --dark: #58806a; --accent: #176b87; --accent-2: #9a3412; --good: #146c43; --bad: #b42318; color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; } body { margin: 0; background: var(--bg); color: var(--ink); }
    .app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header { border-bottom: 1px solid var(--line); background: #fffaf2; padding: 12px 18px; display: grid; grid-template-columns: minmax(260px, 1fr) auto; gap: 14px; align-items: center; }
    h1 { margin: 0; font-size: 18px; font-weight: 760; letter-spacing: 0; }
    .meta { color: var(--muted); font-size: 13px; display: flex; gap: 14px; flex-wrap: wrap; }
    .game-select { display: flex; gap: 8px; align-items: center; justify-content: end; }
    select { height: 36px; min-width: 220px; border: 1px solid var(--line); background: var(--panel); border-radius: 6px; color: var(--ink); padding: 0 8px; }
    main { display: grid; grid-template-columns: minmax(330px, 430px) 1fr; gap: 16px; padding: 16px; align-items: start; }
    .board-zone { display: grid; gap: 12px; position: sticky; top: 12px; }
    .board-wrap { width: min(100%, 430px); aspect-ratio: 1 / 1; border: 1px solid #2f3b35; box-shadow: 0 12px 32px rgb(35 31 26 / 16%); }
    .board { display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr); width: 100%; height: 100%; }
    .sq { position: relative; display: grid; place-items: center; font-size: clamp(28px, 7.5vw, 48px); line-height: 1; }
    .sq.light { background: var(--light); } .sq.dark { background: var(--dark); }
    .sq.last-from::after, .sq.last-to::after { content: ""; position: absolute; inset: 7%; border: 3px solid rgba(255, 232, 82, 0.9); border-radius: 4px; pointer-events: none; }
    .coord { position: absolute; left: 4px; bottom: 3px; font-size: 10px; color: rgb(0 0 0 / 56%); }
    .controls { display: grid; grid-template-columns: 42px 42px 1fr 42px 42px; gap: 8px; align-items: center; }
    button { height: 38px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); border-radius: 6px; font-size: 18px; cursor: pointer; }
    button:hover { border-color: var(--accent); } input[type="range"] { width: 100%; accent-color: var(--accent); }
    .position-strip { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .position-strip.compact { grid-template-columns: repeat(3, 1fr); }
    .stat, .panel, .notice { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .stat, .notice { padding: 10px 12px; }
    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 18px; font-weight: 720; margin-top: 2px; }
    .content { display: grid; gap: 16px; }
    .panels { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }
    .panel { overflow: hidden; } .panel-head { padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .panel-title { font-size: 16px; font-weight: 760; } .style { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .metric { min-width: 0; } .metric .value { font-size: 15px; }
    .best { padding: 0 14px 12px; color: var(--muted); font-size: 13px; }
    .eval-bar-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
    .eval-bar-labels { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    .eval-bar-track { position: relative; height: 18px; border-radius: 999px; overflow: hidden; background: linear-gradient(90deg, #2f3b35 0 50%, #f7ead1 50% 100%); border: 1px solid var(--line); }
    .eval-bar-fill { position: absolute; top: 0; bottom: 0; left: 50%; width: 0; background: var(--accent); }
    .eval-bar-fill.black { left: auto; right: 50%; background: var(--accent-2); }
    .eval-bar-zero { position: absolute; top: -2px; bottom: -2px; left: 50%; width: 2px; background: #fffdfa; box-shadow: 0 0 0 1px rgb(0 0 0 / 18%); }
    .table-note { padding: 0 14px 8px; color: var(--muted); font-size: 12px; }
    .table-wrap { max-height: 420px; overflow: auto; } table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid #eee5da; text-align: right; white-space: nowrap; }
    th { position: sticky; top: 0; background: #fff7eb; color: var(--muted); font-weight: 700; z-index: 1; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
    tr.played td { background: #e7f4f5; } tr.best td { box-shadow: inset 3px 0 0 var(--accent-2); }
    .positive { color: var(--good); } .negative { color: var(--bad); }
    .move-list { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; max-height: 180px; overflow: auto; font-size: 13px; line-height: 1.8; }
    .move-chip { display: inline-block; border: 1px solid transparent; border-radius: 5px; padding: 0 5px; cursor: pointer; }
    .move-chip:hover { border-color: var(--accent); } .move-chip.active { background: var(--accent); color: white; }
    .fen { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
    @media (max-width: 1050px) { header { grid-template-columns: 1fr; } .game-select { justify-content: start; } main { grid-template-columns: 1fr; } .board-zone { position: static; justify-items: center; } .panels { grid-template-columns: 1fr; } }
    @media (max-width: 620px) { main { padding: 10px; } .metrics { grid-template-columns: 1fr 1fr; } .controls { grid-template-columns: 38px 38px 1fr 38px 38px; } }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div><h1>Thermodynamic Chess Match Viewer</h1><div class="meta"><span id="source"></span><span id="result"></span><span id="ply-count"></span></div></div>
      <div class="game-select"><label class="label" for="game-select">Game</label><select id="game-select"></select></div>
    </header>
    <main id="main">
      <section class="board-zone">
        <div class="board-wrap"><div class="board" id="board"></div></div>
        <div class="controls"><button id="first" title="First position">⏮</button><button id="prev" title="Previous ply">◀</button><input id="slider" type="range" min="0" value="0"><button id="next" title="Next ply">▶</button><button id="last" title="Last position">⏭</button></div>
        <div class="position-strip compact"><div class="stat"><div class="label">Position</div><div class="value" id="position-label"></div></div><div class="stat"><div class="label">Side To Move</div><div class="value" id="turn"></div></div><div class="stat"><div class="label">Static E(B)</div><div class="value" id="static-v"></div></div></div>
        <div class="eval-bar-card"><div class="eval-bar-labels"><span id="white-e-label"></span><span id="black-e-label"></span></div><div class="eval-bar-track"><div class="eval-bar-fill" id="white-e-fill"></div><div class="eval-bar-fill black" id="black-e-fill"></div><div class="eval-bar-zero"></div></div></div>
        <div class="fen" id="fen"></div><div class="move-list" id="move-list"></div>
      </section>
      <section class="content"><div class="panels"><article class="panel" id="white-panel"></article><article class="panel" id="black-panel"></article></div></section>
    </main>
  </div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const platform = JSON.parse(document.getElementById('payload').textContent);
    let game = null; let states = []; let index = 0;
    const els = { gameSelect: document.getElementById('game-select'), source: document.getElementById('source'), result: document.getElementById('result'), plyCount: document.getElementById('ply-count'), fen: document.getElementById('fen'), board: document.getElementById('board'), slider: document.getElementById('slider'), first: document.getElementById('first'), prev: document.getElementById('prev'), next: document.getElementById('next'), last: document.getElementById('last'), positionLabel: document.getElementById('position-label'), turn: document.getElementById('turn'), staticV: document.getElementById('static-v'), whiteELabel: document.getElementById('white-e-label'), blackELabel: document.getElementById('black-e-label'), whiteEFill: document.getElementById('white-e-fill'), blackEFill: document.getElementById('black-e-fill'), moveList: document.getElementById('move-list'), whitePanel: document.getElementById('white-panel'), blackPanel: document.getElementById('black-panel'), main: document.getElementById('main') };
    function fmt(x, digits = 3) { if (x === null || x === undefined || Number.isNaN(Number(x))) return '—'; return Number(x).toFixed(digits); }
    function signedClass(x) { const n = Number(x); if (Number.isNaN(n) || Math.abs(n) < 1e-9) return ''; return n > 0 ? 'positive' : 'negative'; }
    function lastSquares(state) { if (!state.uci) return new Set(); return new Set([state.uci.slice(0, 2), state.uci.slice(2, 4)]); }
    function renderBoard(state) { const highlights = lastSquares(state); els.board.innerHTML = ''; state.board.flat().forEach((sq, i) => { const div = document.createElement('div'); const rank = Math.floor(i / 8); const file = i % 8; div.className = `sq ${(rank + file) % 2 === 0 ? 'light' : 'dark'}`; if (highlights.has(sq.square)) div.classList.add(sq.square === state.uci?.slice(0, 2) ? 'last-from' : 'last-to'); div.innerHTML = `<span>${sq.piece}</span>`; if (file === 0 || rank === 7) { const coord = document.createElement('span'); coord.className = 'coord'; coord.textContent = rank === 7 ? sq.square[0] : sq.square[1]; div.appendChild(coord); } els.board.appendChild(div); }); }
    function renderEvalBar(eValue) { const scale = 20; const white = Number(eValue); const black = -white; const pct = Math.min(Math.abs(white) / scale, 1) * 50; els.whiteELabel.textContent = `White E ${fmt(white)}`; els.blackELabel.textContent = `Black E ${fmt(black)}`; els.whiteEFill.style.width = white > 0 ? `${pct}%` : '0%'; els.blackEFill.style.width = white < 0 ? `${pct}%` : '0%'; }
    function feature(move, name) { return move.features && move.features[name] !== undefined ? move.features[name] : 0; }
    function renderPanel(target, title, style, panel, state) { const best = panel.best_advantage_move; const played = state.uci; const styleText = Object.entries(style).map(([k, v]) => `${k} ${Number(v).toFixed(2)}`).join(' · '); const rows = panel.moves.map(move => { const cls = [move.uci === played ? 'played' : '', best && move.uci === best.uci ? 'best' : ''].filter(Boolean).join(' '); return `<tr class="${cls}"><td>${move.advantage_rank}</td><td>${move.san}</td><td>${fmt(move.probability, 4)}</td><td class="${signedClass(move.display_reply_expected)}">${fmt(move.display_reply_expected)}</td><td class="${signedClass(move.delta_u)}">${fmt(move.delta_u)}</td><td class="${signedClass(feature(move, 'material'))}">${fmt(feature(move, 'material'))}</td><td class="${signedClass(feature(move, 'center'))}">${fmt(feature(move, 'center'))}</td><td class="${signedClass(feature(move, 'development'))}">${fmt(feature(move, 'development'))}</td><td class="${signedClass(feature(move, 'castling'))}">${fmt(feature(move, 'castling'))}</td><td class="${signedClass(feature(move, 'king_safety'))}">${fmt(feature(move, 'king_safety'))}</td><td class="${signedClass(feature(move, 'king_pressure'))}">${fmt(feature(move, 'king_pressure'))}</td></tr>`; }).join(''); target.innerHTML = `<div class="panel-head"><div class="panel-title">${title}</div><div class="style">${styleText}</div></div><div class="metrics"><div class="metric"><div class="label">U(B)</div><div class="value ${signedClass(panel.display_expected_value)}">${fmt(panel.display_expected_value)}</div></div><div class="metric"><div class="label">Entropy</div><div class="value">${fmt(panel.entropy)}</div></div><div class="metric"><div class="label">N_eff</div><div class="value">${fmt(panel.effective_moves)}</div></div></div><div class="best">${panel.sort_direction === 'max' ? 'Max deltaU move:' : 'Min deltaU move:'} <strong>${best ? best.san : 'none'}</strong>${best ? ` · deltaU=${fmt(best.delta_u)} · U(Bm)=${fmt(best.display_reply_expected)}` : ''}</div><div class="table-note">${panel.sort_direction === 'max' ? 'Moves are sorted by descending deltaU.' : 'Moves are sorted by ascending deltaU.'} White maximizes deltaU; Black minimizes deltaU. U(B), U(Bm), and deltaU are White-positive.</div><div class="table-wrap"><table><thead><tr><th>#</th><th>move</th><th>p(B)</th><th>U(Bm)</th><th>deltaU</th><th>mat</th><th>ctr</th><th>dev</th><th>cas</th><th>K safe</th><th>K press</th></tr></thead><tbody>${rows || '<tr><td colspan="11">No legal moves</td></tr>'}</tbody></table></div>`; }
    function renderMoveList() { els.moveList.innerHTML = states.map((state, i) => `<span class="move-chip ${i === index ? 'active' : ''}" data-index="${i}">${i === 0 ? 'Start' : `${state.ply}. ${state.san}`}</span>`).join(' '); els.moveList.querySelectorAll('.move-chip').forEach(chip => chip.addEventListener('click', () => setIndex(Number(chip.dataset.index)))); }
    function setIndex(nextIndex) { if (!states.length) return; index = Math.max(0, Math.min(states.length - 1, nextIndex)); const state = states[index]; els.slider.value = index; els.fen.textContent = state.fen; els.positionLabel.textContent = state.move_label; els.turn.textContent = state.is_game_over ? `game over ${state.result}` : state.turn; els.staticV.textContent = fmt(state.static_evaluation); els.staticV.className = `value ${signedClass(state.static_evaluation)}`; renderEvalBar(state.static_evaluation); renderBoard(state); renderPanel(els.whitePanel, 'White Style Measure', game.white_style, state.white_panel, state); renderPanel(els.blackPanel, 'Black Style Measure', game.black_style, state.black_panel, state); renderMoveList(); }
    function setGame(gameIndex) { game = platform.games[gameIndex]; if (!game || game.error || !game.states.length) { els.main.innerHTML = `<div class="notice">No loadable games were embedded from ${platform.results_dir}.</div>`; return; } states = game.states; index = 0; els.source.textContent = game.source_file; els.result.textContent = `Result ${game.result}`; els.plyCount.textContent = `${states.length - 1} plies`; els.slider.max = states.length - 1; setIndex(0); }
    platform.games.forEach((candidate, i) => { const option = document.createElement('option'); option.value = String(i); option.textContent = `${candidate.label} · ${candidate.result || 'unknown'} · ${(candidate.states || []).length ? candidate.states.length - 1 : 0} plies`; els.gameSelect.appendChild(option); });
    els.gameSelect.addEventListener('change', event => setGame(Number(event.target.value))); els.slider.addEventListener('input', event => setIndex(Number(event.target.value))); els.first.addEventListener('click', () => setIndex(0)); els.prev.addEventListener('click', () => setIndex(index - 1)); els.next.addEventListener('click', () => setIndex(index + 1)); els.last.addEventListener('click', () => setIndex(states.length - 1)); window.addEventListener('keydown', event => { if (event.key === 'ArrowLeft') setIndex(index - 1); if (event.key === 'ArrowRight') setIndex(index + 1); if (event.key === 'Home') setIndex(0); if (event.key === 'End') setIndex(states.length - 1); }); setGame(0);
  </script>
</body>
</html>
"""


def write_match_viewer(
    results_dir: Path = Path("data/games/json"),
    output_html: Path = Path("match_viewer.html"),
) -> Path:
    payload = build_platform_payload(results_dir)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        HTML_TEMPLATE.replace("__PAYLOAD__", payload_json), encoding="utf-8"
    )
    return output_html


def write_interactive_viewer(
    match_json: Path = Path("data/games/json/thermo_match.json"),
    output_html: Path = Path("match_viewer.html"),
) -> Path:
    """Backward-compatible wrapper for older callers.

    The platform now embeds every JSON match in the selected file's directory.
    """

    return write_match_viewer(match_json.parent, output_html)
