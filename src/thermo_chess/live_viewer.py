"""Dynamic browser viewer for saved thermodynamic chess matches."""

from __future__ import annotations

import argparse
import json
import posixpath
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse



LIVE_VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thermodynamic Chess Live Viewer</title>
  <style>
    :root { --bg: #f4f1ea; --ink: #161616; --muted: #66615a; --line: #c8c1b5; --panel: #fffdfa; --light: #e9d8b7; --dark: #58806a; --accent: #176b87; --accent-2: #9a3412; --good: #146c43; --bad: #b42318; color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; } body { margin: 0; background: var(--bg); color: var(--ink); }
    .app { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
    header { border-bottom: 1px solid var(--line); background: #fffaf2; padding: 12px 18px; display: grid; grid-template-columns: minmax(260px, 1fr) auto; gap: 14px; align-items: center; }
    h1 { margin: 0; font-size: 18px; font-weight: 760; letter-spacing: 0; }
    .meta { color: var(--muted); font-size: 13px; display: flex; gap: 14px; flex-wrap: wrap; }
    .game-select { display: flex; gap: 8px; align-items: center; justify-content: end; flex-wrap: wrap; }
    select { height: 36px; min-width: 280px; border: 1px solid var(--line); background: var(--panel); border-radius: 6px; color: var(--ink); padding: 0 8px; }
    button { height: 36px; border: 1px solid var(--line); background: var(--panel); color: var(--ink); border-radius: 6px; font-size: 15px; cursor: pointer; padding: 0 10px; }
    button:hover { border-color: var(--accent); }
    main { display: grid; grid-template-columns: minmax(330px, 430px) 1fr; gap: 16px; padding: 16px; align-items: start; }
    .board-zone { display: grid; gap: 12px; position: sticky; top: 12px; }
    .board-stage { width: min(100%, 480px); display: grid; grid-template-columns: 40px minmax(0, 430px); gap: 10px; align-items: stretch; }
    .board-wrap { width: 100%; aspect-ratio: 1 / 1; border: 1px solid #2f3b35; box-shadow: 0 12px 32px rgb(35 31 26 / 16%); }
    .captured-row { width: min(100%, 430px); min-height: 30px; display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 4px 8px; background: rgb(255 253 250 / 72%); border: 1px solid var(--line); border-radius: 6px; }
    .captured-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .captured-pieces { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; font-size: 18px; line-height: 1; }
    .captured-group { white-space: nowrap; }
    .captured-adv { font-weight: 760; min-width: 34px; text-align: right; }
    .board { display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr); width: 100%; height: 100%; }
    .sq { position: relative; display: grid; place-items: center; font-size: clamp(28px, 7.5vw, 48px); line-height: 1; }
    .sq.light { background: var(--light); } .sq.dark { background: var(--dark); }
    .sq.last-from::after, .sq.last-to::after { content: ""; position: absolute; inset: 7%; border: 3px solid rgba(255, 232, 82, 0.9); border-radius: 4px; pointer-events: none; }
    .coord { position: absolute; left: 4px; bottom: 3px; font-size: 10px; color: rgb(0 0 0 / 56%); }
    .controls { display: grid; grid-template-columns: 42px 42px 1fr 42px 42px; gap: 8px; align-items: center; }
    .controls button { font-size: 18px; padding: 0; } input[type="range"] { width: 100%; accent-color: var(--accent); }
    .position-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .stat, .panel, .notice { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .stat, .notice { padding: 10px 12px; }
    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
    .value { font-size: 18px; font-weight: 720; margin-top: 2px; }
    .content { display: grid; gap: 16px; min-width: 0; }
    .panels { display: grid; gap: 10px; align-items: start; }
    .panel { overflow: hidden; } .panel-head { padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .comparison-table th.group-head { text-align: center; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink); }
    .comparison-table .shared-col { text-align: center; background: #fffaf2; }
    .comparison-table .shared-start { border-left: 2px solid var(--line); }
    .comparison-table .shared-end { border-right: 2px solid var(--line); }
    .panel-title { font-size: 16px; font-weight: 760; } .style { color: var(--muted); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .metric { min-width: 0; } .metric .value { font-size: 15px; }
    .best { padding: 0 14px 12px; color: var(--muted); font-size: 13px; }
    .eval-bar-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 7px; display: grid; min-height: 100%; }
    .eval-bar-labels { display: none; }
    .eval-bar-labels span { writing-mode: vertical-rl; justify-self: center; white-space: nowrap; line-height: 1; }
    #white-e-label { grid-row: 1; transform: rotate(180deg); }
    #black-e-label { grid-row: 3; }
    .eval-bar-track { position: relative; width: 24px; min-height: 0; height: 100%; justify-self: center; border-radius: 999px; overflow: hidden; background: linear-gradient(180deg, #111 0 50%, #fffdfa 50% 100%); border: 1px solid var(--line); }
    .eval-bar-fill { position: absolute; left: 0; right: 0; bottom: 50%; height: 0; background: var(--good); }
    .eval-bar-fill.black { top: 50%; bottom: auto; background: var(--bad); }
    .eval-bar-marker { position: absolute; left: -5px; right: -5px; top: 50%; height: 8px; border-radius: 999px; background: #fffdfa; border: 1px solid rgb(0 0 0 / 45%); transform: translateY(-50%); box-shadow: 0 1px 4px rgb(0 0 0 / 22%); }
    .eval-bar-marker.black { background: #111; border-color: #fffdfa; }
    .eval-bar-zero { position: absolute; left: -2px; right: -2px; top: 50%; height: 2px; background: #fffdfa; box-shadow: 0 0 0 1px rgb(0 0 0 / 18%); }
    .table-note { padding: 0 14px 8px; color: var(--muted); font-size: 12px; }
    .cycle-panel { height: 126px; border: 1px solid var(--line); border-radius: 8px; background: #fffdfa; overflow: hidden; }
    .cycle-panel.empty { opacity: .72; }
    .cycle-head { padding: 9px 10px 4px; }
    .cycle-title { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; font-weight: 760; }
    .cycle-content { display: grid; grid-template-columns: minmax(180px, .68fr) minmax(290px, 1.32fr); gap: 10px; padding: 5px 10px 10px; align-items: start; }
    .cycle-route-block { display: grid; grid-template-rows: auto auto; gap: 8px; min-width: 0; align-content: start; }
    .cycle-route { display: flex; align-items: center; gap: 7px; min-width: 0; flex-wrap: wrap; }
    .cycle-move { min-width: 42px; border-radius: 6px; border: 1px solid var(--line); padding: 4px 8px; text-align: center; font-weight: 760; }
    .cycle-move.white { background: #fff; color: #111; border-color: #aaa; }
    .cycle-move.black { background: #111; color: #fff; border-color: #111; }
    .cycle-arrow { color: var(--muted); }
    .cycle-box { min-width: 0; min-height: 74px; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: #fffaf2; display: grid; grid-template-rows: .9fr 1.1fr; }
    .cycle-box-half { min-width: 0; padding: 5px 7px; display: grid; align-content: center; }
    .cycle-box-half + .cycle-box-half { border-top: 1px solid var(--line); }
    .cycle-metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 5px; }
    .cycle-ply-metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; align-items: start; }
    .cycle-metric { min-width: 0; font-size: 12px; }
    .cycle-metric .label { font-size: 10px; }
    .cycle-metric .value { font-size: 13px; }
    .observer-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px; align-items: start; }
    .observer-column { display: grid; grid-template-rows: auto auto; align-content: start; align-items: stretch; gap: 8px; min-width: 0; }
    .observer-panel { min-width: 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fffdfa; }
    .observer-panel .panel-head { background: #fff7eb; }
    .tree-trigger { height: auto; min-height: 28px; padding: 2px 8px; font-size: 12px; text-align: left; }
    .tree-trigger.white { border-color: #a3a3a3; background: #fff; }
    .tree-trigger.black { border-color: #111; background: #111; color: #fff; }
    .tree-trigger:disabled { cursor: default; opacity: .55; }
    .tree-overlay { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; padding: 18px; background: rgb(22 22 22 / 38%); z-index: 30; }
    .tree-overlay.open { display: flex; }
    .tree-modal { width: min(720px, 100%); max-height: min(760px, 92vh); overflow: hidden; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 18px 60px rgb(0 0 0 / 28%); display: grid; grid-template-rows: auto 1fr; }
    .tree-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
    .tree-title { font-size: 16px; font-weight: 760; }
    .tree-body { padding: 12px 14px; overflow: auto; }
    .tree-root { border-left: 3px solid var(--accent); padding-left: 10px; margin-bottom: 12px; }
    .tree-list { display: grid; gap: 8px; margin-left: 14px; }
    .tree-node { border-left: 2px solid var(--line); padding: 6px 0 6px 10px; }
    .tree-line { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
    .tree-move { font-weight: 720; }
    .table-wrap { max-height: 520px; overflow: auto; } table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 7px 8px; border-bottom: 1px solid #eee5da; text-align: right; white-space: nowrap; }
    th { position: sticky; top: 0; background: #fff7eb; color: var(--muted); font-weight: 700; z-index: 1; }
    th:first-child, td:first-child, td.shared-move { text-align: left; }
    tr.played td { background: #e7f4f5; }
    tr.white-best td { border-top: 3px solid #a3a3a3; border-bottom: 3px solid #a3a3a3; }
    tr.white-best td:first-child { border-left: 3px solid #a3a3a3; }
    tr.white-best td:last-child { border-right: 3px solid #a3a3a3; }
    tr.black-best td { border-top: 3px solid #111; border-bottom: 3px solid #111; }
    tr.black-best td:first-child { border-left: 3px solid #111; }
    tr.black-best td:last-child { border-right: 3px solid #111; }
    .positive { color: var(--good); } .negative { color: var(--bad); }
    .move-list { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; max-height: 180px; overflow: auto; font-size: 13px; line-height: 1.8; }
    .move-chip { display: inline-block; border: 1px solid transparent; border-radius: 5px; padding: 0 5px; cursor: pointer; }
    .move-chip:hover { border-color: var(--accent); } .move-chip.active { background: var(--accent); color: white; }
    .fen { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
    .loading { padding: 16px; color: var(--muted); }
    .positive-bg { background: #edf8f2; }
    @media (max-width: 1050px) { header { grid-template-columns: 1fr; } .game-select { justify-content: start; } main { grid-template-columns: 1fr; } .board-zone { position: static; justify-items: center; } .panels, .observer-grid { grid-template-columns: 1fr; } }
    @media (max-width: 620px) { main { padding: 10px; } .metrics { grid-template-columns: 1fr 1fr; } .controls { grid-template-columns: 38px 38px 1fr 38px 38px; } select { min-width: 180px; width: 100%; } }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div><h1>Thermodynamic Chess Live Viewer</h1><div class="meta"><span id="source"></span><span id="result"></span><span id="ply-count"></span><span id="status"></span></div></div>
      <div class="game-select"><label class="label" for="game-select">Game</label><select id="game-select"></select><button id="refresh" title="Refresh games">Refresh</button></div>
    </header>
    <main id="main">
      <section class="board-zone">
        <div class="captured-row" id="captured-black"><span class="captured-label">Black captured</span><span class="captured-pieces"></span><span class="captured-adv"></span></div>
        <div class="board-stage"><div class="eval-bar-card"><div class="eval-bar-labels"><span id="white-e-label"></span><span id="black-e-label"></span></div><div class="eval-bar-track"><div class="eval-bar-fill" id="white-e-fill"></div><div class="eval-bar-fill black" id="black-e-fill"></div><div class="eval-bar-zero"></div><div class="eval-bar-marker" id="eval-marker"></div></div></div><div class="board-wrap"><div class="board" id="board"></div></div></div>
        <div class="captured-row" id="captured-white"><span class="captured-label">White captured</span><span class="captured-pieces"></span><span class="captured-adv"></span></div>
        <div class="controls"><button id="first" title="First position">⏮</button><button id="prev" title="Previous ply">◀</button><input id="slider" type="range" min="0" value="0"><button id="next" title="Next ply">▶</button><button id="last" title="Last position">⏭</button></div>
        <div class="position-strip"><div class="stat"><div class="label">Position</div><div class="value" id="position-label"></div></div><div class="stat"><div class="label">Side To Move</div><div class="value" id="turn"></div></div><div class="stat"><div class="label">Static E(B)</div><div class="value" id="static-v"></div></div></div>
        <div class="fen" id="fen"></div><div class="move-list" id="move-list"></div>
      </section>
      <section class="content"><div class="panels"><article class="panel comparison-panel" id="comparison-panel"><div class="loading">Loading games...</div></article></div></section>
    </main>
  </div>
  <div class="tree-overlay" id="tree-overlay" aria-hidden="true"><div class="tree-modal" role="dialog" aria-modal="true" aria-labelledby="tree-title"><div class="tree-head"><div><div class="tree-title" id="tree-title">Predicted Move Tree</div><div class="style" id="tree-subtitle"></div></div><button id="tree-close" title="Close">Close</button></div><div class="tree-body" id="tree-body"></div></div></div>
  <script>
    const PIECES = { P:'♙', N:'♘', B:'♗', R:'♖', Q:'♕', K:'♔', p:'♟', n:'♞', b:'♝', r:'♜', q:'♛', k:'♚' };
    let summaries = []; let game = null; let states = []; let index = 0; let loadedId = '';
    const els = { gameSelect: document.getElementById('game-select'), refresh: document.getElementById('refresh'), source: document.getElementById('source'), result: document.getElementById('result'), plyCount: document.getElementById('ply-count'), status: document.getElementById('status'), fen: document.getElementById('fen'), board: document.getElementById('board'), slider: document.getElementById('slider'), first: document.getElementById('first'), prev: document.getElementById('prev'), next: document.getElementById('next'), last: document.getElementById('last'), positionLabel: document.getElementById('position-label'), turn: document.getElementById('turn'), staticV: document.getElementById('static-v'), whiteELabel: document.getElementById('white-e-label'), blackELabel: document.getElementById('black-e-label'), whiteEFill: document.getElementById('white-e-fill'), blackEFill: document.getElementById('black-e-fill'), evalMarker: document.getElementById('eval-marker'), moveList: document.getElementById('move-list'), comparisonPanel: document.getElementById('comparison-panel'), treeOverlay: document.getElementById('tree-overlay'), treeTitle: document.getElementById('tree-title'), treeSubtitle: document.getElementById('tree-subtitle'), treeBody: document.getElementById('tree-body'), treeClose: document.getElementById('tree-close'), capturedWhite: document.getElementById('captured-white'), capturedBlack: document.getElementById('captured-black') };
    function fmt(x, digits = 3) { if (x === null || x === undefined || Number.isNaN(Number(x))) return '—'; return Number(x).toFixed(digits); }
    function signedClass(x) { const n = Number(x); if (Number.isNaN(n) || Math.abs(n) < 1e-9) return ''; return n > 0 ? 'positive' : 'negative'; }
    function turnFromFen(fen) { return fen.split(' ')[1] === 'w' ? 'white' : 'black'; }

    const PIECE_ORDER = ['Q', 'R', 'B', 'N', 'P'];
    const PIECE_VALUES = { P: 1, N: 3, B: 3, R: 5, Q: 9 };
    const BASE_COUNTS = { P: 8, N: 2, B: 2, R: 2, Q: 1 };
    function capturedFromFen(fen) { const counts = { white: { P:0,N:0,B:0,R:0,Q:0 }, black: { P:0,N:0,B:0,R:0,Q:0 } }; const board = (fen || '').split(' ')[0] || ''; for (const ch of board) { const up = ch.toUpperCase(); if (!PIECE_VALUES[up]) continue; if (ch === up) counts.white[up] += 1; else counts.black[up] += 1; } const sidePayload = (capturedColor) => { const pieces = {}; let value = 0; PIECE_ORDER.forEach(piece => { const count = Math.max(0, BASE_COUNTS[piece] - counts[capturedColor][piece]); pieces[piece] = count; value += count * PIECE_VALUES[piece]; }); return { pieces, value }; }; const whiteValue = Object.entries(counts.white).reduce((sum, [p,c]) => sum + c * PIECE_VALUES[p], 0); const blackValue = Object.entries(counts.black).reduce((sum, [p,c]) => sum + c * PIECE_VALUES[p], 0); const balance = whiteValue - blackValue; return { white: sidePayload('black'), black: sidePayload('white'), balance, white_advantage: Math.max(0, balance), black_advantage: Math.max(0, -balance) }; }
    function renderCapturedSide(target, payload, side, balance) { const pieces = target.querySelector('.captured-pieces'); const adv = target.querySelector('.captured-adv'); const symbols = side === 'white' ? { Q:'♛', R:'♜', B:'♝', N:'♞', P:'♟' } : { Q:'♕', R:'♖', B:'♗', N:'♘', P:'♙' }; pieces.innerHTML = PIECE_ORDER.map(piece => { const count = Number(payload?.pieces?.[piece] || 0); if (!count) return ''; return `<span class="captured-group">${symbols[piece].repeat(count)}</span>`; }).join('') || '<span class="style">none</span>'; const sideAdv = side === 'white' ? Number(balance || 0) : -Number(balance || 0); adv.textContent = sideAdv > 0 ? `+${fmt(sideAdv, 0)}` : ''; adv.className = 'captured-adv'; }
    function renderCapturedMaterial(state) { const captured = state.captured_material || capturedFromFen(state.fen); renderCapturedSide(els.capturedBlack, captured.black, 'black', captured.balance); renderCapturedSide(els.capturedWhite, captured.white, 'white', captured.balance); }
    function boardMatrix(fen) { const rows = []; const ranks = fen.split(' ')[0].split('/'); for (let r = 0; r < ranks.length; r++) { const row = []; let file = 0; for (const ch of ranks[r]) { if (/\d/.test(ch)) { for (let i = 0; i < Number(ch); i++) row.push({ square: 'abcdefgh'[file++] + String(8 - r), piece: '' }); } else { row.push({ square: 'abcdefgh'[file++] + String(8 - r), piece: PIECES[ch] || ch }); } } rows.push(row); } return rows; }
    function lastSquares(state) { if (!state.uci) return new Set(); return new Set([state.uci.slice(0, 2), state.uci.slice(2, 4)]); }
    function renderBoard(state) { const highlights = lastSquares(state); els.board.innerHTML = ''; boardMatrix(state.fen).flat().forEach((sq, i) => { const div = document.createElement('div'); const rank = Math.floor(i / 8); const file = i % 8; div.className = `sq ${(rank + file) % 2 === 0 ? 'light' : 'dark'}`; if (highlights.has(sq.square)) div.classList.add(sq.square === state.uci?.slice(0, 2) ? 'last-from' : 'last-to'); div.innerHTML = `<span>${sq.piece}</span>`; if (file === 0 || rank === 7) { const coord = document.createElement('span'); coord.className = 'coord'; coord.textContent = rank === 7 ? sq.square[0] : sq.square[1]; div.appendChild(coord); } els.board.appendChild(div); }); }
    function renderEvalBar(eValue) { const scale = 20; const white = Number(eValue); const pct = Math.min(Math.abs(white) / scale, 1) * 50; const markerPct = 50 - Math.max(-50, Math.min(50, (white / scale) * 50)); els.whiteEFill.style.height = white > 0 ? `${pct}%` : '0%'; els.blackEFill.style.height = white < 0 ? `${pct}%` : '0%'; els.evalMarker.style.top = `${markerPct}%`; els.evalMarker.classList.toggle('black', white < 0); }
    function stateFromSaved(saved, i, plies) { const nextEntry = plies[i] || null; const previousEntry = i > 0 ? (plies[i - 1] || null) : null; return { ...saved, ply: Number(saved.ply ?? i), moveLabel: saved.move_label || (i === 0 ? 'Start' : `${saved.ply}. ${saved.san}`), staticEvaluation: saved.static_evaluation, nextEntry, previousEntry }; }
    function buildLegacyStates(raw) { const plies = raw.plies || []; const built = []; if (!plies.length) return [{ ply: 0, moveLabel: 'Start', fen: raw.final_fen || '', uci: '', san: '', turn: raw.final_fen ? turnFromFen(raw.final_fen) : '', staticEvaluation: null, nextEntry: null, previousEntry: null }]; const first = plies[0].row; built.push({ ply: 0, moveLabel: 'Start', fen: first.fen_before, uci: '', san: '', turn: turnFromFen(first.fen_before), staticEvaluation: first.E_before, nextEntry: plies[0], previousEntry: null }); plies.forEach((entry, i) => { const row = entry.row; const nextEntry = plies[i + 1] || null; built.push({ ply: Number(row.ply), moveLabel: `${row.ply}. ${row.side} ${row.san}`, fen: row.fen_after, uci: row.uci, san: row.san, turn: turnFromFen(row.fen_after), staticEvaluation: row.E_after, predictionError: row.prediction_error, nextEntry, previousEntry: entry }); }); return built; }
    function buildStates(raw) { const plies = raw.plies || []; if (Array.isArray(raw.viewer_states) && raw.viewer_states.length) return raw.viewer_states.map((saved, i) => stateFromSaved(saved, i, plies)); return buildLegacyStates(raw); }
    async function fetchJson(url) { const response = await fetch(url, { cache: 'no-store' }); if (!response.ok) throw new Error(`${response.status} ${response.statusText}`); return response.json(); }
    async function loadGames(preserve = false) { const current = preserve ? els.gameSelect.value : loadedId; const data = await fetchJson('/api/games'); summaries = data.games || []; els.gameSelect.innerHTML = ''; summaries.forEach((summary) => { const option = document.createElement('option'); option.value = summary.id; option.textContent = `${summary.label} · ${summary.result || 'unknown'} · ${summary.plies || 0} plies`; if (!summary.has_viewer_states) option.textContent += ' · regenerate'; if (summary.error) option.textContent += ' · failed'; els.gameSelect.appendChild(option); }); const target = summaries.some(g => g.id === current) ? current : (summaries[0]?.id || ''); if (target) { els.gameSelect.value = target; if (target !== loadedId || !game) await loadGame(target); } els.status.textContent = `${summaries.length} files`; }
    async function loadGame(id) { loadedId = id; els.status.textContent = 'Loading game...'; const summary = summaries.find(g => g.id === id); if (summary?.error) { els.comparisonPanel.innerHTML = `<div class="notice">${summary.error}</div>`; return; } renderPanelLoading('Loading game data...'); const raw = await fetchJson(`/api/games/${encodeURIComponent(id)}`); game = { ...raw, id, label: summary?.label || id, source_file: summary?.source_file || `${id}.json` }; states = buildStates(game); index = 0; els.source.textContent = game.source_file; els.result.textContent = `Result ${game.result || '*'}`; els.plyCount.textContent = `${Math.max(states.length - 1, 0)} plies`; els.slider.max = Math.max(states.length - 1, 0); els.status.textContent = Array.isArray(game.viewer_states) && game.viewer_states.length ? 'Live' : 'Needs regeneration'; setIndex(0); }
    function renderPanelLoading(message = 'Loading comparison panels...') { els.comparisonPanel.innerHTML = `<div class="loading">${message}</div>`; }
    function esc(value) { return String(value ?? '').replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch])); }
    function styleSummary(style) { return Object.entries(style || {}).filter(([k]) => k !== 'safety').map(([k, v]) => `${k} ${Number(v).toFixed(2)}`).join(' · '); }
    function panelMetric(label, value, digits = 3) { return `<div class="metric"><div class="label">${label}</div><div class="value ${signedClass(value)}">${fmt(value, digits)}</div></div>`; }

    function boardMapFromFen(fen) { const map = {}; const ranks = (fen || '').split(' ')[0].split('/'); for (let r = 0; r < ranks.length; r++) { let file = 0; for (const ch of ranks[r]) { if (/\d/.test(ch)) { file += Number(ch); } else { map['abcdefgh'[file] + String(8 - r)] = ch; file += 1; } } } return map; }
    function applyUciToMap(map, uci) { const next = { ...map }; if (!uci || uci.length < 4) return next; const from = uci.slice(0, 2); const to = uci.slice(2, 4); const promo = uci.slice(4, 5); const piece = next[from]; if (!piece) return next; delete next[from]; if ((piece === 'P' || piece === 'p') && !next[to] && from[0] !== to[0]) { const captured = to[0] + from[1]; delete next[captured]; } if ((piece === 'K' || piece === 'k') && Math.abs('abcdefgh'.indexOf(from[0]) - 'abcdefgh'.indexOf(to[0])) === 2) { if (to[0] === 'g') { const rookFrom = 'h' + from[1]; const rookTo = 'f' + from[1]; next[rookTo] = next[rookFrom]; delete next[rookFrom]; } else if (to[0] === 'c') { const rookFrom = 'a' + from[1]; const rookTo = 'd' + from[1]; next[rookTo] = next[rookFrom]; delete next[rookFrom]; } } next[to] = promo ? (piece === piece.toUpperCase() ? promo.toUpperCase() : promo.toLowerCase()) : piece; return next; }
    function sanLikeFromUci(fen, rootUci, response) { const stored = String(response?.san || ''); if (stored && stored !== response?.uci) return stored; const uci = String(response?.uci || ''); if (uci.length < 4) return uci; const board = applyUciToMap(boardMapFromFen(fen), rootUci || ''); const from = uci.slice(0, 2); const to = uci.slice(2, 4); const promo = uci.slice(4, 5); const piece = board[from]; if (!piece) return uci; const isWhite = piece === piece.toUpperCase(); const type = piece.toUpperCase(); if (type === 'K' && from === 'e1' && to === 'g1') return 'O-O'; if (type === 'K' && from === 'e1' && to === 'c1') return 'O-O-O'; if (type === 'K' && from === 'e8' && to === 'g8') return 'O-O'; if (type === 'K' && from === 'e8' && to === 'c8') return 'O-O-O'; const capture = Boolean(board[to]) || (type === 'P' && from[0] !== to[0]); const names = { K: 'K', Q: 'Q', R: 'R', B: 'B', N: 'N', P: '' }; let label = names[type] || ''; if (type === 'P' && capture) label += from[0]; label += capture ? 'x' : ''; label += to; if (promo) label += `=${promo.toUpperCase()}`; return label; }
    function moveNotation(move, fen, rootUci) { return sanLikeFromUci(fen, rootUci, move); }
    function moveButton(move, side, clickable) { const label = esc(move.san || move.uci || 'move'); if (!clickable) return label; return `<button class="tree-trigger ${side}" data-side="${side}" data-uci="${esc(move.uci)}" title="Show predicted tree">${label}</button>`; }
    function refinementNote() { return 'P(m|B), largest first'; }
    function sortedPanelMoves(panel, side) { const moves = [...(panel?.moves || [])]; moves.sort((a, b) => Number(b.probability || 0) - Number(a.probability || 0)); return moves.map((move, i) => ({ ...move, refinement_rank: i + 1, advantage_rank: i + 1 })); }
    function candidateDelta(move) { return move?.g_tilde; }
    function candidateU(move) { return move?.terminal_u; }
    function refinedQ(move) { return move?.q_tilde; }
    function refinedW(move) { return move?.w_tilde; }
    function refinedA(move) { return move?.a_tilde; }
    function refinedMass(move) { return move?.retained_probability_mass; }
    function refinedCount(move) { return move?.retained_response_count; }
    function branchDelta(response, key) { return response?.[`branch_${key}_tilde`]; }
    function cycleMoveHtml(move) { return `<span class="cycle-move ${esc(move?.color || '')}" title="${esc(move?.color || '')}">${esc(move?.san || '—')}</span>`; }
    function cycleMetric(label, value) { return `<div class="cycle-metric"><div class="label">${label}</div><div class="value ${signedClass(value)}">${fmt(value)}</div></div>`; }
    function cyclePanelHtml(side, cycle) { const title = `${side.toUpperCase()} CYCLE`; if (!cycle) return `<div class="cycle-panel empty"><div class="cycle-head"><div class="cycle-title">${title}</div><div class="style">No completed cycle yet.</div></div></div>`; const moves = cycle.moves || []; const firstLabel = `ΔE_${cycle.first_side || side}`; const secondLabel = `ΔE_${cycle.second_side || (side === 'white' ? 'black' : 'white')}`; return `<div class="cycle-panel"><div class="cycle-head"><div class="cycle-title">${title}</div></div><div class="cycle-content"><div class="cycle-route-block"><div class="cycle-route">${cycleMoveHtml(moves[0])}<span class="cycle-arrow">→</span>${cycleMoveHtml(moves[1])}</div><div class="cycle-ply-metrics">${cycleMetric(firstLabel, cycle.delta_u_first)}${cycleMetric(secondLabel, cycle.delta_u_second)}</div></div><div class="cycle-box"><div class="cycle-box-half">${cycleMetric('ΔU_cycle', cycle.delta_u_cycle)}</div><div class="cycle-box-half"><div class="cycle-metrics">${cycleMetric('ΔQ', cycle.delta_q_cycle)}${cycleMetric('ΔW', cycle.delta_w_cycle)}${cycleMetric('ΔA', cycle.delta_a_cycle)}</div></div></div></div></div>`; }
    function observerRows(panel, side, played) { return sortedPanelMoves(panel, side).map(move => { const rank = Number(move.refinement_rank ?? move.advantage_rank ?? 0); const clickable = Boolean(move.selected_for_refinement); const cls = [move.uci === played ? 'played' : '', rank === 1 ? `${side}-best` : ''].filter(Boolean).join(' '); const g = candidateDelta(move); const q = refinedQ(move); const w = refinedW(move); const a = refinedA(move); return `<tr class="${cls}"><td>${rank || ''}</td><td class="shared-move">${moveButton(move, side, clickable)}</td><td class="${signedClass(move.static_after)}">${fmt(move.static_after)}</td><td>${fmt(move.probability, 4)}</td><td class="${signedClass(g)}">${fmt(g)}</td><td class="${signedClass(q)}">${fmt(q)}</td><td class="${signedClass(w)}">${fmt(w)}</td><td class="${signedClass(a)}">${fmt(a)}</td></tr>`; }).join(''); }
    function observerPanel(side, title, style, panel, played, cycle) { const unavailable = panel?.counterfactual_available === false; const rows = observerRows(panel, side, played); return `<div class="observer-column">${cyclePanelHtml(side, cycle)}<section class="observer-panel"><div class="panel-head"><div class="panel-title">${title}</div><div class="style">${styleSummary(style)}${unavailable ? '<br>Counterfactual not precomputed' : ''}</div></div><div class="metrics">${panelMetric('U½(B)', panel?.expected_value)}${panelMetric('S(B)', panel?.entropy)}${panelMetric('N_eff', panel?.effective_moves)}${panelMetric('K', panel?.expanded_count, 0)}</div><div class="table-note">Moves are sorted by ${refinementNote()} for branch refinement. G̃ is stored only for refined root moves; Q̃/W̃/Ã are the candidate-cycle decomposition of that stored G̃.</div><div class="table-wrap"><table><thead><tr><th>#</th><th>move</th><th>E(Bm)</th><th>P(m|B)</th><th>G̃</th><th>Q̃</th><th>W̃</th><th>Ã</th></tr></thead><tbody>${rows || '<tr><td colspan="8">No legal moves</td></tr>'}</tbody></table></div></section></div>`; }
    function findPanelMove(side, uci) { const state = states[index] || {}; const panel = side === 'white' ? state.white_panel : state.black_panel; return (panel?.moves || []).find(move => move.uci === uci) || null; }
    function renderTree(side, move) { const state = states[index] || {}; const responses = (move.responses || []).filter(response => response.selected_for_refinement).sort((a, b) => Number(b.probability || 0) - Number(a.probability || 0)); const sideLabel = side === 'white' ? 'White' : 'Black'; const responderLabel = side === 'white' ? 'Black' : 'White'; const rootExpected = Number(candidateU(move)); const criterion = `${responderLabel} refinement: higher transition probability is selected`; els.treeTitle.textContent = `${sideLabel} candidate: ${move.san || move.uci}`; els.treeSubtitle.textContent = `G̃ ${fmt(candidateDelta(move))} · Q̃ ${fmt(refinedQ(move))} · W̃ ${fmt(refinedW(move))} · Ã ${fmt(refinedA(move))} · ${criterion}`; const responseRows = responses.map((response, index) => { const responseU = Number(response.terminal_u ?? response.value); const du = branchDelta(response, 'g'); const dq = branchDelta(response, 'q'); const dw = branchDelta(response, 'w'); const da = branchDelta(response, 'a'); const status = Number(response.depth_used || 0) > 0 || response.recursively_deepened ? `recursive continuation ${response.depth_used} plies` : 'terminal shallow landscape'; return { response, responseU, status, du, dq, dw, da, rank: index + 1 }; }); const responseHtml = responseRows.map(({ response, responseU, status, du, dq, dw, da, rank }) => `<div class="tree-node"><div class="tree-line"><span class="tree-move">#${rank} ${esc(moveNotation(response, state.fen, move.uci))}</span><span>P(r|Bm) ${fmt(response.probability, 4)}</span></div><div class="style">${status} · retained path probability ${fmt(response.conditional_probability, 4)} · terminal U ${fmt(responseU)} · branch G̃ ${fmt(du)} · Q̃ ${fmt(dq)} · W̃ ${fmt(dw)} · Ã ${fmt(da)} · residual ${fmt(response.branch_decomposition_error, 2)}</div></div>`).join(''); const rootStatus = move.selected_for_refinement ? 'refined root' : 'not refined'; els.treeBody.innerHTML = `<div class="tree-root"><div class="tree-line"><span class="tree-move">${esc(move.san || move.uci)}</span><span class="${signedClass(candidateDelta(move))}">G̃ ${fmt(candidateDelta(move))}</span></div><div class="style">${rootStatus} · P(m|B) ${fmt(move.probability, 4)} · terminal U ${fmt(rootExpected)} · E(Bm) ${fmt(move.static_after)} · retained response mass ${fmt(refinedMass(move), 4)} · retained responses ${fmt(refinedCount(move), 0)} · Q̃ ${fmt(refinedQ(move))} · W̃ ${fmt(refinedW(move))} · Ã ${fmt(refinedA(move))}</div></div><div class="table-note">${criterion}. Response rows are ordered by probability; Q̃/W̃/Ã branch values are diagnostics for retained responses.</div><div class="tree-list">${responseHtml || '<div class="notice">No selected responses are stored for this move.</div>'}</div>`; els.treeOverlay.classList.add('open'); els.treeOverlay.setAttribute('aria-hidden', 'false'); }
    function closeTree() { els.treeOverlay.classList.remove('open'); els.treeOverlay.setAttribute('aria-hidden', 'true'); }
    function bindTreeButtons() { els.comparisonPanel.querySelectorAll('.tree-trigger').forEach(button => button.addEventListener('click', () => { const move = findPanelMove(button.dataset.side, button.dataset.uci); if (move) renderTree(button.dataset.side, move); })); }
    function renderComparison(state) { if (!state.white_panel && !state.black_panel) { els.comparisonPanel.innerHTML = '<div class="notice">This match JSON does not contain saved search results. Regenerate the match with the current simulation code.</div>'; return; } const played = state.nextEntry?.row?.uci || ''; const depth = state.white_panel?.depth ?? state.black_panel?.depth ?? game.config?.depth ?? 3; const searchMode = state.white_panel?.search_mode ?? state.black_panel?.search_mode ?? game.config?.search_mode ?? 'accurate'; const adaptiveC = state.white_panel?.adaptive_c ?? state.black_panel?.adaptive_c ?? game.config?.adaptive_c; els.comparisonPanel.innerHTML = `<div class="panel-head"><div class="panel-title">White / Black Style Measures</div><div class="style"><strong>Depth:</strong> ${depth} plies · ${searchMode} · c ${fmt(adaptiveC)} · probability refinement</div></div><div class="observer-grid">${observerPanel('white', 'White Observer', game.white_style, state.white_panel, played, state.white_realized_cycle)}${observerPanel('black', 'Black Observer', game.black_style, state.black_panel, played, state.black_realized_cycle)}</div>`; bindTreeButtons(); }
    function renderMoveList() { els.moveList.innerHTML = states.map((state, i) => `<span class="move-chip ${i === index ? 'active' : ''}" data-index="${i}">${i === 0 ? 'Start' : `${state.ply}. ${state.san}`}</span>`).join(' '); els.moveList.querySelectorAll('.move-chip').forEach(chip => chip.addEventListener('click', () => setIndex(Number(chip.dataset.index)))); }
    function setIndex(nextIndex) { if (!states.length) return; index = Math.max(0, Math.min(states.length - 1, nextIndex)); const state = states[index]; els.slider.value = index; els.fen.textContent = state.fen; els.positionLabel.textContent = state.moveLabel; els.turn.textContent = state.turn || 'unknown'; els.staticV.textContent = fmt(state.staticEvaluation); els.staticV.className = `value ${signedClass(state.staticEvaluation)}`; renderEvalBar(state.staticEvaluation); renderBoard(state); renderCapturedMaterial(state); renderMoveList(); closeTree(); renderComparison(state); }
    els.gameSelect.addEventListener('change', event => loadGame(event.target.value)); els.refresh.addEventListener('click', () => loadGames(true)); els.slider.addEventListener('input', event => setIndex(Number(event.target.value))); els.first.addEventListener('click', () => setIndex(0)); els.prev.addEventListener('click', () => setIndex(index - 1)); els.next.addEventListener('click', () => setIndex(index + 1)); els.last.addEventListener('click', () => setIndex(states.length - 1)); els.treeClose.addEventListener('click', closeTree); els.treeOverlay.addEventListener('click', event => { if (event.target === els.treeOverlay) closeTree(); }); window.addEventListener('keydown', event => { if (event.key === 'Escape') closeTree(); if (event.key === 'ArrowLeft') setIndex(index - 1); if (event.key === 'ArrowRight') setIndex(index + 1); if (event.key === 'Home') setIndex(0); if (event.key === 'End') setIndex(states.length - 1); });
    loadGames().catch(error => { els.status.textContent = 'Error'; els.comparisonPanel.innerHTML = `<div class="notice">${error.message}</div>`; }); setInterval(() => loadGames(true).catch(() => {}), 10000);
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class LiveViewerConfig:
    results_dir: Path = Path("data/results")
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = False


def _read_match_summary(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "id": path.stem,
            "label": path.stem.replace("_", " "),
            "source_file": str(path),
            "result": data.get("result", "*"),
            "terminal_reason": data.get("terminal_reason", ""),
            "final_fen": data.get("final_fen", ""),
            "plies": len(data.get("plies", [])),
            "has_viewer_states": bool(data.get("viewer_states")),
            "modified": path.stat().st_mtime,
            "size": path.stat().st_size,
        }
    except Exception as exc:
        return {
            "id": path.stem,
            "label": f"{path.stem} (failed to load)",
            "source_file": str(path),
            "result": "*",
            "plies": 0,
            "has_viewer_states": False,
            "modified": path.stat().st_mtime if path.exists() else 0,
            "size": path.stat().st_size if path.exists() else 0,
            "error": str(exc),
        }


def game_summaries(results_dir: Path) -> list[dict[str, Any]]:
    return [_read_match_summary(path) for path in sorted(results_dir.glob("*.json"))]


def _game_path(results_dir: Path, game_id: str) -> Path | None:
    candidate = results_dir / f"{game_id}.json"
    if candidate.exists() and candidate.is_file() and candidate.parent.resolve() == results_dir.resolve():
        return candidate
    for path in results_dir.glob("*.json"):
        if path.stem == game_id:
            return path
    return None


class _LiveViewerHandler(BaseHTTPRequestHandler):
    server: "LiveViewerServer"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(data).encode("utf-8"), "application/json; charset=utf-8", status)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = posixpath.normpath(parsed.path)
        if route in ("/", "/index.html"):
            self._send_bytes(LIVE_VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/games":
            self._send_json(
                {
                    "results_dir": str(self.server.config.results_dir),
                    "games": game_summaries(self.server.config.results_dir),
                }
            )
            return
        if route.startswith("/api/games/"):
            game_id = unquote(route.removeprefix("/api/games/"))
            path = _game_path(self.server.config.results_dir, game_id)
            if path is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"No game named {game_id!r}")
                return
            self._send_bytes(path.read_bytes(), "application/json; charset=utf-8")
            return
        if route == "/favicon.ico":
            self._send_bytes(b"", "image/x-icon")
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")


class LiveViewerServer(ThreadingHTTPServer):
    def __init__(self, config: LiveViewerConfig):
        self.config = config
        super().__init__((config.host, config.port), _LiveViewerHandler)


def serve_match_viewer(config: LiveViewerConfig) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    server = LiveViewerServer(config)
    url = f"http://{config.host}:{server.server_address[1]}/"
    print(f"Live viewer: {url}")
    print(f"Results dir: {config.results_dir}")
    print("Press Ctrl+C to stop.")
    if config.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a live viewer for saved match JSON files.")
    parser.add_argument("--results-dir", default="data/results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the viewer in a browser.")
    args = parser.parse_args(argv)
    serve_match_viewer(
        LiveViewerConfig(
            results_dir=Path(args.results_dir),
            host=args.host,
            port=args.port,
            open_browser=args.open,
        )
    )


if __name__ == "__main__":
    main()
