import { Chess } from '../lib/chess.js';
import { createBoard } from './board.js';
import { pieceMarkup } from './pieces.js';
import { createMockModel } from './mock-model.js';
import { pickComputerMove, ENGINE_CONFIG } from './engine.js';
import { VOCAB_SIZE, GLYPH_BY_QUALITY, sanToTokens } from './vocab.js';

const $ = (id) => document.getElementById(id);

let game = new Chess();
let model = createMockModel();
let phase = 'setup'; // 'setup' | 'playing' | 'over'
let humanColor = 'w';
let opponent = { rating: null, site: null };
let busy = false;
let selected = null;
let targets = new Map();      // square -> 'move' | 'capture'
let targetMoves = new Map();  // square -> verbose moves landing there
let annotations = new Map();  // ply index -> quality, for computer moves
let lastResult = null;        // last pickComputerMove() result, for inspector
let pendingPromotion = null;
let tokenStream = [];         // the game as a flat token stream, nerf tokens included

const board = createBoard($('board'), onSquareClick);

// ---------------------------------------------------------------- rendering

function checkSquare() {
  if (!game.inCheck()) return null;
  for (const row of game.board()) {
    for (const cell of row) {
      if (cell && cell.type === 'k' && cell.color === game.turn()) return cell.square;
    }
  }
  return null;
}

function lastMove() {
  const history = game.history({ verbose: true });
  const m = history[history.length - 1];
  return m ? { from: m.from, to: m.to } : null;
}

function render() {
  board.render({
    board: game.board(),
    orientation: humanColor,
    selected,
    targets,
    lastMove: lastMove(),
    checkSquare: checkSquare(),
  });
  renderStatus();
  renderMoves();
}

function renderStatus() {
  const el = $('status');
  if (phase === 'over') {
    el.textContent = gameOverText();
    el.className = 'status over';
    return;
  }
  if (busy) {
    el.textContent = 'chess-gpt is thinking…';
    el.className = 'status thinking';
    return;
  }
  const yours = game.turn() === humanColor;
  const side = game.turn() === 'w' ? 'White' : 'Black';
  el.textContent = yours ? `Your move — ${side}${game.inCheck() ? ' (check)' : ''}` : `chess-gpt to move`;
  el.className = 'status';
}

function gameOverText() {
  if (game.isCheckmate()) {
    const winner = game.turn() === humanColor ? 'chess-gpt wins' : 'you win';
    return `Checkmate — ${winner}`;
  }
  if (game.isStalemate()) return 'Draw — stalemate';
  if (game.isThreefoldRepetition()) return 'Draw — threefold repetition';
  if (game.isInsufficientMaterial()) return 'Draw — insufficient material';
  if (game.isDrawByFiftyMoves()) return 'Draw — fifty-move rule';
  return 'Draw';
}

function renderMoves() {
  const history = game.history();
  const rows = [];
  for (let i = 0; i < history.length; i += 2) {
    const num = i / 2 + 1;
    rows.push(`<span class="num">${num}.</span>${moveCell(history, i)}${moveCell(history, i + 1)}`);
  }
  const list = $('moves');
  list.innerHTML = rows.join('');
  list.scrollTop = list.scrollHeight;
}

function moveCell(history, ply) {
  const san = history[ply];
  if (san === undefined) return '<span class="mv"></span>';
  const quality = annotations.get(ply);
  const glyph = quality ? `<em class="${quality}" title="${quality} token">${GLYPH_BY_QUALITY[quality]}</em>` : '';
  return `<span class="mv">${san}${glyph}</span>`;
}

const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function renderInspector() {
  const el = $('inspector-body');
  if (!lastResult) {
    el.innerHTML = '<p class="muted">Plays after the computer’s first move.</p>';
    return;
  }
  const r = lastResult;
  const pct = (p) => `${(p * 100).toFixed(1)}%`;
  const chain = r.steps.map((s) => `[${escapeHtml(s.token)}]`).join(' ');
  const lines = [];
  lines.push(`<p>Decoded <strong>${r.san}</strong> as <code>${chain}</code>${r.quality ? ` <em class="${r.quality}">— ${r.quality}</em>` : ''}</p>`);
  lines.push(`<p class="muted">${r.legalCount} legal moves · ${VOCAB_SIZE.toLocaleString()} tokens in vocab · nerf mass ${pct(r.nerfMass)}</p>`);
  r.steps.forEach((step, n) => {
    const max = Math.max(...step.top.map((t) => t.p), 1e-9);
    lines.push(`<p class="step-label">step ${n + 1} — sampled <strong>${escapeHtml(step.token)}</strong> at ${pct(step.p)}</p>`);
    lines.push('<div class="bars">');
    for (const t of step.top) {
      lines.push(
        `<div class="bar-row${t.nerf ? ' special' : ''}${t.token === step.token ? ' sampled' : ''}">` +
        `<span class="tok">${escapeHtml(t.token)}</span>` +
        `<span class="bar"><span style="width:${Math.max(1.5, (t.p / max) * 100)}%"></span></span>` +
        `<span class="pct">${pct(t.p)}</span></div>`,
      );
    }
    lines.push('</div>');
  });
  el.innerHTML = lines.join('');
}

// ------------------------------------------------------------- interaction

function onSquareClick(square) {
  if (phase !== 'playing' || busy || game.turn() !== humanColor || pendingPromotion) return;

  if (selected && targetMoves.has(square)) {
    const moves = targetMoves.get(square);
    if (moves[0].promotion) {
      showPromotionPicker(moves[0]);
    } else {
      playHumanMove({ from: moves[0].from, to: moves[0].to });
    }
    return;
  }

  const piece = game.get(square);
  if (piece && piece.color === humanColor && square !== selected) {
    selected = square;
    targets = new Map();
    targetMoves = new Map();
    for (const m of game.moves({ square, verbose: true })) {
      targets.set(m.to, m.captured ? 'capture' : 'move');
      if (!targetMoves.has(m.to)) targetMoves.set(m.to, []);
      targetMoves.get(m.to).push(m);
    }
  } else {
    clearSelection();
  }
  render();
}

function clearSelection() {
  selected = null;
  targets = new Map();
  targetMoves = new Map();
}

function playHumanMove(move) {
  const played = game.move(move);
  tokenStream.push(...sanToTokens(played.san));
  clearSelection();
  render();
  if (game.isGameOver()) return endGame();
  computerTurn();
}

async function computerTurn() {
  busy = true;
  render();
  const [lo, hi] = ENGINE_CONFIG.thinkDelayMs;
  await new Promise((res) => setTimeout(res, lo + Math.random() * (hi - lo)));

  const ctx = {
    fen: game.fen(),
    moves: game.history(),
    historyTokens: [...tokenStream],
    turn: game.turn(),
    opponent,
    legalMoves: game.moves(),
  };
  try {
    const result = await pickComputerMove(model, ctx);
    game.move(result.san);
    if (result.nerf) tokenStream.push(result.nerf.token);
    tokenStream.push(...result.tokens);
    if (result.quality) annotations.set(game.history().length - 1, result.quality);
    lastResult = result;
  } catch (err) {
    console.error('model failed, playing a uniform-random legal move', err);
    const legal = game.moves();
    const played = game.move(legal[Math.floor(Math.random() * legal.length)]);
    tokenStream.push(...sanToTokens(played.san));
  }
  busy = false;
  renderInspector();
  render();
  if (game.isGameOver()) endGame();
}

function endGame() {
  phase = 'over';
  game.setHeader('Result', resultTag());
  render();
}

function resultTag() {
  if (!game.isCheckmate()) return '1/2-1/2';
  return game.turn() === 'b' ? '1-0' : '0-1';
}

// --------------------------------------------------------------- promotion

function showPromotionPicker(move) {
  pendingPromotion = move;
  const picker = $('promo');
  const color = humanColor === 'w' ? 'white' : 'black';
  picker.querySelectorAll('button[data-piece]').forEach((btn) => {
    btn.innerHTML = `<svg viewBox="0 0 60 60"><g class="piece ${color}">${pieceMarkup(btn.dataset.piece)}</g></svg>`;
  });
  picker.hidden = false;
}

$('promo').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-piece]');
  const move = pendingPromotion;
  pendingPromotion = null;
  $('promo').hidden = true;
  if (btn && move) {
    playHumanMove({ from: move.from, to: move.to, promotion: btn.dataset.piece });
  } else {
    clearSelection();
    render();
  }
});

// ------------------------------------------------------------------ setup

let pickedColor = 'w';

$('pick-white').addEventListener('click', () => setPickedColor('w'));
$('pick-black').addEventListener('click', () => setPickedColor('b'));

function setPickedColor(c) {
  pickedColor = c;
  $('pick-white').classList.toggle('active', c === 'w');
  $('pick-black').classList.toggle('active', c === 'b');
}

$('site').addEventListener('change', () => {
  $('site-other').hidden = $('site').value !== 'other';
});

$('start').addEventListener('click', () => {
  const rating = parseInt($('rating').value, 10);
  const siteSel = $('site').value;
  const site = siteSel === 'other' ? $('site-other').value.trim() || null : siteSel || null;
  opponent = {
    rating: Number.isFinite(rating) ? Math.min(4000, Math.max(100, rating)) : null,
    site,
  };
  humanColor = pickedColor;
  startGame();
});

function startGame() {
  game = new Chess();
  annotations = new Map();
  lastResult = null;
  tokenStream = [];
  clearSelection();
  phase = 'playing';
  busy = false;

  const you = `Human${opponent.rating ? ` (${opponent.rating}${opponent.site ? ` ${opponent.site}` : ''})` : ''}`;
  const bot = `chess-gpt [${model.name}]`;
  game.setHeader('Event', 'chess-gpt skeleton game');
  game.setHeader('White', humanColor === 'w' ? you : bot);
  game.setHeader('Black', humanColor === 'b' ? you : bot);
  if (opponent.rating) game.setHeader(humanColor === 'w' ? 'WhiteElo' : 'BlackElo', String(opponent.rating));
  if (opponent.site) game.setHeader('RatingSite', opponent.site);

  $('setup').hidden = true;
  $('play').hidden = false;
  renderInspector();
  render();
  if (humanColor === 'b') computerTurn();
}

$('new-game').addEventListener('click', () => {
  phase = 'setup';
  game = new Chess();
  annotations = new Map();
  lastResult = null;
  tokenStream = [];
  clearSelection();
  busy = false;
  pendingPromotion = null;
  $('promo').hidden = true;
  $('setup').hidden = false;
  $('play').hidden = true;
  render();
});

$('copy-pgn').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(game.pgn());
    flashButton($('copy-pgn'), 'Copied');
  } catch {
    flashButton($('copy-pgn'), 'Clipboard blocked');
  }
});

function flashButton(btn, text) {
  const original = btn.textContent;
  btn.textContent = text;
  setTimeout(() => { btn.textContent = original; }, 1200);
}

// ------------------------------------------------- debug / integration seam

// window.chessGpt.setModel(yourModel) swaps the AI at runtime — handy while
// loading real weights asynchronously. loadFen() jumps to a position.
window.chessGpt = {
  get game() { return game; },
  vocabSize: VOCAB_SIZE,
  config: ENGINE_CONFIG,
  setModel(m) {
    model = m;
    console.log(`chess-gpt: model set to "${m.name ?? 'unnamed'}"`);
  },
  loadFen(fen) {
    game.load(fen);
    annotations = new Map();
    tokenStream = [];
    clearSelection();
    phase = 'playing';
    $('setup').hidden = true;
    $('play').hidden = false;
    render();
    if (game.isGameOver()) return endGame();
    if (game.turn() !== humanColor) computerTurn();
  },
};

render();
