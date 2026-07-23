import { Chess } from '../lib/chess.js';
import { createBoard } from './board.js';
import { pieceMarkup } from './pieces.js';
import { createMockModel } from './mock-model.js';
import { detectRemoteModel, lastDetectError, SERVER_BASE } from './remote-model.js';
import { pickComputerMove, ENGINE_CONFIG } from './engine.js';
import { VOCAB_SIZE, GLYPH_BY_QUALITY, sanToTokens } from './vocab.js';
import './settings.js'; // the sidebar engine-settings panel (self-wiring)

const $ = (id) => document.getElementById(id);

let game = new Chess();
let model = createMockModel();
// Swap in the real checkpoint once the backend answers. Keep retrying for a
// while: a Hugging Face Space waking from sleep takes a minute or two, and a
// single 1.5s probe at page load silently stranded phones on the mock model.
// The footer badge always says which model is live.
async function connectModel() {
  const badge = $('backend-status');
  for (let attempt = 1; attempt <= 24; attempt++) {
    const remote = await detectRemoteModel();
    if (remote) {
      model = remote;
      console.log(`GPCT: model server detected — using "${remote.name}"`);
      if (badge) badge.textContent = `model: ${remote.name}`;
      return;
    }
    console.log(`GPCT: model server unreachable (attempt ${attempt}, ${lastDetectError}) — using the mock model`);
    if (badge) badge.textContent = `model: mock — connecting to ${SERVER_BASE} (try ${attempt}: ${lastDetectError})`;
    await new Promise((r) => setTimeout(r, 5000));
  }
  const badge2 = $('backend-status');
  if (badge2) badge2.textContent = `model: mock — ${SERVER_BASE} unreachable (${lastDetectError})`;
}
connectModel();
let phase = 'setup'; // 'setup' | 'playing' | 'over'
let humanColor = 'w';
let opponent = { rating: null, site: null };
let busy = false;
let selected = null;
let targets = new Map();      // square -> 'move' | 'capture'
let targetMoves = new Map();  // square -> verbose moves landing there
let annotations = new Map();  // ply index -> quality, for computer moves
let moveRecords = [];         // one record per ply: computer plies keep the full
                              // pickComputerMove() result; human plies { san, human }
let viewPly = null;           // null = live; else the number of plies shown (review mode)
let pendingPromotion = null;
let pendingDrawOffer = null;  // resolve fn of the open draw-offer dialog
let drawDeclined = false;     // human declined once -> <draw> masked all game
let gameOutcome = null;       // { result, reason } for resignation / agreed draw
let tokenStream = [];         // the game as a BARE flat token stream — never
                              // contains nerf tokens (the bare-history training
                              // contract; see chess-tokeniser/nerf_batch.py)

const board = createBoard($('board'), onSquareClick);

// ---------------------------------------------------------------- rendering

function checkSquare(g = game) {
  if (!g.inCheck()) return null;
  for (const row of g.board()) {
    for (const cell of row) {
      if (cell && cell.type === 'k' && cell.color === g.turn()) return cell.square;
    }
  }
  return null;
}

function lastMove(g = game) {
  const history = g.history({ verbose: true });
  const m = history[history.length - 1];
  return m ? { from: m.from, to: m.to } : null;
}

// The position on display: the live game, or a replayed prefix when reviewing.
function viewedGame() {
  if (viewPly === null) return game;
  const g = new Chess();
  const hist = game.history();
  for (let i = 0; i < viewPly; i++) g.move(hist[i]);
  return g;
}

function render() {
  const vg = viewedGame();
  board.render({
    board: vg.board(),
    orientation: humanColor,
    selected,
    targets,
    lastMove: lastMove(vg),
    checkSquare: checkSquare(vg),
  });
  renderStatus();
  renderMoves();
  renderNav();
}

function renderStatus() {
  const el = $('status');
  if (viewPly !== null) {
    el.textContent = `Reviewing move ${viewPly} of ${game.history().length} — ▶ returns to live`;
    el.className = 'status thinking';
    return;
  }
  if (phase === 'over') {
    el.textContent = gameOverText();
    el.className = 'status over';
    return;
  }
  if (pendingDrawOffer) {
    el.textContent = 'GPCT offers a draw';
    el.className = 'status thinking';
    return;
  }
  if (busy) {
    el.textContent = 'GPCT is thinking…';
    el.className = 'status thinking';
    return;
  }
  const yours = game.turn() === humanColor;
  const side = game.turn() === 'w' ? 'White' : 'Black';
  el.textContent = yours ? `Your move — ${side}${game.inCheck() ? ' (check)' : ''}` : `GPCT to move`;
  el.className = 'status';
}

function gameOverText() {
  if (gameOutcome?.reason === 'resign') return 'GPCT resigns — you win';
  if (gameOutcome?.reason === 'draw-agreed') return 'Draw agreed';
  if (game.isCheckmate()) {
    const winner = game.turn() === humanColor ? 'GPCT wins' : 'you win';
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
  const viewing = viewPly !== null && viewPly - 1 === ply ? ' viewing' : '';
  return `<span class="mv${viewing}" data-ply="${ply}" title="review this move">${san}${glyph}</span>`;
}

const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function renderInspector() {
  const el = $('inspector-body');
  const shown = viewPly === null ? game.history().length : viewPly;
  const r = shown > 0 ? moveRecords[shown - 1] : null;
  if (!r) {
    el.innerHTML = shown === 0 && game.history().length > 0
      ? '<p class="muted">Start position — no move yet.</p>'
      : '<p class="muted">Plays after the computer’s first move.</p>';
    return;
  }
  if (r.human) {
    el.innerHTML = `<p><strong>${escapeHtml(r.san)}</strong> — your move; the model is not queried for it.</p>`;
    return;
  }
  if (r.fallback) {
    el.innerHTML = `<p><strong>${escapeHtml(r.san)}</strong> — the model call failed; a uniform-random legal move was played.</p>`;
    return;
  }
  const pct = (p) => `${(p * 100).toFixed(1)}%`;
  const chain = r.steps.map((s) => `[${escapeHtml(s.token)}]`).join(' ');
  const lines = [];
  lines.push(`<p>Decoded <strong>${r.san}</strong> as <code>${chain}</code>${r.quality ? ` <em class="${r.quality}">— ${r.quality}</em>` : ''}</p>`);
  const endGauges = r.pDraw !== undefined || r.pResign !== undefined
    ? ` · P(draw) ${pct(r.pDraw ?? 0)} · P(resign) ${pct(r.pResign ?? 0)}`
      + (r.pOppResign ? ` · P(you resign) ${pct(r.pOppResign)}` : '')
    : '';
  lines.push(`<p class="muted">${r.legalCount} legal moves · ${VOCAB_SIZE.toLocaleString()} tokens in vocab · nerf mass ${pct(r.nerfMass)}${ENGINE_CONFIG.allowNerfTokens ? '' : ' (masked)'} · P(#) ${pct(r.mate?.p ?? 0)}${r.mate?.available ? ' — <strong>mate available</strong>' : ''}${endGauges}</p>`);
  if (r.topMoves && r.topMoves.length) {
    const cands = r.topMoves
      .map((m) => `<span class="cand${m.san === r.sampledSan ? ' on' : ''}">${escapeHtml(m.san)} <span class="cp">${pct(m.p)}</span></span>`)
      .join('');
    lines.push(`<p class="muted">chose among the model’s top ${r.topMoves.length} legal moves:</p><div class="cands">${cands}</div>`);
  }
  r.steps.forEach((step, n) => {
    const max = Math.max(...step.top.map((t) => t.p), 1e-9);
    const rawNote = step.rawP !== undefined && Math.abs(step.rawP - step.p) > 0.0005
      ? ` <span class="muted">(raw ${pct(step.rawP)})</span>` : '';
    lines.push(step.forced
      ? `<p class="step-label">step ${n + 1} — <strong>#</strong> forced (mate available; model gave it ${pct(step.rawP ?? step.p)})</p>`
      : `<p class="step-label">step ${n + 1} — sampled <strong>${escapeHtml(step.token)}</strong> at ${pct(step.p)}${rawNote}</p>`);
    lines.push('<div class="bars">');
    for (const t of step.top) {
      // bars rank and size by the model's RAW weight; the arrow shows what
      // temperature + top-p + the floor turned it into ("cut" = trimmed out)
      const shaped = t.sp === undefined || Math.abs(t.sp - t.p) <= 0.0005 ? ''
        : t.sp > 0 ? ` <span class="sp">→ ${pct(t.sp)}</span>` : ' <span class="sp">→ cut</span>';
      lines.push(
        `<div class="bar-row${t.special ? ' special' : ''}${t.token === step.token ? ' sampled' : ''}${t.sp === 0 ? ' cut' : ''}">` +
        `<span class="tok">${escapeHtml(t.token)}</span>` +
        `<span class="bar"><span style="width:${Math.max(1.5, (t.p / max) * 100)}%"></span></span>` +
        `<span class="pct">${pct(t.p)}${shaped}</span></div>`,
      );
    }
    lines.push('</div>');
  });
  el.innerHTML = lines.join('');
}

// ------------------------------------------------------------- interaction

function onSquareClick(square) {
  if (viewPly !== null) return; // reviewing: board is read-only until back at live
  if (phase !== 'playing' || busy || game.turn() !== humanColor || pendingPromotion || pendingDrawOffer) return;

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

// ------------------------------------------------------------- move review

function renderNav() {
  const total = game.history().length;
  const cur = viewPly === null ? total : viewPly;
  $('ply-pos').textContent = viewPly === null ? (total ? 'live' : '—') : `${cur} / ${total}`;
  $('ply-back').disabled = cur === 0;
  $('ply-fwd').disabled = viewPly === null;
}

// ply = number of plies to show; at (or past) the live edge we return to live.
function setView(ply) {
  const total = game.history().length;
  viewPly = ply === null || ply >= total ? null : Math.max(0, ply);
  clearSelection();
  render();
  renderInspector();
}

function stepView(delta) {
  const total = game.history().length;
  const cur = viewPly === null ? total : viewPly;
  setView(cur + delta);
}

$('ply-back').addEventListener('click', () => stepView(-1));
$('ply-fwd').addEventListener('click', () => stepView(1));

$('moves').addEventListener('click', (e) => {
  const cell = e.target.closest('.mv[data-ply]');
  if (cell) setView(parseInt(cell.dataset.ply, 10) + 1);
});

document.addEventListener('keydown', (e) => {
  if (phase === 'setup' || pendingPromotion || pendingDrawOffer) return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepView(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepView(1); }
});

function playHumanMove(move) {
  const played = game.move(move);
  tokenStream.push(...sanToTokens(played.san));
  moveRecords.push({ san: played.san, human: true });
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
    excludeDraw: drawDeclined,
  };
  try {
    let result = await pickComputerMove(model, ctx);

    if (result.drawOffer && !(await offerDraw())) {
      // Declined: <draw> stays masked for the rest of the game; decode a
      // normal move from the same position (predict() calls hit the server
      // again, but it is a single position).
      drawDeclined = true;
      result = await pickComputerMove(model, { ...ctx, excludeDraw: true });
    }
    if (result.drawOffer) { // offer stood and was accepted
      busy = false;
      endGame({ result: '1/2-1/2', reason: 'draw-agreed' });
      return;
    }
    if (result.resign) {
      busy = false;
      endGame({ result: humanColor === 'w' ? '1-0' : '0-1', reason: 'resign' });
      return;
    }

    game.move(result.san);
    // Deliberately NOT pushing result.nerf into the stream: history stays bare
    // (the model is trained with past nerfs stripped). The quality lives on in
    // `annotations` for the move list. End tokens never reach the stream either
    // — the game is over, or the offer was declined and the token rejected.
    tokenStream.push(...result.tokens);
    if (result.quality) annotations.set(game.history().length - 1, result.quality);
    moveRecords.push(result);
  } catch (err) {
    console.error('model failed, playing a uniform-random legal move', err);
    const legal = game.moves();
    const played = game.move(legal[Math.floor(Math.random() * legal.length)]);
    tokenStream.push(...sanToTokens(played.san));
    moveRecords.push({ san: played.san, fallback: true });
  }
  busy = false;
  renderInspector();
  render();
  if (game.isGameOver()) endGame();
}

function endGame(outcome = null) {
  gameOutcome = outcome; // null = ended by the rules; else resign / agreed draw
  phase = 'over';
  game.setHeader('Result', resultTag());
  render();
}

function resultTag() {
  if (gameOutcome) return gameOutcome.result;
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

// --------------------------------------------------------------- draw offer

// The engine sampled <draw>: show the dialog and resolve to the human's
// answer. Unlike the promotion picker, clicking the scrim does NOT dismiss —
// an offer deserves an explicit Accept or Decline.
function offerDraw() {
  return new Promise((resolve) => {
    pendingDrawOffer = resolve;
    $('draw-offer').hidden = false;
    render(); // status line: "GPCT offers a draw"
  });
}

$('draw-offer').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-answer]');
  if (!btn || !pendingDrawOffer) return;
  const resolve = pendingDrawOffer;
  pendingDrawOffer = null;
  $('draw-offer').hidden = true;
  resolve(btn.dataset.answer === 'accept');
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
  moveRecords = [];
  viewPly = null;
  tokenStream = [];
  drawDeclined = false;
  gameOutcome = null;
  pendingDrawOffer = null;
  $('draw-offer').hidden = true;
  clearSelection();
  phase = 'playing';
  busy = false;

  const you = `Human${opponent.rating ? ` (${opponent.rating}${opponent.site ? ` ${opponent.site}` : ''})` : ''}`;
  const bot = `GPCT [${model.name}]`;
  game.setHeader('Event', 'GPCT skeleton game');
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
  moveRecords = [];
  viewPly = null;
  tokenStream = [];
  drawDeclined = false;
  gameOutcome = null;
  pendingDrawOffer = null;
  clearSelection();
  busy = false;
  pendingPromotion = null;
  $('promo').hidden = true;
  $('draw-offer').hidden = true;
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
    console.log(`GPCT: model set to "${m.name ?? 'unnamed'}"`);
  },
  loadFen(fen) {
    game.load(fen);
    annotations = new Map();
    moveRecords = [];
    viewPly = null;
    tokenStream = [];
    drawDeclined = false;
    gameOutcome = null;
    pendingDrawOffer = null;
    $('draw-offer').hidden = true;
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
