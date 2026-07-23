// Generates the static token dictionary at js/vocab-data.js.
//
//   node tools/build-vocab.mjs
//
// Run this ONLY when the tokenization scheme itself changes. The emitted file
// is the frozen source of truth for token <-> index; regenerating it with a
// different scheme reorders indices and invalidates any model trained against
// the old dictionary.
//
// Core generation is a deterministic, slightly loose superset of real SAN:
// every core is syntactically valid and geometrically possible on an empty
// board. Cores that can never occur in a position are simply masked to zero
// by the engine, so overapproximation is harmless.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const FILES = [...'abcdefgh'];
const RANKS = [...'12345678'];

// Must match the constants in js/vocab.js (it asserts this at load).
const MODIFIER_TOKENS = ['+', '#', 'x', '=Q', '=R', '=B', '=N'];
const NERF_TOKENS = ['<inaccuracy>', '<mistake>', '<blunder>'];

// vocab v2 — structural framing tokens for the training pipeline, appended
// AFTER the nerf tokens so every pre-existing id (0..5251) is preserved.
// <bos>/<eos> wrap each game (game separators); the <elo-*> buckets condition
// the model on both players' strength. Elo bucketing is 200-point steps with
// open-ended tails; must match eloBucketToken() in js/vocab.js and
// elo_bucket_token() in chess-tokeniser/vocab.py.
const ELO_TOKENS = ['<elo-u800>'];
for (let e = 800; e <= 2800; e += 200) ELO_TOKENS.push(`<elo-${e}>`);
ELO_TOKENS.push('<elo-3000p>');
// "strength unspecified" sentinel. During packing, each game's real Elo bucket
// is swapped for this for a fraction of games (per player, independently), so the
// model also learns UNCONDITIONED play — letting inference omit/neutralise Elo.
const ELO_ANY_TOKEN = '<elo-any>';
const STRUCTURAL_TOKENS = ['<bos>', '<eos>', ...ELO_TOKENS, ELO_ANY_TOKEN];

// vocab v3 — game-end tokens, appended AFTER the structural block so every
// pre-existing id (0..5267) is preserved (the invariant is pure append, never
// reorder; this tail was redefined once on 2026-07-22 while nothing was
// trained past id 5267). The packer emits one just before <eos>, naming HOW
// the game ended and WHO acted:
//   <white-resign>/<black-resign> — that colour resigned (either parity:
//     players often resign right after their own move, so colour cannot be
//     inferred from whose turn it is);
//   <white-flag>/<black-flag> — that colour lost on time (the flagging side
//     is always the side to move — your clock only runs on your turn — but
//     the explicit colour keeps the scheme uniform); never sampled in play;
//   <draw> — the game ends drawn (no colour: the outcome is symmetric).
// During play, sampling the engine's OWN resign token means it resigns, and
// <draw> means it offers you a draw.
const END_TOKENS = ['<white-resign>', '<black-resign>', '<white-flag>', '<black-flag>', '<draw>'];

const KNIGHT_STEPS = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
const KING_STEPS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];

// All squares from which `piece` could move to (df, dr) on an empty board.
function sourceSquares(piece, df, dr) {
  const out = [];
  const add = (f, r) => { if (f >= 0 && f < 8 && r >= 0 && r < 8) out.push([f, r]); };
  if (piece === 'N') for (const [a, b] of KNIGHT_STEPS) add(df + a, dr + b);
  if (piece === 'K') for (const [a, b] of KING_STEPS) add(df + a, dr + b);
  if (piece === 'R' || piece === 'Q') {
    for (let f = 0; f < 8; f++) if (f !== df) out.push([f, dr]);
    for (let r = 0; r < 8; r++) if (r !== dr) out.push([df, r]);
  }
  if (piece === 'B' || piece === 'Q') {
    for (let d = -7; d <= 7; d++) {
      if (d === 0) continue;
      add(df + d, dr + d);
      add(df + d, dr - d);
    }
  }
  return out;
}

// Disambiguation prefixes that standard SAN could ever require for this
// piece/destination, derived from the source-square set S:
//  - file letter: some source is on that file and sources span several files
//  - rank digit:  some source on that rank shares its file with another source
//                 (SAN falls back to rank only when the file is ambiguous)
//  - full square: the source has both a file-mate and a rank-mate in S
//                 (needs 3+ identical pieces, i.e. after promotions)
function disambiguations(S) {
  const forms = [''];
  const files = new Set(S.map(([f]) => f));
  if (files.size >= 2) for (const f of [...files].sort((a, b) => a - b)) forms.push(FILES[f]);
  for (const [f, r] of S) {
    if (S.some(([f2, r2]) => f2 === f && r2 !== r) && !forms.includes(RANKS[r])) forms.push(RANKS[r]);
  }
  for (const [f, r] of S) {
    const fileMate = S.some(([f2, r2]) => f2 === f && r2 !== r);
    const rankMate = S.some(([f2, r2]) => r2 === r && f2 !== f);
    if (fileMate && rankMate) forms.push(FILES[f] + RANKS[r]);
  }
  return forms;
}

function buildCoreTokens() {
  const cores = [];

  // Pawn pushes: plain file+rank. Rank 1/8 arrivals exist as cores too — they
  // only ever appear behind a promotion modifier, e.g. ['=Q', 'e8'].
  for (const f of FILES) for (const r of RANKS) cores.push(`${f}${r}`);

  // Pawn captures: source file + destination square (adjacent files only).
  for (let to = 0; to < 8; to++) {
    for (const from of [to - 1, to + 1]) {
      if (from < 0 || from > 7) continue;
      for (const r of RANKS) cores.push(`${FILES[from]}${FILES[to]}${r}`);
    }
  }

  // Piece moves, with every disambiguation SAN could require.
  for (const piece of ['N', 'B', 'R', 'Q', 'K']) {
    for (let dr = 7; dr >= 0; dr--) {
      for (let df = 0; df < 8; df++) {
        const dest = FILES[df] + RANKS[dr];
        const S = sourceSquares(piece, df, dr);
        const forms = piece === 'K' ? [''] : disambiguations(S);
        for (const d of forms) cores.push(`${piece}${d}${dest}`);
      }
    }
  }

  cores.push('O-O', 'O-O-O');
  return cores;
}

const cores = buildCoreTokens();
if (new Set(cores).size !== cores.length) throw new Error('duplicate core tokens generated');
const tokens = [...cores, ...MODIFIER_TOKENS, ...NERF_TOKENS, ...STRUCTURAL_TOKENS, ...END_TOKENS];
if (new Set(tokens).size !== tokens.length) throw new Error('duplicate tokens generated');

const lines = [];
for (let i = 0; i < tokens.length; i += 12) {
  lines.push(tokens.slice(i, i + 12).map((t) => `'${t}'`).join(','));
}

const out = `// Static token dictionary — generated by tools/build-vocab.mjs. DO NOT EDIT.
// ${tokens.length} tokens = ${cores.length} core moves + ${MODIFIER_TOKENS.length} modifiers + ${NERF_TOKENS.length} nerf tokens
// + ${STRUCTURAL_TOKENS.length} structural tokens (<bos>, <eos>, ${ELO_TOKENS.length} <elo-NNN> buckets, <elo-any>)
// + ${END_TOKENS.length} end tokens (<white/black-resign>, <white/black-flag>, <draw>), in that order.
// Index in this array IS the model's token id: regenerating with a changed
// scheme reorders ids and invalidates trained checkpoints. Structural tokens
// (vocab v2) and end tokens (vocab v3) were appended without disturbing any
// pre-existing id.
export const CORE_COUNT = ${cores.length};
export const TOKENS = Object.freeze([
${lines.join(',\n')}
]);
`;

const dest = fileURLToPath(new URL('../js/vocab-data.js', import.meta.url));
writeFileSync(dest, out);
console.log(`wrote ${dest}`);
console.log(`${tokens.length} tokens = ${cores.length} cores + ${MODIFIER_TOKENS.length} modifiers + ${NERF_TOKENS.length} nerf + ${STRUCTURAL_TOKENS.length} structural + ${END_TOKENS.length} end`);
