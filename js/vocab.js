// Token vocabulary for the move-prediction model.
//
// The dictionary itself is STATIC: js/vocab-data.js (generated once by
// tools/build-vocab.mjs, checked into git) holds the frozen, ordered token
// list — index in that array is the model's token id. This module wraps it
// with lookups and the SAN <-> token-sequence converters.
//
// A move is a SEQUENCE of tokens: optional modifier tokens followed by one
// core move token. The core token is what terminates a move — when the model
// emits it, the move is over. Canonical order:
//
//   [#|+]  [x]  [=Q|=R|=B|=N]  [core]
//
//   fxe8=Q+  ->  ['+', 'x', '=Q', 'fe8']     (pawn capture-promotion, check)
//   Nxf3     ->  ['x', 'Nf3']
//   Raxe1#   ->  ['#', 'x', 'Rae1']
//   e4       ->  ['e4']
//
// The core keeps SAN's disambiguation, so tokenization is lossless: 'fe8' is
// the pawn-capture core (source file + destination), 'Rae1'/'R1e2'/'Qh4e1'
// keep their file/rank/square disambiguators, castling cores are 'O-O' and
// 'O-O-O'.
//
// The three "nerf" tokens sit at the very end of the vocabulary. They are
// only ever valid as the FIRST token of a move sequence: sampling one means
// "the coming move is an inaccuracy/mistake/blunder", and the engine then
// re-queries the model conditioned on that quality.
//
// When your real tokenizer is ready, swap vocab-data.js for its dictionary
// (and adjust sanToTokens/tokensToSan here if the scheme differs).

import { TOKENS, CORE_COUNT } from './vocab-data.js';

export { TOKENS };

export const MODIFIER_TOKENS = ['+', '#', 'x', '=Q', '=R', '=B', '=N'];
export const NERF_TOKENS = ['<inaccuracy>', '<mistake>', '<blunder>'];

// nerf token -> quality label passed back into model.predict({quality})
export const QUALITY_BY_TOKEN = {
  '<inaccuracy>': 'inaccuracy',
  '<mistake>': 'mistake',
  '<blunder>': 'blunder',
};

// quality label -> conventional annotation glyph, used in the move list
export const GLYPH_BY_QUALITY = {
  inaccuracy: '?!',
  mistake: '?',
  blunder: '??',
};

export const CORE_TOKENS = Object.freeze(TOKENS.slice(0, CORE_COUNT));
export const TOKEN_INDEX = new Map(TOKENS.map((t, i) => [t, i]));
export const VOCAB_SIZE = TOKENS.length;

// The dictionary must end with exactly the modifier + nerf tokens this module
// (and the engine) reason about. Fail loudly if the data file has drifted.
{
  const tail = TOKENS.slice(CORE_COUNT);
  const expected = [...MODIFIER_TOKENS, ...NERF_TOKENS];
  if (tail.length !== expected.length || tail.some((t, i) => t !== expected[i])) {
    throw new Error('vocab-data.js is out of sync with vocab.js — regenerate it: node tools/build-vocab.mjs');
  }
}

export const isNerfToken = (t) => Object.hasOwn(QUALITY_BY_TOKEN, t);

const SAN_RE = /^([KQRBN]?[a-h]?[1-8]?)(x?)([a-h][1-8])(=[QRBN])?([+#])?$/;
const CASTLE_RE = /^(O-O(?:-O)?)([+#])?$/;

// SAN string -> canonical token sequence. Throws on strings that are not SAN.
export function sanToTokens(san) {
  const tokens = [];
  const castle = san.match(CASTLE_RE);
  if (castle) {
    if (castle[2]) tokens.push(castle[2]);
    tokens.push(castle[1]);
    return tokens;
  }
  const m = san.match(SAN_RE);
  if (!m) throw new Error(`not a SAN move: "${san}"`);
  const [, head, capture, dest, promo, suffix] = m;
  if (suffix) tokens.push(suffix);
  if (capture) tokens.push('x');
  if (promo) tokens.push(promo);
  tokens.push(head + dest);
  return tokens;
}

// Token sequence -> SAN string (inverse of sanToTokens).
export function tokensToSan(tokens) {
  let suffix = '', capture = '', promo = '';
  let core = null;
  for (const t of tokens) {
    if (t === '+' || t === '#') suffix = t;
    else if (t === 'x') capture = 'x';
    else if (t.startsWith('=')) promo = t;
    else core = t;
  }
  if (!core) throw new Error(`token sequence has no core move: [${tokens.join(', ')}]`);
  if (core.startsWith('O-O')) return core + suffix;
  const head = core.slice(0, -2);
  const dest = core.slice(-2);
  return head + capture + dest + promo + suffix;
}
