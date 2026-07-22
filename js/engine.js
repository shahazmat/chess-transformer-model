// The socket the AI plugs into, and the decoding loop around it.
//
// A model is any object with:
//
//   async predict(ctx) -> Float32Array | number[] | { [token]: probability }
//
// giving the probability of the NEXT token. It is called once per token
// while the engine decodes a move sequence  [#|+]? [x]? [=Q..]? [core]
// (see vocab.js for the scheme).
//
//   ctx = {
//     fen:           current position (FEN string)
//     moves:         SAN history of the game, e.g. ['e4', 'c5', 'Nf3']
//     historyTokens: the same history as a BARE flat token stream — it never
//                    contains nerf tokens, not even ones the computer itself
//                    emitted earlier (training strips all past nerfs from
//                    context; see chess-tokeniser/nerf_batch.py). A real LM
//                    adapter builds its context as
//                      <bos> <eloW> <eloB> ...historyTokens
//                             [+ the nerf token matching `quality`, if set]
//                             ...moveTokens
//     moveTokens:    tokens emitted so far for the move being decoded,
//                    e.g. [] then ['x'] then ['x', '=Q'] ...
//     turn:          'w' | 'b'  — side the model is playing
//     opponent:      { rating: number | null, site: string | null }
//     legalMoves:    SAN strings of every legal move (convenience; a real LM
//                    may ignore this — illegal output is masked anyway)
//     quality:       null normally. If the move's first sample drew a nerf
//                    token, every later predict() call for this move has
//                    quality set to 'inaccuracy' | 'mistake' | 'blunder' —
//                    the model is now conditioned on "this move is a <quality>".
//     vocab:         the token list (index i of a vector output = probability
//                    of vocab[i])
//   }
//
// Decoding one computer move:
//   1. tokenize every legal move; the sequences form a prefix tree
//   2. at each step, mask the model's distribution down to the tokens that
//      can continue some legal sequence (plus, on the very first step, the
//      three nerf tokens) — all illegal tokens get probability zero
//   3. renormalize, sample
//   4. nerf token drawn -> remember the quality, restart decoding with
//      quality set and nerf tokens masked out
//   5. a core token completes the sequence -> play the matching SAN

import { TOKENS, TOKEN_INDEX, NERF_TOKENS, QUALITY_BY_TOKEN, VOCAB_SIZE, isNerfToken, sanToTokens } from './vocab.js';

export const ENGINE_CONFIG = {
  temperature: 1.0,        // p_i ^ (1/T) over each masked step; 1 = untouched
  allowNerfTokens: false,  // masked: the model plays only its clean branch (full
                           // strength). true = human-like mode — it may announce
                           // a nerf and play a deliberately degraded move.
  thinkDelayMs: [350, 900],
};

// Adapt the accepted output shapes to a single token -> probability fn.
function lookupFor(output) {
  if (output instanceof Float32Array || output instanceof Float64Array || Array.isArray(output)) {
    if (output.length !== VOCAB_SIZE) {
      console.warn(`model output has length ${output.length}, vocabulary has ${VOCAB_SIZE} tokens`);
    }
    return (token) => {
      const i = TOKEN_INDEX.get(token);
      return i === undefined || i >= output.length ? 0 : output[i];
    };
  }
  if (output && typeof output === 'object') return (token) => output[token] ?? 0;
  throw new Error('model.predict() must return a Float32Array, number[], or {token: probability} object');
}

// Mask the model output down to `allowed`, then renormalize what survives.
function maskedDistribution(output, allowed, temperature) {
  const prob = lookupFor(output);
  let probs = allowed.map((t) => {
    if (!TOKEN_INDEX.has(t)) console.warn(`token "${t}" is missing from the vocabulary`);
    const v = prob(t);
    return Number.isFinite(v) && v > 0 ? v : 0;
  });
  if (temperature !== 1) probs = probs.map((v) => (v > 0 ? v ** (1 / temperature) : 0));
  let sum = probs.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    // Degenerate model output (no mass on any allowed token): uniform fallback.
    console.warn('model put zero mass on every allowed token — falling back to uniform');
    probs = allowed.map((t) => (isNerfToken(t) ? 0 : 1));
    sum = probs.reduce((a, b) => a + b, 0);
  }
  return { tokens: allowed, probs: probs.map((v) => v / sum) };
}

function sampleIndex(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  for (let i = probs.length - 1; i >= 0; i--) if (probs[i] > 0) return i;
  return 0;
}

// Top-of-distribution view for the UI inspector.
function topOf({ tokens, probs }, limit = 5) {
  return tokens
    .map((t, i) => ({ token: t, p: probs[i], nerf: isNerfToken(t) }))
    .sort((a, b) => b.p - a.p)
    .slice(0, limit);
}

export async function pickComputerMove(model, ctx) {
  const T = ENGINE_CONFIG.temperature;
  const seqs = ctx.legalMoves.map((san) => ({ san, tokens: sanToTokens(san) }));

  let quality = null;
  let nerf = null;       // the drawn nerf token and its probability, if any
  let nerfMass = 0;      // total nerf mass on the first, unconditioned step
  const steps = [];      // one entry per model call, for the inspector

  let prefix = [];
  let candidates = seqs;

  for (let guard = 0; guard < 8; guard++) {
    const first = prefix.length === 0;
    const legalNext = [...new Set(candidates.map((s) => s.tokens[prefix.length]))];
    const allowed = first && quality === null && ENGINE_CONFIG.allowNerfTokens
      ? [...legalNext, ...NERF_TOKENS]
      : legalNext;

    const out = await model.predict({ ...ctx, quality, moveTokens: [...prefix], vocab: TOKENS });
    const dist = maskedDistribution(out, allowed, T);
    if (first && quality === null) {
      // nerfMass is always gauged over legal moves + nerf tokens, so with
      // allowNerfTokens=false it reports the HYPOTHETICAL chance the model
      // would have opened this move with a nerf — sampling stays masked.
      const gauge = ENGINE_CONFIG.allowNerfTokens
        ? dist
        : maskedDistribution(out, [...legalNext, ...NERF_TOKENS], T);
      nerfMass = gauge.tokens.reduce((a, t, i) => a + (isNerfToken(t) ? gauge.probs[i] : 0), 0);
    }

    const i = sampleIndex(dist.probs);
    const token = dist.tokens[i];
    steps.push({ token, p: dist.probs[i], top: topOf(dist) });

    if (isNerfToken(token)) {
      // Nerfed: re-decode from scratch, conditioned on the drawn quality.
      quality = QUALITY_BY_TOKEN[token];
      nerf = { token, p: dist.probs[i] };
      prefix = [];
      candidates = seqs;
      continue;
    }

    prefix = [...prefix, token];
    candidates = candidates.filter((s) => s.tokens[prefix.length - 1] === token);
    const complete = candidates.find((s) => s.tokens.length === prefix.length);
    if (complete) {
      return {
        san: complete.san,
        tokens: prefix,
        steps,
        quality,          // null | 'inaccuracy' | 'mistake' | 'blunder'
        nerf,
        nerfMass,
        legalCount: ctx.legalMoves.length,
      };
    }
  }
  throw new Error('decode did not terminate — token sequences should be at most 4 tokens + 1 nerf');
}
