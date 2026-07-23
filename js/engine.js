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
//     excludeDraw:   true once the human has declined a draw offer this game —
//                    <draw> is then masked for every remaining move.
//     vocab:         the token list (index i of a vector output = probability
//                    of vocab[i])
//   }
//
// Decoding one computer move:
//   1. tokenize every legal move; the sequences form a prefix tree
//   2. at each step, mask the model's distribution down to the tokens that
//      can continue some legal sequence (plus, on the very first step, the
//      three nerf tokens, the engine's OWN colour's resign token, and
//      <draw>) — all illegal tokens get probability zero. The opponent's
//      resign token and the <white/black-flag> tokens are never sampleable;
//      they exist as training signal and inspector gauges.
//   3. renormalize, sample
//   4. nerf token drawn -> remember the quality, restart decoding with
//      quality set and nerf tokens masked out
//   5. own resign token drawn -> return { resign: true } (the engine
//      resigns); <draw> drawn -> return { drawOffer: true } (app.js runs the
//      dialog, and on decline re-decodes with ctx.excludeDraw set)
//   6. a core token completes the sequence -> play the matching SAN

import {
  TOKENS, TOKEN_INDEX, NERF_TOKENS, QUALITY_BY_TOKEN, VOCAB_SIZE,
  DRAW_TOKEN, END_TOKENS, resignTokenFor, isNerfToken, isEndToken, sanToTokens,
} from './vocab.js';

export const ENGINE_CONFIG = {
  temperature: 1,          // p_i ^ (1/T) over each masked step; <1 sharpens.
                           // This is the OPENING temperature — the stages below
                           // step it down as the game goes on.
  tempMidFrom: 10,         // fullmove the middlegame temperature kicks in (0 = never)
  tempMid: 0.05,
  tempEndFrom: 30,         // fullmove the endgame temperature kicks in (0 = never)
  tempEnd: 0.4,
  allowNerfTokens: ['<inaccuracy>'],
                           // false = masked: the model plays only its clean branch
                           // (full strength). true = human-like mode — it may announce
                           // a nerf and play a deliberately degraded move. An array
                           // of nerf tokens allows only those to be sampled.
  forceQuality: null,      // 'inaccuracy' | 'mistake' | 'blunder' | null — force
                           // EVERY computer move down that nerf branch: the token
                           // is taken as drawn (p=1) on the first step, the move
                           // decodes conditioned on it, and the UI annotates it.
                           // Overrides allowNerfTokens (no second nerf is drawn).
  minP: 0.05,               // decode-step floor: tokens under this masked
                           // probability are dropped before sampling (0 = off).
                           // If nothing clears the bar, the best token survives.
  minPExempt: [],
                           // tokens the minP floor never drops — they stay
                           // sampleable at their (renormalized) model weight.
  minPFromMove: 10,        // apply the minP floor only from this fullmove number
                           // onwards (1 = the whole game); earlier moves sample
                           // the unfloored distribution.
  topP: 0.7,               // nucleus sampling per decode step: keep the smallest
                           // set of highest-probability tokens whose cumulative
                           // mass reaches this, renormalize, sample (1 = off).
                           // minPExempt tokens always survive at their weight.
  topK: 0,                 // clean play samples only among the model's K best
                           // legal moves (by full-sequence probability); a legal
                           // mate is always kept. 0 = off (all legal moves).
  forceMate: true,         // if a mate is legal, always play it. '#' leads every
                           // mating sequence, so we force '#' on the first step
                           // when it is available (the model still picks WHICH
                           // mate). The model tags most mates as check, so a mere
                           // probability boost isn't enough — this is a hard rule.
  allowResign: true,       // the model may sample its own colour's resign token
                           // as its move: it resigns and you win. (forceMate
                           // outranks it — with a mate on the board the engine
                           // always mates.)
  allowDrawOffer: true,    // the model may sample <draw>: a draw-offer dialog
                           // opens; declining masks <draw> for the rest of the
                           // game and the engine decodes a normal move instead.
  thinkDelayMs: [350, 900],
};

// Adapt the accepted output shapes to a single token -> probability fn.
let warnedOutputLength = false; // once per session — an old 5268-wide checkpoint
                                // is fine (missing tail tokens read as p=0)
function lookupFor(output) {
  if (output instanceof Float32Array || output instanceof Float64Array || Array.isArray(output)) {
    if (output.length !== VOCAB_SIZE && !warnedOutputLength) {
      warnedOutputLength = true;
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
    // Degenerate model output (no mass on any allowed token): uniform fallback
    // over the real moves only — never nerf/end tokens by accident.
    console.warn('model put zero mass on every allowed token — falling back to uniform');
    probs = allowed.map((t) => (isNerfToken(t) || isEndToken(t) ? 0 : 1));
    sum = probs.reduce((a, b) => a + b, 0);
  }
  return { tokens: allowed, probs: probs.map((v) => v / sum) };
}

// Sampling temperature for a given fullmove: the opening value, stepping down
// at the configured middlegame / endgame move numbers (0 disables a stage).
function temperatureForMove(moveNumber) {
  const c = ENGINE_CONFIG;
  if (c.tempEndFrom > 0 && moveNumber >= c.tempEndFrom) return c.tempEnd;
  if (c.tempMidFrom > 0 && moveNumber >= c.tempMidFrom) return c.tempMid;
  return c.temperature;
}

// Nucleus (top-p) trim for decode steps: walk tokens from most to least
// probable, keep them until their cumulative mass reaches ENGINE_CONFIG.topP,
// zero the rest, renormalize. minPExempt tokens are never trimmed — a
// sampleable nerf token's mass is tiny and would otherwise never make the
// nucleus — and don't count toward the cumulative mass.
function applyTopP(dist) {
  const topP = ENGINE_CONFIG.topP;
  if (!topP || topP >= 1) return dist;
  const exempt = ENGINE_CONFIG.minPExempt ?? [];
  const order = dist.probs.map((_, i) => i).sort((a, b) => dist.probs[b] - dist.probs[a]);
  const keep = new Set();
  let cum = 0;
  for (const i of order) {
    if (dist.probs[i] <= 0) break;
    if (exempt.includes(dist.tokens[i])) continue;
    keep.add(i);
    cum += dist.probs[i];
    if (cum >= topP) break;
  }
  dist.tokens.forEach((t, i) => { if (exempt.includes(t) && dist.probs[i] > 0) keep.add(i); });
  const probs = dist.probs.map((p, i) => (keep.has(i) ? p : 0));
  const sum = probs.reduce((a, b) => a + b, 0);
  return sum > 0 ? { tokens: dist.tokens, probs: probs.map((v) => v / sum) } : dist;
}

// The minP floor for decode steps: zero tokens below ENGINE_CONFIG.minP
// (measured on the masked, renormalized distribution), then renormalize the
// survivors. minPExempt tokens are never dropped. If no non-exempt token
// clears the bar, the single best one survives so a move always exists.
function applyMinP(dist, moveNumber) {
  const minP = ENGINE_CONFIG.minP;
  if (!minP || moveNumber < (ENGINE_CONFIG.minPFromMove ?? 1)) return dist;
  const exempt = ENGINE_CONFIG.minPExempt ?? [];
  const keep = dist.probs.map((p, i) => p >= minP || exempt.includes(dist.tokens[i]));
  if (!keep.some((k, i) => k && !exempt.includes(dist.tokens[i]))) {
    let best = -1;
    dist.probs.forEach((p, i) => {
      if (!exempt.includes(dist.tokens[i]) && (best === -1 || p > dist.probs[best])) best = i;
    });
    if (best !== -1) keep[best] = true;
  }
  const probs = dist.probs.map((p, i) => (keep[i] ? p : 0));
  const sum = probs.reduce((a, b) => a + b, 0);
  return sum > 0 ? { tokens: dist.tokens, probs: probs.map((v) => v / sum) } : dist;
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

// The model's probability of one whole move token sequence, decoded with
// legal-move masking at each step (temperature 1 — a raw ranking used only to
// pick the top-K legal moves). `predict` is memoized by the caller, so the many
// single-token moves all share the one empty-prefix model call.
async function scoreSeq(predict, tokens, allSeqs) {
  let p = 1;
  const prefix = [];
  for (let depth = 0; depth < tokens.length; depth++) {
    const cont = allSeqs.filter((s) => prefix.every((t, i) => s.tokens[i] === t));
    const allowed = [...new Set(cont.map((s) => s.tokens[prefix.length]).filter((t) => t !== undefined))];
    const dist = maskedDistribution(await predict(prefix, null), allowed, 1);
    const idx = dist.tokens.indexOf(tokens[depth]);
    if (idx === -1) return 0;
    p *= dist.probs[idx];
    prefix.push(tokens[depth]);
  }
  return p;
}

export async function pickComputerMove(model, ctx) {
  const moveNumber = Math.floor(ctx.moves.length / 2) + 1;
  const T = temperatureForMove(moveNumber);
  const allSeqs = ctx.legalMoves.map((san) => ({ san, tokens: sanToTokens(san) }));

  // One model call per (quality, prefix), memoized and shared between the
  // top-K scoring pass and the decode loop below.
  const cache = new Map();
  const predict = async (prefix, quality) => {
    const key = `${quality ?? ''}|${prefix.join(',')}`;
    if (!cache.has(key)) {
      cache.set(key, await model.predict({ ...ctx, quality, moveTokens: [...prefix], vocab: TOKENS }));
    }
    return cache.get(key);
  };

  // ---- gauges over the FULL legal set, unconditioned first step ----
  const out0 = await predict([], null);
  const firstAll = [...new Set(allSeqs.map((s) => s.tokens[0]))];
  const gaugeDist = maskedDistribution(out0, [...firstAll, ...NERF_TOKENS, ...END_TOKENS], T);
  const nerfMass = gaugeDist.tokens.reduce((a, t, i) => a + (isNerfToken(t) ? gaugeDist.probs[i] : 0), 0);
  const gi = gaugeDist.tokens.indexOf('#');
  const rawSharp = lookupFor(out0)('#');
  // '#' leads every mating sequence, so a legal mate exists iff '#' is a legal
  // first token. p = the model's (unboosted) belief it announces mate now, or
  // its raw '#' mass when no mate is on the board (a miscalibration tell).
  const mate = {
    available: firstAll.includes('#'),
    p: gi !== -1 ? gaugeDist.probs[gi] : (Number.isFinite(rawSharp) && rawSharp > 0 ? rawSharp : 0),
  };
  // End-token gauges for the inspector: the model's masked belief that the
  // side to move should resign, that the OPPONENT is about to resign (a
  // win-confidence tell — that token is never sampleable), and that the game
  // should be drawn. Old 5268-wide checkpoints have no mass on any of them,
  // so all read 0 and the tokens are never sampled.
  const gaugeP = (tok) => {
    const i = gaugeDist.tokens.indexOf(tok);
    return i !== -1 ? gaugeDist.probs[i] : 0;
  };
  const ownResignToken = resignTokenFor(ctx.turn);
  const oppResignToken = resignTokenFor(ctx.turn === 'w' ? 'b' : 'w');
  const pResign = gaugeP(ownResignToken);
  const pOppResign = gaugeP(oppResignToken);
  const pDraw = gaugeP(DRAW_TOKEN);

  // forceQuality: pretend that nerf token was drawn on the first step — same
  // branch as a sampled nerf (conditioned decode over ALL legal moves, no
  // top-K), but deterministic, so every computer move carries the annotation.
  const forcedTok = ENGINE_CONFIG.forceQuality
    ? NERF_TOKENS.find((t) => QUALITY_BY_TOKEN[t] === ENGINE_CONFIG.forceQuality)
    : undefined;

  // ---- restrict clean play to the model's top-K legal moves (+ any mate) ----
  let topMoves = null;
  let baseCandidates = allSeqs;
  if (!forcedTok && ENGINE_CONFIG.topK > 0 && allSeqs.length > ENGINE_CONFIG.topK) {
    const scored = [];
    for (const s of allSeqs) scored.push({ s, p: await scoreSeq(predict, s.tokens, allSeqs) });
    scored.sort((a, b) => b.p - a.p);
    const keep = new Set(scored.slice(0, ENGINE_CONFIG.topK).map((x) => x.s));
    for (const x of scored) if (x.s.san.includes('#')) keep.add(x.s); // never drop a mate
    baseCandidates = allSeqs.filter((s) => keep.has(s));
    topMoves = scored.filter((x) => keep.has(x.s)).map((x) => ({ san: x.s.san, p: x.p }));
  }

  // ---- decode token-by-token over the (restricted) candidate moves ----
  let quality = null;
  let nerf = null;
  const steps = [];
  let prefix = [];
  let candidates = baseCandidates;

  if (forcedTok) {
    quality = ENGINE_CONFIG.forceQuality;
    nerf = { token: forcedTok, p: 1 };
    steps.push({ token: forcedTok, p: 1, top: [{ token: forcedTok, p: 1, sp: 1, special: true }] });
  }

  for (let guard = 0; guard < 8; guard++) {
    const first = prefix.length === 0;
    const legalNext = [...new Set(candidates.map((s) => s.tokens[prefix.length]).filter((t) => t !== undefined))];
    // Special tokens are only ever candidates on the very first, unconditioned
    // step of a move: nerf tokens (config-gated) and the engine's OWN end
    // actions — its own colour's resign token and <draw>. The opponent's
    // resign token and the flag tokens are never sampleable (they are not the
    // engine's actions), and a move committed to a quality is never a
    // resignation or a draw offer.
    const allowed = first && quality === null
      ? [
          ...legalNext,
          ...(ENGINE_CONFIG.allowNerfTokens === true ? NERF_TOKENS
            : Array.isArray(ENGINE_CONFIG.allowNerfTokens) ? ENGINE_CONFIG.allowNerfTokens : []),
          ...(ENGINE_CONFIG.allowResign ? [ownResignToken] : []),
          ...(ENGINE_CONFIG.allowDrawOffer && !ctx.excludeDraw ? [DRAW_TOKEN] : []),
        ]
      : legalNext;

    const out = await predict(prefix, quality);
    // The model's honest weights over the allowed tokens (no temperature, no
    // top-p, no floor) — recorded per step so the UI can show raw vs sampled.
    const rawDist = maskedDistribution(out, allowed, 1);
    const dist = applyMinP(applyTopP(maskedDistribution(out, allowed, T)), moveNumber);

    // Force mate: '#' is a legal first token exactly when a mate is playable.
    // If so, take it deterministically (the model tags most mates as check, so
    // its '#' mass is tiny — a boost isn't enough). Later steps then pick which
    // mating move. p is left at the model's honest belief for the inspector.
    const mateIdx = first && (quality === null || forcedTok) && ENGINE_CONFIG.forceMate ? dist.tokens.indexOf('#') : -1;
    const i = mateIdx !== -1 ? mateIdx : sampleIndex(dist.probs);
    const token = dist.tokens[i];
    // top-of-step view, ranked by RAW weight: p = the model's honest
    // probability, sp = what it became after temperature + top-p + the floor
    // (0 = trimmed out of the sampling pool).
    const top = dist.tokens
      .map((t, k) => ({ token: t, p: rawDist.probs[k], sp: dist.probs[k], special: isNerfToken(t) || isEndToken(t) }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 5);
    steps.push({ token, p: dist.probs[i], rawP: rawDist.probs[i], top, forced: mateIdx !== -1 });

    if (isEndToken(token)) {
      // Game-end action sampled instead of a move — hand control to app.js:
      // the engine's own resign token ends the game outright, <draw> opens
      // the offer dialog (and a decline re-enters pickComputerMove with
      // ctx.excludeDraw set). Only those two can ever be sampled here.
      return {
        ...(token === ownResignToken ? { resign: true } : { drawOffer: true }),
        steps,
        nerfMass,
        mate,
        pResign,
        pOppResign,
        pDraw,
        topMoves,
        legalCount: ctx.legalMoves.length,
      };
    }

    if (isNerfToken(token)) {
      // Nerfed: re-decode from scratch, conditioned on the drawn quality. Drop
      // the top-K restriction — a deliberate blunder wants the model's bad moves.
      quality = QUALITY_BY_TOKEN[token];
      nerf = { token, p: dist.probs[i] };
      prefix = [];
      candidates = allSeqs;
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
        mate,             // { available, p } — the P('#') gauge for the UI
        pResign,          // masked P(own resign token) on this turn (inspector)
        pOppResign,       // masked P(opponent's resign token) — win-confidence tell
        pDraw,            // masked P(<draw>) on this turn (inspector)
        topMoves,         // [{ san, p }] the restricted pool, or null when off
        sampledSan: complete.san,
        legalCount: ctx.legalMoves.length,
      };
    }
  }
  throw new Error('decode did not terminate — token sequences should be at most 4 tokens + 1 nerf');
}
