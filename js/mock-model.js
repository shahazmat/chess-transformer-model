// Placeholder model. Replace with the real thing.
//
// Implements the exact contract engine.js expects (see engine.js header) and
// returns a full-vocabulary Float32Array per token step, so the skeleton
// exercises the same code path a real next-token model will use.
//
// Behaviour, so games feel plausible while testing the harness:
//  - scores each legal move with a tiny 1.5-ply greedy heuristic
//    (material won, material left hanging, checks, promotion, centre, noise)
//    and softmaxes that into a distribution over whole moves
//  - answers each predict() call with the exact next-token marginals of that
//    distribution given ctx.moveTokens — e.g. after ['x'] it spreads mass
//    over the cores of capturing moves only, the way a calibrated LM would
//  - on the first token of a move it also puts mass on the three nerf
//    tokens, scaled by the opponent's rating when one was entered (weaker
//    opponent -> the mock blunders more often); the rating *site* is
//    deliberately unused — a real model might embed it, since 1500 lichess
//    and 1500 chess.com are different strengths
//  - when re-queried with ctx.quality set, reshapes the move distribution:
//    'inaccuracy' plays sloppily, 'mistake' prefers bad moves, 'blunder'
//    actively hunts for the worst move on the board

import { Chess } from '../lib/chess.js';
import { TOKEN_INDEX, VOCAB_SIZE, NERF_TOKENS, sanToTokens } from './vocab.js';

const VAL = { p: 1, n: 3, b: 3.1, r: 5, q: 9, k: 0 };

function scoreMove(game, m) {
  let s = 0;
  if (m.captured) s += VAL[m.captured];
  if (m.promotion) s += VAL[m.promotion] - VAL.p;
  game.move(m.san);
  if (game.isCheckmate()) {
    s += 100;
  } else {
    if (game.inCheck()) s += 0.4;
    let hanging = 0;
    for (const reply of game.moves({ verbose: true })) {
      if (reply.captured) hanging = Math.max(hanging, VAL[reply.captured]);
    }
    s -= 0.9 * hanging;
  }
  game.undo();
  const file = m.to.charCodeAt(0) - 97;
  const rank = m.to.charCodeAt(1) - 49;
  s += 0.04 * ((3.5 - Math.abs(file - 3.5)) + (3.5 - Math.abs(rank - 3.5)));
  s += (Math.random() - 0.5) * 0.6;
  return s;
}

function shapeByQuality(scores, quality) {
  switch (quality) {
    case 'inaccuracy': return { logits: scores, temp: 2.6 };
    case 'mistake': return { logits: scores.map((s) => -0.6 * s), temp: 1.6 };
    case 'blunder': return { logits: scores.map((s) => -s), temp: 0.8 };
    default: return { logits: scores, temp: 1.0 };
  }
}

function softmax(logits, temp) {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp((l - max) / temp));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// Probability of opening a move with <inaccuracy>, <mistake>, <blunder>.
function nerfMasses(rating) {
  const r = Math.min(2800, Math.max(400, rating ?? 1500));
  const w = (2800 - r) / 2400; // 0 = strong opponent, 1 = weak
  return [0.02 + 0.10 * w, 0.012 + 0.08 * w, 0.006 + 0.09 * w];
}

export function createMockModel() {
  return {
    name: 'mock (greedy heuristic + noise)',

    async predict(ctx) {
      const out = new Float32Array(VOCAB_SIZE);
      const game = new Chess(ctx.fen);
      const moves = game.moves({ verbose: true });
      if (moves.length === 0) return out;

      const scores = moves.map((m) => scoreMove(game, m));
      const { logits, temp } = shapeByQuality(scores, ctx.quality);
      const probs = softmax(logits, temp);
      const seqs = moves.map((m) => sanToTokens(m.san));
      const prefix = ctx.moveTokens ?? [];

      let nerfMass = 0;
      if (prefix.length === 0 && !ctx.quality) {
        const masses = nerfMasses(ctx.opponent?.rating);
        NERF_TOKENS.forEach((t, i) => { out[TOKEN_INDEX.get(t)] = masses[i]; });
        nerfMass = masses.reduce((a, b) => a + b, 0);
      }

      // Next-token marginals of the move distribution, given the prefix.
      let denom = 0;
      const marginal = new Map();
      moves.forEach((_, i) => {
        const tokens = seqs[i];
        if (tokens.length <= prefix.length) return;
        if (!prefix.every((t, k) => tokens[k] === t)) return;
        denom += probs[i];
        const next = tokens[prefix.length];
        marginal.set(next, (marginal.get(next) ?? 0) + probs[i]);
      });
      if (denom > 0) {
        for (const [token, p] of marginal) {
          out[TOKEN_INDEX.get(token)] = (p / denom) * (1 - nerfMass);
        }
      }
      return out;
    },
  };
}
