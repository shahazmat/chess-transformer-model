// Sanity check for the sequential tokenization scheme. Plays random games
// and asserts, for every legal move chess.js can produce:
//   1. sanToTokens(san) only emits tokens that exist in the vocabulary
//   2. tokensToSan(sanToTokens(san)) round-trips back to the same SAN
//   node tools/vocab-check.mjs [games]
import { Chess } from '../lib/chess.js';
import {
  TOKEN_INDEX, VOCAB_SIZE, CORE_TOKENS, MODIFIER_TOKENS, NERF_TOKENS,
  sanToTokens, tokensToSan,
} from '../js/vocab.js';

const GAMES = Number(process.argv[2] ?? 500);
const missing = new Map();
const broken = new Map();
let plies = 0;
let checked = 0;

for (let g = 0; g < GAMES; g++) {
  const game = new Chess();
  for (let i = 0; i < 300 && !game.isGameOver(); i++) {
    const legal = game.moves();
    for (const san of legal) {
      checked++;
      let tokens;
      try {
        tokens = sanToTokens(san);
      } catch (err) {
        if (!broken.has(san)) broken.set(san, `${err.message} in ${game.fen()}`);
        continue;
      }
      for (const t of tokens) {
        if (!TOKEN_INDEX.has(t) && !missing.has(t)) missing.set(t, `${san} in ${game.fen()}`);
      }
      const back = tokensToSan(tokens);
      if (back !== san && !broken.has(san)) {
        broken.set(san, `round-trips to "${back}" in ${game.fen()}`);
      }
    }
    game.move(legal[Math.floor(Math.random() * legal.length)]);
    plies++;
  }
}

console.log(`vocabulary: ${VOCAB_SIZE} tokens = ${CORE_TOKENS.length} core moves + ${MODIFIER_TOKENS.length} modifiers + ${NERF_TOKENS.length} nerf`);
console.log(`checked ${checked} legal moves across ${GAMES} random games (${plies} plies)`);
let failed = false;
if (missing.size > 0) {
  failed = true;
  console.error(`\nTOKENS MISSING from vocabulary (${missing.size}):`);
  for (const [t, where] of missing) console.error(`  ${t.padEnd(8)} from ${where}`);
}
if (broken.size > 0) {
  failed = true;
  console.error(`\nROUND-TRIP failures (${broken.size}):`);
  for (const [san, why] of broken) console.error(`  ${san.padEnd(10)} ${why}`);
}
if (failed) process.exit(1);
console.log('every legal move tokenizes into the vocabulary and round-trips exactly');
