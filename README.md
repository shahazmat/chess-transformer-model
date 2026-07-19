# chess-gpt

A static-site harness for testing a chess AI that plays by **predicting the
next token** of a chess game. The AI isn't here yet — a mock model stands in
behind the exact interface the real one will use. No build step, no
dependencies, hostable on GitHub Pages as-is.

## Run it

```sh
node tools/serve.mjs        # -> http://localhost:4173
```

(any static server works; ES modules just can't load from `file://`)

**GitHub Pages:** push, then Settings → Pages → Deploy from a branch →
`main` / root. The site is fully static.

## How a computer move happens

1. Legal moves come from [chess.js](https://github.com/jhlywa/chess.js)
   and are tokenized into the scheme below.
2. The model is asked for a next-token distribution over the whole
   vocabulary.
3. **Mask:** every token that cannot continue a legal move sequence is set to
   probability zero. (On the first token of a move, the three nerf tokens are
   also allowed.)
4. Renormalize, then **sample**.
5. If a **nerf token** was drawn, the whole move is re-decoded with
   `quality` set — the model is now conditioned on "this move is an
   inaccuracy / mistake / blunder". Nerf tokens are masked out on the retry.
6. Steps 2–4 repeat per token until a core move token completes the
   sequence; the matching SAN is played. The move gets a `?!` / `?` / `??`
   glyph in the move list if it was nerfed.

The **Model output** panel in the UI shows the sampled chain and the top of
the masked distribution at every step of the last computer move.

## Tokenization

A move is a sequence of modifier tokens followed by **one core move token** —
the core token is what ends a move:

```
[#|+]?  [x]?  [=Q|=R|=B|=N]?  [core]

fxe8=Q+   ->  [+] [x] [=Q] [fe8]
Nxf3      ->  [x] [Nf3]
Raxe1#    ->  [#] [x] [Rae1]
e4        ->  [e4]
O-O       ->  [O-O]
```

Cores keep SAN's disambiguation so nothing is lossy: `fe8` is a pawn capture
core (source file + destination), piece cores may carry file / rank / square
disambiguators (`Nbd7`, `R1e2`, `Qh4e1`). `sanToTokens()` / `tokensToSan()`
in [js/vocab.js](js/vocab.js) convert both ways.

The vocabulary is **5,252 tokens**:

| group | count |
| --- | --- |
| core moves (pawn 176, N/B/R/Q/K 5,064, castling 2, generated geometrically) | 5,242 |
| modifiers `+` `#` `x` `=Q` `=R` `=B` `=N` | 7 |
| nerf `<inaccuracy>` `<mistake>` `<blunder>` | 3 |

Core generation is a deterministic, slightly loose superset of real SAN
(every core is geometrically possible on an empty board; impossible ones are
just permanently masked). `tools/vocab-check.mjs` plays random games and
asserts every legal move tokenizes into the vocabulary and round-trips:

```sh
node tools/vocab-check.mjs 500
```

## Plugging in the real AI

A model is any object with an async `predict(ctx)` returning either a
`Float32Array` / `number[]` of length `VOCAB_SIZE` (index `i` = probability
of `TOKENS[i]`) or a plain `{token: probability}` map. It is called **once
per token**:

```js
{
  fen,            // current position
  moves,          // SAN history: ['e4', 'c5', ...]
  historyTokens,  // same history as a flat token stream, nerf tokens included
  moveTokens,     // tokens emitted so far for the move being decoded
  turn,           // 'w' | 'b' — the side the model plays
  opponent,       // { rating: 1500 | null, site: 'lichess' | null }
  legalMoves,     // SAN list, provided for convenience — free to ignore
  quality,        // null, or 'inaccuracy'|'mistake'|'blunder' after a nerf
  vocab,          // the token list
}
```

Outputs don't need to be pre-masked or normalized — the engine masks
illegal tokens to zero and renormalizes whatever you return.

Swap the model by replacing [js/mock-model.js](js/mock-model.js) (imported in
[js/app.js](js/app.js)), or at runtime — useful while weights load
asynchronously:

```js
window.chessGpt.setModel({ name: 'the real one', async predict(ctx) { ... } });
```

Other seams: `window.chessGpt.loadFen(fen)` jumps to a position,
`window.chessGpt.config` exposes `ENGINE_CONFIG` (sampling temperature,
disable nerf tokens, think-delay).

## Files

```
index.html          page shell
style.css           minimalist theme
js/vocab.js         token vocabulary + sanToTokens / tokensToSan   <- swap for your tokenizer
js/engine.js        mask -> renormalize -> sample decode loop      <- the AI's socket
js/mock-model.js    placeholder model                              <- replace with the real AI
js/app.js           game flow, UI wiring
js/board.js         SVG board rendering + input
js/pieces.js        geometric piece shapes
lib/chess.js        vendored chess.js 1.4.0 (rules, legality, SAN)
tools/serve.mjs     dev server:  node tools/serve.mjs
tools/vocab-check.mjs  vocabulary/round-trip validation
```

## Training the real model

[TRAINING.md](TRAINING.md) is the plan for building the training set from
evaluated Lichess games (eval-derived nerf labels, this vocabulary, nanoGPT
packing).

## Notes

- The mock scores moves with a shallow greedy heuristic, answers each
  predict() call with exact next-token marginals of that distribution, and
  emits nerf tokens more often the lower the opponent's rating (`900 lichess`
  blunders a lot; leave rating empty for a middling default). The rating
  *site* is plumbed through but unused by the mock — a real model might embed
  it, since ratings aren't comparable across sites.
- Sampling is plain categorical over the masked distribution
  (`ENGINE_CONFIG.temperature`, default 1.0).
- If the model puts zero mass on every allowed token, the engine falls back
  to uniform over legal continuations and logs a console warning.
