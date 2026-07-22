# Chess annotation tokeniser pipeline

Turns `Lichess/standard-chess-games` (Hugging Face) into a tokenised training
dataset using the scheme:

```
[accuracy][mate/check][capture][promotion][base-move]
exf8=Q+??  ->  [??] [+] [x] [=Q] [ef8]
Nc6        ->  [Nc6]
```

Accuracy tokens (`[?!]` `[?]` `[??]`) come either from the glyphs Lichess
embeds inline in analyzed games (`--accuracy-source glyph`, default) or are
recomputed from the `[%eval]` comments via Lichess's win%-drop formula
(`--accuracy-source computed`). Both are cross-checked; per-game disagreement
counts are stored in the output.

## The pipeline

Two stages. Stage 1 (`build_dataset.py`) turns raw movetext into an intermediate
parquet of per-game annotated token *strings* + per-ply evals — human-readable,
re-labellable, and independent of any model. Stage 2 (`pack.py`) frames those
games with game/Elo tokens, maps every token to its **frozen id** from
`../js/vocab-data.js`, and streams the result into nanoGPT-style `train.bin` /
`val.bin`.

```
HF Lichess parquet
      │  build_dataset.py   (filter [%eval] + Elo · parse · tokenise · per-ply cp)
      ▼
  tokenised/*.parquet       tokens="[Nf3] [x] [Bb5] [??] [e5] …"  + cp[] + metadata
      │  pack.py            (frame · map to frozen ids · time-split · uint16)
      ▼
  data/chess/{train.bin, val.bin, *.idx.npy, meta.pkl}   →  nanoGPT
```

Each game is framed with the vocab-v2 structural tokens plus, when the game
ended that way, a vocab-v3 game-end token:

```
<bos> <elo-W> <elo-B>   [move tokens…]   [<white-resign>|<black-resign>|<white-flag>|<black-flag>|<draw>]   <eos>
```

`<bos>`/`<eos>` are the **start/end-game tokens** that separate games in the
packed stream; the two `<elo-*>` buckets condition the model on each player's
strength. `pack.py --elo-dropout P` (default 0.15) independently replaces each
bucket with the `<elo-any>` sentinel for a fraction of games, so the model also
learns **unconditioned play** — at inference you can feed `<elo-any>` (per side)
to omit/neutralise Elo instead of being forced to specify it. The game-end
token says HOW the game ended and WHO acted, with the colour taken from the
Result header — never from parity, because players often resign right after
their own move: resign tokens name the loser of a decisive Termination-"Normal"
game without a mate in the movetext (mate is its own signal); flag tokens name
the loser of a decisive "Time forfeit" (training/gauge only — the harness has
no clocks, so they are never sampled in play); `<draw>` marks `1/2-1/2` with
Termination "Normal" (no colour — the outcome is symmetric). Mates,
timeout-draws, and junk terminations keep a bare `<eos>`. Structural tokens sit
at ids 5252–5267 and the end tokens at 5268–5272 in `../js/vocab-data.js` —
every extension is a pure append that disturbs no pre-existing id.

**The training vocabulary is `../js/vocab-data.js` (5,273 tokens), not
`build_dataset`'s `vocab.json`.** That `vocab.json` is a frequency report over
one run; the frozen dictionary is the geometry-derived superset of all legal SAN
that the JS harness, the packed data, and the trained model all share. `pack.py`
verifies coverage and drops (with a logged count) any game containing a token
outside it — ~0 on real data.

## Files

- `tokeniser.py` — movetext parser, eval maths, classifier, tokeniser
- `build_dataset.py` — Stage 1 CLI: streams HF parquet (or local files), filters
  to eval-annotated games, tokenises in parallel, writes zstd parquet + vocab
- `vocab.py` — loads the frozen dictionary from `../js/vocab-data.js`; owns the
  bracket→id translation, Elo bucketing, `<bos>/<eos>` game framing, and the
  colour-differentiated game-end classification (`Vocab._end_id`)
- `pack.py` — Stage 2 CLI: frames + maps + time-splits into `train.bin`/
  `val.bin`/`meta.pkl`, plus per-split `train.idx.npy`/`val.idx.npy` game-start
  indexes (uint64 offsets + total sentinel) for one-game-per-row sampling
- `nerf_batch.py` — training-time batch sampler: STRICT one-game-per-row
  (length-bucketed, right-padded, hard-guarded so games never share an
  attention window) with the bare-history nerf rule inside each game — a nerf
  token is only ever in context beside the move it labels — and the end group
  (`<resign>/<draw>/<eos>`) graded in every row
  (`train_chess_hf.py` embeds a byte-identical copy and patches it into
  nanoGPT's `get_batch`; see TRAINING.md "bare-history rule")
- `test_tokeniser.py`, `test_pack.py`, `test_nerf_batch.py` — unit tests
  (incl. a verbatim real game; the frozen vocab is checked for full coverage
  and JS↔Python parity; the bare-history rows are pinned against hand-worked
  examples and the embedded copy is checked byte-for-byte)
- `gen_synthetic.py` — offline validation corpus generator

## Run

```bash
pip install duckdb pyarrow numpy
python test_tokeniser.py && python test_pack.py && python test_nerf_batch.py  # sanity check

# --- Stage 1: tokenise (needs internet access to huggingface.co) ---
# validation slice: ~200k games from one month (minutes)
python build_dataset.py --months 2025-05 --limit 200000 --out ./tok_slice

# real run: ~2 months ≈ 10-12M annotated games before filters
python build_dataset.py --months 2025-05 2025-06 --out ./tokenised \
    --min-elo 1600 --min-plies 20 --accuracy-source glyph

# --- Stage 2: pack for training (offline, local) ---
# hold out a whole month as validation (recommended — no leakage, tracks reality)
python pack.py --in ./tokenised --out ./data/chess --val-months 2025-06
# or a deterministic per-game fraction when everything is one month:
python pack.py --in ./tok_slice --out ./data/chess --val-frac 0.05
```

`meta.pkl` carries `vocab_size`, `stoi`, `itos`, and the `<bos>`/`<eos>` +
game-end token ids so nanoGPT's `sample.py` can print readable token streams.

To push the Stage 1 intermediate to your HF dataset repo:

```bash
pip install huggingface_hub && hf auth login
hf upload-large-folder <you>/lichess-annotated-tokenised --repo-type dataset ./tokenised
```

## Output schema

| column | type | notes |
|---|---|---|
| site | string | lichess game URL (id) |
| utc_date, time_control, result, eco | string | metadata for filtering |
| termination | string | Lichess Termination header — drives the game-end token in pack.py |
| white_elo, black_elo | int16 | |
| n_plies | int16 | |
| tokens | large_string | space-joined token stream |
| cp | list<int32> | eval after each ply, White POV; mate → ±(30000−n); 30001 = missing |
| n_disagreements | int16 | plies where inline glyph ≠ computed label |

Keeping `cp` per ply means you can re-derive labels under different
thresholds later without re-pulling anything.

## Measured performance (validation run)

100k synthetic games in the exact dump format, full pipeline, 2-core box:
~2,700 games/s (≈1,400/s/core), 0 SAN parse errors, output ≈ 276 bytes/game
(zstd parquet). Extrapolation: 5M games ≈ 10 min of tokenisation on 8 cores;
wall-clock is dominated by scanning the source parquet over the network
(the `contains(movetext, '[%eval')` filter runs inside DuckDB, so only
matching rows are materialised). Expect ~1.5 GB output per 5M games.
