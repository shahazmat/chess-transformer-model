"""Stage D — pack tokenised games into a nanoGPT-style training corpus.

Reads the parquet shards written by build_dataset.py, frames each game with the
frozen vocabulary

    <bos> <elo-W> <elo-B>  <move tokens...>  <eos>

maps every token to its id in js/vocab-data.js, splits into train/val, and
streams the id stream to `train.bin` / `val.bin` as uint16 (the shape nanoGPT's
data loaders memmap directly). Also writes `meta.pkl` (vocab_size, stoi, itos,
special ids) so sample.py can print readable token streams, and a
`pack_stats.json` report.

    pip install pyarrow numpy
    # time-based split (recommended): hold out one month as validation
    python pack.py --in ./tokenised --out ./data/chess --val-months 2025-06

    # or a deterministic per-game fraction (e.g. when all data is one month)
    python pack.py --in ./tokenised --out ./data/chess --val-frac 0.05

The split is leakage-free either way: whole games go to exactly one side, and
the fraction split hashes the game id so it is reproducible across runs.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import time

import random

import numpy as np
import pyarrow.parquet as pq

from vocab import ELO_ANY, get_vocab

# columns we need from the build_dataset schema
READ_COLS = ["utc_date", "white_elo", "black_elo", "n_plies", "tokens", "site"]


def _year_month(utc_date: str) -> str:
    """'2025.05.15' or '2025-05-15' -> '2025-05'."""
    return (utc_date or "")[:7].replace(".", "-")


def _in_val(site: str, val_frac: float) -> bool:
    """Deterministic per-game hash split (whole games, reproducible)."""
    h = int(hashlib.md5((site or "").encode()).hexdigest()[:8], 16)
    return (h % 10_000) < val_frac * 10_000


def _shard_paths(spec: str) -> list[str]:
    if any(c in spec for c in "*?["):
        return sorted(glob.glob(spec))
    if os.path.isdir(spec):
        return sorted(glob.glob(os.path.join(spec, "*.parquet")))
    return [spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="dir, glob, or file of build_dataset parquet shards")
    ap.add_argument("--out", default="./data/chess", help="output dir for train.bin/val.bin/meta.pkl")
    ap.add_argument("--vocab", default=None, help="path to js/vocab-data.js (defaults to repo copy)")
    ap.add_argument("--val-months", nargs="*", default=[], help="YYYY-MM months routed to val.bin (time split)")
    ap.add_argument("--val-frac", type=float, default=0.0, help="per-game fraction to val when no --val-months")
    ap.add_argument("--min-elo", type=int, default=0, help="post-filter: drop games below this (min of both)")
    ap.add_argument("--max-elo", type=int, default=0, help="post-filter: drop games above this (max of both); 0=off")
    ap.add_argument("--min-plies", type=int, default=0, help="post-filter: drop games shorter than this")
    ap.add_argument("--elo-dropout", type=float, default=0.15,
                    help="per-slot prob of replacing an Elo bucket with <elo-any>, for unconditioned play; 0=off")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed for the Elo-dropout choices (reproducible)")
    ap.add_argument("--batch", type=int, default=50_000)
    args = ap.parse_args()

    shards = _shard_paths(args.inp)
    if not shards:
        ap.error(f"no parquet shards matched: {args.inp!r}")
    os.makedirs(args.out, exist_ok=True)
    vocab = get_vocab(args.vocab)
    val_months = set(args.val_months)
    rng = random.Random(args.seed)   # drives the per-slot Elo dropout, reproducibly

    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")
    t0 = time.time()

    n_seen = n_packed = n_filtered = n_unknown = 0
    tok_train = tok_val = g_train = g_val = 0
    unknown_examples: dict[str, int] = {}

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for shard in shards:
            print(f"== {shard}")
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=args.batch, columns=READ_COLS):
                cols = {name: batch.column(name).to_pylist() for name in READ_COLS}
                for utc_date, w_elo, b_elo, n_plies, tokens, site in zip(
                    cols["utc_date"], cols["white_elo"], cols["black_elo"],
                    cols["n_plies"], cols["tokens"], cols["site"],
                ):
                    n_seen += 1
                    w_elo, b_elo = int(w_elo or 0), int(b_elo or 0)
                    if args.min_elo and min(w_elo, b_elo) < args.min_elo:
                        n_filtered += 1
                        continue
                    if args.max_elo and max(w_elo, b_elo) > args.max_elo:
                        n_filtered += 1
                        continue
                    if args.min_plies and (n_plies or 0) < args.min_plies:
                        n_filtered += 1
                        continue

                    ids, unknown = vocab.encode_game(tokens, w_elo, b_elo,
                                                     elo_dropout=args.elo_dropout, rng=rng)
                    if ids is None:
                        n_unknown += 1
                        for u in unknown:
                            unknown_examples[u] = unknown_examples.get(u, 0) + 1
                        continue

                    to_val = _year_month(utc_date) in val_months if val_months else _in_val(site, args.val_frac)
                    buf = np.asarray(ids, dtype=np.uint16).tobytes()
                    if to_val:
                        f_val.write(buf)
                        tok_val += len(ids)
                        g_val += 1
                    else:
                        f_train.write(buf)
                        tok_train += len(ids)
                        g_train += 1
                    n_packed += 1
            print(f"  packed {n_packed:,} games ({n_packed / max(1e-9, time.time() - t0):,.0f}/s)")

    # meta.pkl — nanoGPT reads vocab_size/stoi/itos; the rest aids the harness.
    with open(os.path.join(args.out, "meta.pkl"), "wb") as f:
        pickle.dump(
            {
                "vocab_size": vocab.size,
                "stoi": vocab.stoi,
                "itos": vocab.itos,
                "bos_id": vocab.bos_id,
                "eos_id": vocab.eos_id,
                "core_count": vocab.core_count,
                "elo_tokens": [t for t in vocab.tokens if t.startswith("<elo-") and t != ELO_ANY],
                "elo_any_id": vocab.elo_any_id,
            },
            f,
        )

    stats = {
        "shards": len(shards),
        "games_seen": n_seen,
        "games_packed": n_packed,
        "games_filtered": n_filtered,
        "games_dropped_unknown_token": n_unknown,
        "train_games": g_train,
        "val_games": g_val,
        "train_tokens": tok_train,
        "val_tokens": tok_val,
        "tokens_per_game": round((tok_train + tok_val) / max(1, n_packed), 1),
        "split": ("months:" + ",".join(sorted(val_months))) if val_months else f"frac:{args.val_frac}",
        "elo_dropout": args.elo_dropout,
        "unknown_token_top": dict(sorted(unknown_examples.items(), key=lambda kv: -kv[1])[:20]),
        "vocab_size": vocab.size,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "pack_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    if n_unknown:
        print(
            f"\nNOTE: {n_unknown:,} games dropped for out-of-vocab tokens. On real "
            "Lichess data this should be ~0; a nonzero count on real data means the "
            "frozen dictionary does not cover some legal SAN — investigate before training."
        )


if __name__ == "__main__":
    main()
