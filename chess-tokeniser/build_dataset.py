"""Build the tokenised training dataset from Lichess/standard-chess-games.

Reads the HF parquet shards month by month (via DuckDB's hf:// support, or
local parquet/CSV files), keeps only games with [%eval] annotations, applies
quality filters, tokenises with tokeniser.py, and writes zstd parquet shards
plus a vocab.json.

Run on a machine with internet access to huggingface.co, e.g.:

    pip install duckdb pyarrow
    python build_dataset.py --months 2025-05 2025-06 --out ./tokenised \
        --min-elo 1600 --min-plies 20 --accuracy-source glyph

For a quick validation slice add:  --limit 200000
For local files instead of hf://:  --local-glob 'path/to/*.parquet'
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from tokeniser import MovetextError, parse_movetext, tokenise_game

HF_GLOB = "hf://datasets/Lichess/standard-chess-games/data/year={y}/month={m}/*.parquet"
COLS = '"Site", "UTCDate", "WhiteElo", "BlackElo", "TimeControl", "Result", "Termination", "ECO", "movetext"'

OUT_SCHEMA = pa.schema(
    [
        ("site", pa.string()),
        ("utc_date", pa.string()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("time_control", pa.string()),
        ("result", pa.string()),
        ("eco", pa.string()),
        ("n_plies", pa.int16()),
        ("tokens", pa.large_string()),      # space-joined token stream
        ("cp", pa.list_(pa.int32())),       # eval after each ply; 30001 = missing
        ("n_disagreements", pa.int16()),    # inline glyph vs computed label
    ]
)


def process_row(row, accuracy_source: str):
    site, utc_date, w_elo, b_elo, tc, result, termination, eco, movetext = row
    game = parse_movetext(movetext)
    if not game.has_evals:
        return None
    tokens, cps, dis = tokenise_game(game, accuracy_source=accuracy_source)
    return (
        site,
        utc_date,
        int(w_elo or 0),
        int(b_elo or 0),
        tc,
        result,
        eco,
        len(game.plies),
        " ".join(tokens),
        cps,
        dis,
    )


def _worker(args):
    rows, accuracy_source = args
    out, errors = [], 0
    for row in rows:
        try:
            rec = process_row(row, accuracy_source)
            if rec is not None:
                out.append(rec)
        except MovetextError:
            errors += 1
    return out, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=[], help="e.g. 2025-05 2025-06")
    ap.add_argument("--local-glob", default=None, help="local parquet glob instead of hf://")
    ap.add_argument("--out", default="./tokenised")
    ap.add_argument("--min-elo", type=int, default=1600, help="min of both players' Elo")
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="cap games read per month (validation)")
    ap.add_argument("--accuracy-source", choices=["glyph", "computed", "none"], default="glyph")
    ap.add_argument("--batch", type=int, default=50_000)
    ap.add_argument("--shard-rows", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sources = (
        [args.local_glob]
        if args.local_glob
        else [HF_GLOB.format(y=m.split("-")[0], m=m.split("-")[1]) for m in args.months]
    )
    if not sources:
        ap.error("provide --months or --local-glob")

    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")

    vocab: Counter[str] = Counter()
    shard_rows: list[tuple] = []
    shard_idx = n_games = n_errors = n_dis_plies = total_plies = 0
    t0 = time.time()

    def flush(final=False):
        nonlocal shard_rows, shard_idx
        while len(shard_rows) >= args.shard_rows or (final and shard_rows):
            chunk, shard_rows = shard_rows[: args.shard_rows], shard_rows[args.shard_rows :]
            table = pa.Table.from_arrays(
                [pa.array(col, type=f.type) for col, f in zip(zip(*chunk), OUT_SCHEMA)],
                schema=OUT_SCHEMA,
            )
            path = os.path.join(args.out, f"shard-{shard_idx:05d}.parquet")
            pq.write_table(table, path, compression="zstd")
            print(f"  wrote {path} ({table.num_rows} games)")
            shard_idx += 1

    pool = mp.Pool(args.workers)
    for src in sources:
        print(f"== source: {src}")
        limit = f"LIMIT {args.limit}" if args.limit else ""
        query = f"""
            SELECT {COLS} FROM read_parquet('{src}')
            WHERE contains(movetext, '[%eval')
              AND "WhiteElo" >= {args.min_elo} AND "BlackElo" >= {args.min_elo}
            {limit}
        """
        reader = con.execute(query).fetch_record_batch(args.batch)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break
            rows = list(zip(*[batch.column(i).to_pylist() for i in range(batch.num_columns)]))
            n = max(1, len(rows) // args.workers)
            jobs = [(rows[i : i + n], args.accuracy_source) for i in range(0, len(rows), n)]
            for out, errs in pool.map(_worker, jobs):
                n_errors += errs
                for rec in out:
                    if rec[7] < args.min_plies:
                        continue
                    vocab.update(rec[8].split(" "))
                    total_plies += rec[7]
                    n_dis_plies += rec[10]
                    shard_rows.append(rec)
                    n_games += 1
            flush()
            rate = n_games / max(1e-9, time.time() - t0)
            print(f"  {n_games:,} games tokenised ({rate:,.0f}/s)", end="\r")
        print()
    pool.close()
    flush(final=True)

    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(
            {tok: c for tok, c in sorted(vocab.items(), key=lambda kv: -kv[1])}, f, indent=1
        )
    stats = {
        "games": n_games,
        "plies": total_plies,
        "tokens": int(sum(vocab.values())),
        "vocab_size": len(vocab),
        "san_parse_errors": n_errors,
        "glyph_vs_computed_disagreement_plies": n_dis_plies,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
