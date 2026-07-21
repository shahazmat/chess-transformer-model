"""Inspect a packed chess dataset — the pack summary, a few decoded games from
the .bin stream, and a few rows of the parquet intermediate.

    hf download <you>/lichess-chess-tokens --repo-type dataset --local-dir ./data-inspect
    python inspect_data.py ./data-inspect

Works on a local pack output dir too. Reads only small prefixes, so it's fine on
a full-size dataset, not just the smoke-test slice.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "."


def _find(*names):
    for n in names:
        hits = glob.glob(os.path.join(d, n))
        if hits:
            return sorted(hits)[0]
    return None


# 1. pack summary --------------------------------------------------------------
stats = _find("pack_stats.json")
if stats:
    print("== pack_stats.json")
    print(json.dumps(json.load(open(stats)), indent=1))

# 2. meta + decode the first few framed games from the bins --------------------
meta_path = _find("meta.pkl", "*/meta.pkl")
if meta_path:
    meta = pickle.load(open(meta_path, "rb"))
    itos, eos = meta["itos"], meta["eos_id"]
    print(f"\n== meta: vocab_size={meta['vocab_size']} bos={meta['bos_id']} eos={eos}")
    for name in ("train.bin", "val.bin"):
        p = _find(name)
        if not p:
            continue
        arr = np.memmap(p, dtype=np.uint16, mode="r")
        head = np.asarray(arr[: min(len(arr), 20_000)])          # bounded read
        eos_pos = np.flatnonzero(head == eos)[:3]
        print(f"\n== {name}: {len(arr):,} tokens total - first {len(eos_pos)} games decoded:")
        start = 0
        for gi, e in enumerate(eos_pos):
            print(f"  [{gi}] " + " ".join(itos[int(i)] for i in head[start : e + 1]))
            start = e + 1

# 3. parquet intermediate rows -------------------------------------------------
shard = _find("tokenised/*.parquet", "*.parquet")
if shard:
    import pyarrow.parquet as pq

    t = pq.read_table(shard)
    print(f"\n== {os.path.relpath(shard, d)}: {t.num_rows:,} rows, columns={t.column_names}")
    for r in t.slice(0, 3).to_pylist():
        toks = r["tokens"].split()
        print(
            f"  {r['utc_date']}  {r['white_elo']} vs {r['black_elo']}  "
            f"plies={r['n_plies']}  disagreements={r['n_disagreements']}"
        )
        print(f"     tokens[:16] = {' '.join(toks[:16])} ...")
        print(f"     cp[:6]      = {r['cp'][:6]}")
