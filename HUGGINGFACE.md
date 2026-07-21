# Running the chess-gpt workflow on Hugging Face

This is the operational runbook for taking the pipeline in
[`chess-tokeniser/`](chess-tokeniser/README.md) from raw Lichess games to a
trained checkpoint using **Hugging Face** for both data storage and rented GPU
compute (HF **Jobs**). It complements [TRAINING.md](TRAINING.md) (the *what* and
*why*) with the exact commands.

## Mental model: what runs where

Two environments, and keeping them separate is what keeps the bill small:

| Stage | Where | Why |
|---|---|---|
| **1. Tokenise** (`build_dataset.py`) | your machine or a cheap CPU box | I/O-bound scan of ~30 GB/month of Lichess parquet — never do this on a GPU |
| **2. Pack** (`pack.py`) | same CPU box | seconds of CPU; produces the `.bin`s |
| **Store** train/val/test | HF **dataset repo** | durable, versioned, mountable into Jobs |
| **3. Train** (`train_chess_hf.py`) | HF **Jobs** (rented GPU) | the only part that needs a GPU |
| **Store** checkpoints | HF **model repo** | Jobs storage is ephemeral — push or lose it |

```
   your machine / CPU box                 Hugging Face
 ┌────────────────────────┐        ┌─────────────────────────┐
 │ build_dataset.py        │        │  dataset repo           │
 │ pack.py                 │  push  │  you/lichess-chess-tokens│
 │  → data/chess/*.bin ────┼───────▶│   train.bin val.bin meta │
 └────────────────────────┘        └───────────┬─────────────┘
                                                │ mount/download
                                    ┌───────────▼─────────────┐
                                    │  HF Job (rented GPU)     │
                                    │  train_chess_hf.py       │
                                    │   → out/ckpt.pt ─────────┼──┐
                                    └─────────────────────────┘  │ push
                                    ┌─────────────────────────┐  │
                                    │  model repo you/chess-gpt│◀─┘
                                    └─────────────────────────┘
```

---

## 0. One-time setup

### Install the CLI

```bash
# standalone installer (Linux/macOS)
curl -LsSf https://hf.co/cli/install.sh | bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
# or via pip
pip install -U "huggingface_hub[cli]"

hf version
```

### Authenticate

Create a **write** token at <https://huggingface.co/settings/tokens>, then:

```bash
hf auth login                 # interactive (browser), or:
hf auth login --token hf_xxx  # non-interactive / scripted
```

The token is also read from the `HF_TOKEN` environment variable — set that in
your shell so the training job can forward it (see §3).

### Enable Jobs (billing)

HF Jobs is pay-as-you-go and requires a **positive credit balance** on your
account (Pro, Team, or Enterprise, or prepaid credits): set it up at
<https://huggingface.co/settings/billing>. You only pay for the seconds a job
runs.

### Create the two repos

```bash
hf repo create <you>/lichess-chess-tokens --repo-type dataset
hf repo create <you>/chess-gpt            --repo-type model
```

(`hf upload` will also create a missing repo on first push, so this is optional.)

---

## 1–2. Produce the data (local, CPU)

Full details in [chess-tokeniser/README.md](chess-tokeniser/README.md); the short
version:

```bash
cd chess-tokeniser
pip install duckdb pyarrow numpy
python test_tokeniser.py && python test_pack.py     # sanity check

# Stage 1 — tokenise (reads the public Lichess dataset over hf://, no token needed)
python build_dataset.py --months 2025-05 2025-06 --out ./tokenised \
    --min-elo 1600 --min-plies 20 --accuracy-source glyph

# Stage 2 — pack into train/val bins (hold out June as validation)
python pack.py --in ./tokenised --out ./data/chess --val-months 2025-06
```

You now have `chess-tokeniser/data/chess/{train.bin, val.bin, meta.pkl}`.

> **Want a three-way train/test/val split?** `pack.py` currently emits `train`
> + `val`. Two options: (a) reserve a third month and run `pack.py` a second
> time to a different `--out` to produce a `test` set, or (b) ask me to add a
> `--test-months` flag (small change). nanoGPT only trains on train/val, so the
> test set is for your own final held-out evaluation.

> **Tip — validate cheaply first.** Use `--limit 200000` on `build_dataset.py`
> for a fast slice, and lower `--min-elo` if you want the low `<elo-*>` buckets
> populated (needed for the model to convincingly *play weak*).

---

## 3. Upload the data to your dataset repo

`hf upload` takes `REPO_ID  LOCAL_PATH  PATH_IN_REPO`:

```bash
# put the three files at the root of the dataset repo
hf upload <you>/lichess-chess-tokens ./data/chess . --repo-type=dataset
```

Optionally also archive the re-packable parquet intermediate (recommended as the
*canonical* dataset — it survives a vocab change, since you can re-`pack` it):

```bash
hf upload <you>/lichess-chess-tokens ./tokenised tokenised --repo-type=dataset
```

Large files are handled automatically (Xet/LFS). Verify in the browser at
`https://huggingface.co/datasets/<you>/lichess-chess-tokens`.

---

## 4. Launch the training job (rented GPU)

The entrypoint is [`chess-tokeniser/train_chess_hf.py`](chess-tokeniser/train_chess_hf.py)
— a self-contained **UV script** (its dependencies are declared inline). It
fetches nanoGPT, downloads your data from the dataset repo, trains, and pushes
the checkpoint to your model repo.

**Always do a cheap smoke run first** to prove the whole path end-to-end before
paying for a big GPU:

```bash
export HF_TOKEN=hf_xxx     # write token, forwarded into the job below

hf jobs uv run \
  --flavor t4-small --timeout 2h \
  -s HF_TOKEN \
  -e DATA_REPO=<you>/lichess-chess-tokens \
  -e MODEL_REPO=<you>/chess-gpt \
  -e PROFILE=smoke \
  chess-tokeniser/train_chess_hf.py
```

- `--flavor` picks the hardware (`hf jobs hardware` lists them all; table in §7).
- `-s HF_TOKEN` forwards your local `HF_TOKEN` as an **encrypted secret** so the
  script can download/upload repos.
- `-e KEY=VALUE` sets plain environment variables the script reads.
- The default UV image installs a **CUDA build of torch** automatically on GPU
  flavors, so no custom image is needed.

The command prints a **job URL** (`https://huggingface.co/jobs/<you>/<id>`) —
open it for live logs, or watch from the terminal:

```bash
hf jobs ps                 # your running jobs (hf jobs ps -a for all)
hf jobs inspect <job_id>   # status + details
hf jobs logs <job_id>      # stream logs
hf jobs cancel <job_id>    # stop a job
```

When the smoke run works, scale up to the real network:

```bash
hf jobs uv run \
  --flavor a10g-large --timeout 6h \
  -s HF_TOKEN \
  -e DATA_REPO=<you>/lichess-chess-tokens \
  -e MODEL_REPO=<you>/chess-gpt \
  -e PROFILE=full \
  chess-tokeniser/train_chess_hf.py
```

> ⚠️ **The single biggest footgun: the default job timeout is 30 minutes.** A
> job that hits it is *killed*, checkpoint and all. **Always pass `--timeout`**
> (`30m`, `2h`, `1.5h`, `1d`…) with headroom over your expected runtime.

> ⚠️ **Job storage is ephemeral.** Everything in the container vanishes when the
> job ends. That is why the script pushes `ckpt.pt` to your model repo — nothing
> else survives.

### Runs longer than one job: chunked resume

For a multi-hour `full` run you don't want riding on a single job. The script
supports **resume**: each job trains `MAX_ITERS` more iterations, pushes the
checkpoint, and the next job continues from it.

```bash
# run this repeatedly; each call advances the checkpoint by 20k iters
hf jobs uv run --flavor a10g-large --timeout 6h -s HF_TOKEN \
  -e DATA_REPO=<you>/lichess-chess-tokens -e MODEL_REPO=<you>/chess-gpt \
  -e PROFILE=full -e RESUME=1 -e MAX_ITERS=20000 \
  chess-tokeniser/train_chess_hf.py
```

> **Advanced alternative — mountable storage buckets.** The CLI can mount an HF
> Storage Bucket read-write with `-v hf://buckets/<you>/ckpts:/out:rw`; point the
> script's `out_dir` at that mount and checkpoints become durable *live* (they
> survive even a timeout), no push step needed. You can likewise feed data by
> mounting the dataset repo instead of downloading it in-script:
> `-v hf://datasets/<you>/lichess-chess-tokens:/data`. Worth it for very long
> single runs.

---

## 5. Retrieve the checkpoint and plug it into the harness

```bash
hf download <you>/chess-gpt --repo-type model --local-dir ./ckpt
# → ./ckpt/ckpt.pt  and  ./ckpt/meta.pkl
```

That completes the data/compute loop. Turning `ckpt.pt` into something the
browser harness can call (`window.chessGpt.setModel`, milestone 6 in
[TRAINING.md](TRAINING.md)) is a separate export step — ONNX / transformers.js in
the browser, or a small local inference server behind `predict(ctx)`. The
model's output vocabulary is the 5,267-token space in
[`js/vocab-data.js`](js/vocab-data.js); `meta.pkl` carries the matching
`stoi`/`itos`.

---

## 6. Hardware & cost cheat-sheet

Pay-as-you-go, per-minute. Common picks for this project (full list:
`hf jobs hardware`):

| flavor | GPU | $/hr | use |
|---|---|---:|---|
| `cpu-upgrade` | — | $0.03 | (could even run `build_dataset`/`pack` here) |
| `t4-small` | 1× T4 16 GB | $0.40 | smoke runs, the `smoke` profile |
| `l4x1` | 1× L4 24 GB | $0.80 | cheap real training |
| `a10g-large` | 1× A10G 24 GB | $1.50 | the `full` 8×512 run |
| `a100-large` | 1× A100 80 GB | $2.50 | fastest single-GPU |

Rough order of magnitude: a `smoke` run is minutes (cents); the `full` run is a
few hours on `a10g-large` (~$5–15 depending on `max_iters`). Start small, watch
the loss curve in the logs, then scale.

---

## 7. Gotchas & troubleshooting

- **Job killed at ~30 min** → you forgot `--timeout`. Set it every time.
- **Checkpoint gone after the job** → ephemeral storage; make sure the job
  reached the upload step (check logs), or use chunked resume / a bucket.
- **Don't run data prep on GPU** → `build_dataset.py`/`pack.py` are CPU/I/O work;
  running them on a rented GPU just burns money. Prep locally, upload the small
  `.bin`s (README estimates ~1.2 GB/month).
- **`compile=True` errors** on some torch/driver combos → set `compile = False`
  in the config block of `train_chess_hf.py`.
- **Auth inside the job** → the script needs `HF_TOKEN`; pass it with `-s HF_TOKEN`
  (forwards your local env var). A **read** token suffices to pull data; you need
  **write** to push checkpoints.
- **Private data** → reading the public `Lichess/standard-chess-games` over
  `hf://` needs no token; only a *private* mirror or your own private dataset
  repo does. For DuckDB `hf://` reads of private data, configure an HF secret in
  DuckDB or set `HF_TOKEN`.
- **`pack.py` on a remote box** → it reads the frozen vocab from
  `../js/vocab-data.js`; if you run it off your machine, ship the repo's `js/`
  dir too or pass `--vocab /path/to/vocab-data.js`. (Simplest: pack during local
  prep and never run it on HF.)
- **Windows** → your local prep runs fine on Windows; only the Job runs on Linux,
  so multiprocessing/`fork` concerns don't apply to your box.

---

## Sources

- [Run and manage Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
- [Jobs overview](https://huggingface.co/docs/hub/en/jobs-overview) · [Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing)
- [hf CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [UV scripts](https://docs.astral.sh/uv/guides/scripts/) · [nanoGPT](https://github.com/karpathy/nanoGPT)
