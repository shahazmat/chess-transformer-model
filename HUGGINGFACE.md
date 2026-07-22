# Running the GPCT workflow on Hugging Face

This is the operational runbook for taking the pipeline in
[`chess-tokeniser/`](chess-tokeniser/README.md) from raw Lichess games to a
trained checkpoint using **Hugging Face** for both data storage and rented GPU
compute (HF **Jobs**). It complements [TRAINING.md](TRAINING.md) (the *what* and
*why*) with the exact commands.

> **Windows / PowerShell?** The multi-line commands below use bash
> line-continuation (a trailing `\`). PowerShell doesn't understand `\` — it runs
> each line separately, producing errors like `Too small: expected array` and
> `-v : The term '-v' is not recognized`. Fix: put the whole command **on one
> line**, or replace each trailing `\` with a backtick `` ` ``, or run it in **Git
> Bash / WSL**. Also set env vars the PowerShell way — `$env:HF_TOKEN = "hf_…"`,
> not `export HF_TOKEN=…`.

## Mental model: what runs where

Two environments, and keeping them separate is what keeps the bill small:

| Stage | Where | Why |
|---|---|---|
| **1. Tokenise** (`build_dataset.py`) | HF **CPU Job** (recommended) or your machine | I/O-bound scan of ~30 GB/month of Lichess parquet — runs *next to the data* on HF; never on a GPU |
| **2. Pack** (`pack.py`) | same CPU Job / box | seconds of CPU; produces the `.bin`s |
| **Store** train/val/test | HF **dataset repo** | durable, versioned, mountable into Jobs |
| **3. Train** (`train_chess_hf.py`) | HF **Jobs** (rented GPU) | the only part that needs a GPU |
| **Store** checkpoints | HF **model repo** | Jobs storage is ephemeral — push or lose it |

```
   your machine / CPU box                 Hugging Face
 ┌────────────────────────┐         ┌─────────────────────────┐
 │ build_dataset.py        │        │  dataset repo           │
 │ pack.py                 │  push  │  you/lichess-chess-tokens│
 │  → data/chess/*.bin ────┼───────▶│   train.bin val.bin meta │
 └────────────────────────┘         └───────────┬─────────────┘
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
your shell so jobs can forward it with `-s HF_TOKEN`.

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

## 1–3. Build the dataset

Two ways to produce and store the packed bins. **Option A runs everything on
HF's machines** — your laptop only launches the job — and is the recommended
path for the real multi-month run, because `build_dataset.py` scans the Lichess
parquet *and that data already lives on HF*, so the scan (the wall-clock
bottleneck) happens next to it instead of streaming ~30 GB to your machine.
Option B is the local path — handier for quick iteration on the tokeniser.

### Option A — build on HF Jobs (recommended)

One CPU Job does Stage 1 → Stage 2 → upload, via the launcher
[`chess-tokeniser/tokenise_hf.py`](chess-tokeniser/tokenise_hf.py). Run this from
the repo root; the `-v` flags mount your local pipeline code into the job (a
tiny one-time sync — no CPU work on your laptop):

```bash
export HF_TOKEN=hf_xxx        # write token, forwarded into the job

# cheap validation slice first (8 vCPU, one month, capped) — pennies:
hf jobs uv run --flavor cpu-upgrade --timeout 1h -s HF_TOKEN \
  -v ./chess-tokeniser:/code/chess-tokeniser -v ./js:/code/js \
  -e OUT_REPO=<you>/lichess-chess-tokens \
  -e MONTHS=2025-05 -e LIMIT=200000 -e MIN_ELO=1600 -e MIN_PLIES=20 -e VAL_FRAC=0.05 \
  chess-tokeniser/tokenise_hf.py

# the real run (32 vCPU box, all cores used by build_dataset's multiprocessing):
hf jobs uv run --flavor cpu-performance --timeout 4h -s HF_TOKEN \
  -v ./chess-tokeniser:/code/chess-tokeniser -v ./js:/code/js \
  -e OUT_REPO=<you>/lichess-chess-tokens \
  -e MONTHS="2025-05 2025-06" -e MIN_ELO=1600 -e MIN_PLIES=20 -e VAL_MONTHS=2025-06 \
  chess-tokeniser/tokenise_hf.py
```

**Windows / PowerShell** — no `\` continuations; put it on one line and pull the
code from an HF repo (`-e CODE_REPO=…`) instead of `-v` mounts so there's no
nested quoting. Upload the pipeline once, then launch:

```powershell
$env:HF_TOKEN = "hf_xxx"
hf upload <you>/chess-pipeline ./chess-tokeniser chess-tokeniser --repo-type=dataset
hf upload <you>/chess-pipeline ./js/vocab-data.js js/vocab-data.js --repo-type=dataset

# validation slice (one line):
hf jobs uv run --flavor cpu-upgrade --timeout 2h -s HF_TOKEN -e CODE_REPO=<you>/chess-pipeline -e OUT_REPO=<you>/lichess-chess-tokens -e MONTHS=2025-05 -e LIMIT=200000 -e MIN_ELO=1600 -e MIN_PLIES=20 -e VAL_FRAC=0.05 chess-tokeniser/tokenise_hf.py

# full run (one line):
hf jobs uv run --flavor cpu-performance --timeout 4h -s HF_TOKEN -e CODE_REPO=<you>/chess-pipeline -e OUT_REPO=<you>/lichess-chess-tokens -e MONTHS="2025-05 2025-06" -e MIN_ELO=1600 -e MIN_PLIES=20 -e VAL_MONTHS=2025-06 chess-tokeniser/tokenise_hf.py
```

Monitor exactly as in §4 (`hf jobs ps` / `logs` / `cancel`, or the printed job
URL). When it finishes, `train.bin`/`val.bin`/`meta.pkl` plus the
`train.idx.npy`/`val.idx.npy` game indexes are in your dataset repo (with the
re-packable parquet under `tokenised/`), and **your laptop never tokenised
anything**.

> If you'd rather not mount local code each run, upload the pipeline to an HF repo
> once and pass `-e CODE_REPO=<you>/chess-pipeline` instead of the `-v` flags — see
> the header of [`tokenise_hf.py`](chess-tokeniser/tokenise_hf.py). Either way it
> all stays on your HF account.

### Option B — build locally, then upload

```bash
cd chess-tokeniser
pip install duckdb pyarrow numpy
python test_tokeniser.py && python test_pack.py     # sanity check

python build_dataset.py --months 2025-05 2025-06 --out ./tokenised \
    --min-elo 1600 --min-plies 20 --accuracy-source glyph
python pack.py --in ./tokenised --out ./data/chess --val-months 2025-06

# upload the bins (REPO_ID  LOCAL_PATH  PATH_IN_REPO); Xet/LFS handles large files
hf upload <you>/lichess-chess-tokens ./data/chess . --repo-type=dataset
hf upload <you>/lichess-chess-tokens ./tokenised tokenised --repo-type=dataset
```

Verify either way in the browser at
`https://huggingface.co/datasets/<you>/lichess-chess-tokens`.

> **Want a three-way train/test/val split?** `pack.py` currently emits `train`
> + `val`. Either reserve a third month and run `pack.py`/the launcher again to a
> separate output, or ask me to add a `--test-months` flag. nanoGPT only trains
> on train/val, so the test set is for your own final held-out evaluation.

> **Tip.** Lower `--min-elo` (`-e MIN_ELO=…`) if you want the low `<elo-*>`
> buckets populated — needed for the model to convincingly *play weak*.

> **Elo conditioning.** The launcher passes `--elo-dropout 0.15` by default, which
> swaps each player's Elo bucket for the `<elo-any>` sentinel on ~15% of slots so
> the model also learns to play *without* an Elo hint. Tune with `-e ELO_DROPOUT=…`
> (`0` disables it, forcing every game to carry both Elos).

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

- `--flavor` picks the hardware (`hf jobs hardware` lists them all; table in §6).
- `-s HF_TOKEN` forwards your local `HF_TOKEN` as an **encrypted secret** so the
  script can download/upload repos.
- `-e KEY=VALUE` sets plain environment variables the script reads.
- The default UV image installs a **CUDA build of torch** automatically on GPU
  flavors, so no custom image is needed.

> **The job auto-applies the bare-history nerf transform** (TRAINING.md §5):
> `train_chess_hf.py` pins nanoGPT to a commit and reroutes `get_batch` through
> an embedded copy of `chess-tokeniser/nerf_batch.py` (`test_nerf_batch.py`
> keeps the two byte-identical). Only ~10–25% of each batch is graded, so give
> runs more iterations than a vanilla nanoGPT budget — e.g. add
> `-e MAX_ITERS=6000` to the smoke command above — and never compare losses
> against a checkpoint trained without the transform.

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
  --flavor a100-large --timeout 24h \
  -s HF_TOKEN \
  -e DATA_REPO=<you>/lichess-chess-tokens \
  -e MODEL_REPO=<you>/chess-gpt \
  -e PROFILE=full \
  chess-tokeniser/train_chess_hf.py
```

Sizing the full run (8×512, block 1024, batch 64, 100k iters ≈ 6.5B tokens
through the model): roughly **4–6 h ≈ $11–15 on `a100-large`**, or ~9–11 h ≈
$14–17 on the slower `a10g-large`. The a100 is faster *and* usually cheaper
here. `--timeout` is a kill-cap, not a cost — you pay for seconds actually
run, so always set it far above the estimate. While training, the script also
**pushes `ckpt.pt` to the model repo every `PUSH_EVERY_MIN` minutes**
(default 20), so even a crashed or timed-out job keeps all but the last few
minutes of progress.

> ⚠️ **The single biggest footgun: the default job timeout is 30 minutes.** A
> job that hits it is *killed*, checkpoint and all. **Always pass `--timeout`**
> (`30m`, `2h`, `1.5h`, `1d`…) with headroom over your expected runtime.

> ⚠️ **Job storage is ephemeral.** Everything in the container vanishes when the
> job ends. That is why the script pushes `ckpt.pt` to your model repo — nothing
> else survives.

### Runs longer than one job: chunked resume

For a multi-hour `full` run you can split the schedule across jobs. Two facts
make it correct: `MAX_ITERS` is the **absolute iteration this job stops at**
(nanoGPT resumes at the checkpoint's iteration and trains until `max_iters`),
so it must be raised each call — repeating the same value trains nothing. And
`TOTAL_ITERS` pins the lr-decay horizon to the final goal so every chunk
decays on the same curve instead of stretching the schedule.

```bash
# chunk 1 (fresh):            stops at iter 25k
hf jobs uv run --flavor a100-large --timeout 6h -s HF_TOKEN \
  -e DATA_REPO=<you>/lichess-chess-tokens -e MODEL_REPO=<you>/chess-gpt \
  -e PROFILE=full -e TOTAL_ITERS=100000 -e MAX_ITERS=25000 \
  chess-tokeniser/train_chess_hf.py
# chunk 2: add RESUME=1 and raise MAX_ITERS to 50000; then 75000; then 100000
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

That completes the data/compute loop. To **play the checkpoint in the browser
harness**, run the local inference server (milestone 6's `predict(ctx)` seam):

```bash
pip install torch huggingface_hub
python tools/model_server.py --repo <you>/chess-gpt        # downloads ckpt.pt + meta.pkl
# optional: --model-elo 1600   (Elo bucket the model plays at; default <elo-any>)
#           --local ./out      (serve a local ckpt dir instead of downloading)
```

Then open the harness (`node tools/serve.mjs`) — [`js/app.js`](js/app.js)
auto-detects the server on `127.0.0.1:8123` and swaps the mock model for the
real one (the page console names the live model; the server owns the
bare-history context assembly documented in [`js/engine.js`](js/engine.js)).
Restart the server after pushing a new checkpoint to pick it up. ONNX /
transformers.js in-browser inference remains the polished follow-up. The
model's output vocabulary is the 5,273-token space in
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
