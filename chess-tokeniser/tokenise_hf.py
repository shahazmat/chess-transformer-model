# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb", "pyarrow", "numpy", "huggingface_hub"]
# ///
"""Build the dataset on Hugging Face Jobs instead of your laptop (a UV script —
deps declared above are installed by `hf jobs uv run`). It runs the *whole*
tokeniser on a rented CPU box: Stage 1 (`build_dataset.py`, which scans the
Lichess parquet — fast here because it runs next to the data on HF), Stage 2
(`pack.py`), then uploads the result to your dataset repo. Your laptop only
kicks off the job.

The pipeline code (build_dataset.py, tokeniser.py, pack.py, vocab.py and
js/vocab-data.js) has to be reachable inside the job. Two ways, pick one:

  A) Mount it from your machine (nothing to pre-upload) — run from the repo root:
       hf jobs uv run --flavor cpu-performance --timeout 4h -s HF_TOKEN \
         -v ./chess-tokeniser:/code/chess-tokeniser -v ./js:/code/js \
         -e OUT_REPO=<you>/lichess-chess-tokens \
         -e MONTHS="2025-05 2025-06" -e MIN_ELO=1600 -e MIN_PLIES=20 -e VAL_MONTHS=2025-06 \
         chess-tokeniser/tokenise_hf.py

  B) Pull it from an HF repo you upload once (keeps everything on HF):
       hf upload <you>/chess-pipeline ./chess-tokeniser chess-tokeniser --repo-type=dataset
       hf upload <you>/chess-pipeline ./js/vocab-data.js js/vocab-data.js --repo-type=dataset
       hf jobs uv run --flavor cpu-performance --timeout 4h -s HF_TOKEN \
         -e CODE_REPO=<you>/chess-pipeline -e OUT_REPO=<you>/lichess-chess-tokens \
         -e MONTHS="2025-05 2025-06" -e MIN_ELO=1600 -e MIN_PLIES=20 -e VAL_MONTHS=2025-06 \
         chess-tokeniser/tokenise_hf.py

Env vars:
  OUT_REPO   (required)  dataset repo to upload the bins to, e.g. you/lichess-chess-tokens
  HF_TOKEN   (secret)    write token (huggingface_hub reads it automatically)
  CODE_DIR   where the mounted pipeline lives (default /code/chess-tokeniser)
  CODE_REPO  HF dataset repo to pull the pipeline from if CODE_DIR is absent (option B)
  MONTHS     space-separated YYYY-MM list for build_dataset (default "2025-05 2025-06")
  MIN_ELO / MIN_PLIES / ACCURACY_SOURCE / LIMIT   build_dataset filters
  VAL_MONTHS space-separated months routed to val.bin; empty -> VAL_FRAC split
  VAL_FRAC   per-game val fraction when VAL_MONTHS is empty (default 0.05)
  UPLOAD_INTERMEDIATE  "1" (default) also uploads the re-packable parquet under tokenised/

NOTE: starting skeleton, not yet run on HF as-is. Do a cheap validation pass
first: --flavor cpu-upgrade with -e MONTHS=2025-05 -e LIMIT=200000.
"""
import os
import subprocess
import sys

from huggingface_hub import create_repo, snapshot_download, upload_folder

OUT_REPO = os.environ["OUT_REPO"]
CODE_DIR = os.environ.get("CODE_DIR", "/code/chess-tokeniser")

# ------------------------------------------------------------------ locate the code
if not os.path.isdir(CODE_DIR):
    code_repo = os.environ.get("CODE_REPO")
    if not code_repo:
        sys.exit(
            "pipeline code not found. Either mount it "
            "(-v ./chess-tokeniser:/code/chess-tokeniser -v ./js:/code/js) "
            "or set -e CODE_REPO=<you>/chess-pipeline."
        )
    print(f"== downloading pipeline code from {code_repo}", flush=True)
    root = "/tmp/code"
    snapshot_download(repo_id=code_repo, repo_type="dataset", local_dir=root)
    CODE_DIR = os.path.join(root, "chess-tokeniser")
VOCAB = os.environ.get("VOCAB", os.path.join(os.path.dirname(CODE_DIR), "js", "vocab-data.js"))
assert os.path.exists(os.path.join(CODE_DIR, "build_dataset.py")), f"build_dataset.py not under {CODE_DIR}"

# ------------------------------------------------------------------ Stage 1: tokenise
months = os.environ.get("MONTHS", "2025-05 2025-06").split()
build = [
    sys.executable, "build_dataset.py", "--months", *months, "--out", "/tmp/tok",
    "--min-elo", os.environ.get("MIN_ELO", "1600"),
    "--min-plies", os.environ.get("MIN_PLIES", "20"),
    "--accuracy-source", os.environ.get("ACCURACY_SOURCE", "glyph"),
]
if os.environ.get("LIMIT"):
    build += ["--limit", os.environ["LIMIT"]]
print("== Stage 1:", " ".join(build), flush=True)
subprocess.run(build, cwd=CODE_DIR, check=True)

# ------------------------------------------------------------------ Stage 2: pack
pack = [sys.executable, "pack.py", "--in", "/tmp/tok", "--out", "/tmp/data", "--vocab", VOCAB]
val_months = os.environ.get("VAL_MONTHS", "").split()
if val_months:
    pack += ["--val-months", *val_months]
else:
    pack += ["--val-frac", os.environ.get("VAL_FRAC", "0.05")]
print("== Stage 2:", " ".join(pack), flush=True)
subprocess.run(pack, cwd=CODE_DIR, check=True)

# ------------------------------------------------------------------ upload results
print(f"== uploading to {OUT_REPO}", flush=True)
create_repo(OUT_REPO, repo_type="dataset", exist_ok=True)
upload_folder(repo_id=OUT_REPO, repo_type="dataset", folder_path="/tmp/data",
              allow_patterns=["*.bin", "meta.pkl", "pack_stats.json"],
              commit_message="packed train/val bins")
if os.environ.get("UPLOAD_INTERMEDIATE", "1") == "1":
    upload_folder(repo_id=OUT_REPO, repo_type="dataset", folder_path="/tmp/tok", path_in_repo="tokenised",
                  allow_patterns=["*.parquet", "*.json"], commit_message="tokenised parquet intermediate")
print("== done", flush=True)
