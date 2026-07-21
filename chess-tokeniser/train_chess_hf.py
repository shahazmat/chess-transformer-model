# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "huggingface_hub"]
# ///
"""Training entrypoint for Hugging Face Jobs (a UV script — deps are declared
above and installed by `hf jobs uv run`). It is deliberately thin: it does the
HF-specific glue and delegates the actual transformer to nanoGPT, matching
TRAINING.md.

What it does, in order:
  1. fetch nanoGPT (zip, no git needed in the container),
  2. download the packed data (train.bin/val.bin/meta.pkl) from your dataset repo
     into nanoGPT/data/chess/,
  3. (optional) resume from the last checkpoint pushed to your model repo,
  4. write a nanoGPT config and run training,
  5. push the checkpoint (+ meta.pkl) back to your model repo — Jobs storage is
     EPHEMERAL, so anything not pushed is lost when the job ends.

Configure via environment variables (set with `-e` / `-s` on `hf jobs uv run`):
  DATA_REPO   (required)  e.g. you/lichess-chess-tokens   — dataset repo with the bins
  MODEL_REPO  (required)  e.g. you/chess-gpt              — model repo for checkpoints
  HF_TOKEN    (secret)    write token; huggingface_hub reads it automatically
  PROFILE     smoke|full  smoke = tiny 4x128 validation net (default); full = 8x512
  MAX_ITERS   override the iteration count for this job (enables chunked runs)
  RESUME      1 to continue from the checkpoint in MODEL_REPO (default 0)

NOTE: this is a starting skeleton — it has not been run on a GPU as-is. Read it,
adjust the config to your data size, and do a cheap `--flavor t4-small` smoke run
before committing to an expensive flavor.
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from huggingface_hub import snapshot_download, upload_folder

DATA_REPO = os.environ["DATA_REPO"]
MODEL_REPO = os.environ["MODEL_REPO"]
PROFILE = os.environ.get("PROFILE", "smoke")
RESUME = os.environ.get("RESUME", "0") == "1"
NANO = "nanoGPT-master"

# ------------------------------------------------------------------ 1. nanoGPT
print("== fetching nanoGPT", flush=True)
if not os.path.isdir(NANO):
    data = urllib.request.urlopen("https://github.com/karpathy/nanoGPT/archive/refs/heads/master.zip").read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(".")

# ------------------------------------------------------------------ 2. data
print(f"== downloading data from {DATA_REPO}", flush=True)
data_dir = os.path.join(NANO, "data", "chess")
os.makedirs(data_dir, exist_ok=True)
snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=data_dir,
                  allow_patterns=["*.bin", "meta.pkl"])
assert os.path.exists(os.path.join(data_dir, "train.bin")), "train.bin not found in DATA_REPO"

# ------------------------------------------------------------------ 3. resume
init_from = "scratch"
if RESUME:
    print(f"== resuming from {MODEL_REPO}", flush=True)
    out_dir = os.path.join(NANO, "out")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPO, repo_type="model", local_dir=out_dir, allow_patterns=["ckpt.pt"])
    if os.path.exists(os.path.join(out_dir, "ckpt.pt")):
        init_from = "resume"

# ------------------------------------------------------------------ 4. config + train
# Two profiles from TRAINING.md: a cheap pipeline-validation net, then the real one.
# vocab_size is read automatically by nanoGPT from data/chess/meta.pkl (5,267).
cfg = dict(
    smoke=dict(n_layer=4, n_head=4, n_embd=128, block_size=512, batch_size=32,
               max_iters=2000, lr_decay_iters=2000, eval_interval=250, warmup_iters=100),
    full=dict(n_layer=8, n_head=8, n_embd=512, block_size=1024, batch_size=64,
              max_iters=100_000, lr_decay_iters=100_000, eval_interval=1000, warmup_iters=1000),
)[PROFILE]
if "MAX_ITERS" in os.environ:
    cfg["max_iters"] = cfg["lr_decay_iters"] = int(os.environ["MAX_ITERS"])

config = f"""
dataset = 'chess'
out_dir = 'out'
init_from = '{init_from}'
eval_iters = 100
log_interval = 50
always_save_checkpoint = True   # push whatever we have; storage is ephemeral
gradient_accumulation_steps = 1
learning_rate = 1e-3
min_lr = 1e-4
dropout = 0.0
device = 'cuda'
dtype = 'bfloat16'
compile = True
n_layer = {cfg['n_layer']}
n_head = {cfg['n_head']}
n_embd = {cfg['n_embd']}
block_size = {cfg['block_size']}
batch_size = {cfg['batch_size']}
max_iters = {cfg['max_iters']}
lr_decay_iters = {cfg['lr_decay_iters']}
eval_interval = {cfg['eval_interval']}
warmup_iters = {cfg['warmup_iters']}
"""
with open(os.path.join(NANO, "config", "train_chess.py"), "w") as f:
    f.write(config)
print("== config\n" + config, flush=True)

print("== training", flush=True)
subprocess.run([sys.executable, "train.py", "config/train_chess.py"], cwd=NANO, check=True)

# ------------------------------------------------------------------ 5. push checkpoint
print(f"== uploading checkpoint to {MODEL_REPO}", flush=True)
out_dir = os.path.join(NANO, "out")
shutil.copyfile(os.path.join(data_dir, "meta.pkl"), os.path.join(out_dir, "meta.pkl"))  # bundle for sampling
upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=out_dir,
              allow_patterns=["ckpt.pt", "meta.pkl"], commit_message=f"checkpoint ({PROFILE})")
print("== done", flush=True)
