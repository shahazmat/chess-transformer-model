"""Local inference server: serve a trained chess-gpt checkpoint to the harness.

Milestone 6's "small local inference server behind predict(ctx)". Downloads
ckpt.pt + meta.pkl from your HF model repo, loads them with the SAME pinned
nanoGPT the training job used, and exposes the model over localhost HTTP.
The browser harness auto-detects it via js/remote-model.js (mock model stays
the fallback when the server is down).

    pip install torch huggingface_hub
    python tools/model_server.py                       # serves shazmate/gpct-trial
    python tools/model_server.py --repo you/chess-gpt --model-elo 1600
    python tools/model_server.py --local ./out         # use a local ckpt dir instead

Then open the harness (node tools/serve.mjs) — the page picks the server up on
load; the console logs which model is live. Restart this server after a new
checkpoint is pushed to pick it up (hf_hub_download re-checks the remote).

Context contract (must mirror engine.js + nerf_batch.py — the bare-history
rule): the model context is

    <bos> <eloW> <eloB> ...historyTokens [nerf token if quality] ...moveTokens

historyTokens arrive BARE from the harness (never contain nerf tokens); the
current move's own nerf conditioning arrives as `quality`. The Elo slot for
the side the model plays comes from --model-elo ('any' -> <elo-any>, or a
number -> its 200-point bucket); the human's slot from the rating typed into
the setup form (missing -> <elo-any>).

API:
    GET  /health   -> {repo, iter_num, best_val_loss, params, model_args}
    POST /predict  -> body {historyTokens, quality, moveTokens, turn,
                            opponentRating}; returns a JSON array of
                     vocab_size probabilities for the next token.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Same commit train_chess_hf.py trains with — architecture code must match.
NANO_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "chess-gpt")

QUALITY_TO_NERF = {"inaccuracy": "<inaccuracy>", "mistake": "<mistake>", "blunder": "<blunder>"}


def elo_bucket_token(elo) -> str:
    """Numeric Elo -> bucket token; mirrors chess-tokeniser/vocab.py."""
    try:
        e = int(elo)
    except (TypeError, ValueError):
        return "<elo-any>"
    if e < 800:
        return "<elo-u800>"
    if e >= 3000:
        return "<elo-3000p>"
    return f"<elo-{(e // 200) * 200}>"


def fetch_nanogpt_model_module():
    """Import the pinned nanoGPT model.py (cached under ~/.cache/chess-gpt)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"nanogpt_model_{NANO_COMMIT[:8]}.py")
    if not os.path.exists(path):
        url = f"https://raw.githubusercontent.com/karpathy/nanoGPT/{NANO_COMMIT}/model.py"
        print(f"== fetching nanoGPT model.py @ {NANO_COMMIT[:8]}")
        with open(path, "wb") as f:
            f.write(urllib.request.urlopen(url).read())
    spec = importlib.util.spec_from_file_location("nanogpt_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(args):
    import torch

    if args.local:
        ckpt_path = os.path.join(args.local, "ckpt.pt")
        meta_path = os.path.join(args.local, "meta.pkl")
    else:
        from huggingface_hub import hf_hub_download

        print(f"== downloading checkpoint from {args.repo}")
        ckpt_path = hf_hub_download(args.repo, "ckpt.pt")
        meta_path = hf_hub_download(args.repo, "meta.pkl")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    nano = fetch_nanogpt_model_module()
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file
    model_args = ckpt["model_args"]
    model = nano.GPT(nano.GPTConfig(**model_args))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    info = {
        "repo": args.local or args.repo,
        "iter_num": int(ckpt.get("iter_num", -1)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "params": sum(p.numel() for p in model.parameters()),
        "model_args": {k: v for k, v in model_args.items()},
        "vocab_size": meta["vocab_size"],
    }
    print(f"== model ready: {info['params']:,} params, iter {info['iter_num']}, "
          f"best val loss {info['best_val_loss']:.4f}")
    return model, meta, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="shazmate/gpct-trial", help="HF model repo with ckpt.pt + meta.pkl")
    ap.add_argument("--local", default=None, help="local dir with ckpt.pt + meta.pkl (skips download)")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--model-elo", default="any",
                    help="Elo the model plays at: a number for its bucket, 'any' for <elo-any>")
    args = ap.parse_args()

    import torch

    model, meta, info = load_model(args)
    stoi, vocab_size = meta["stoi"], meta["vocab_size"]
    block_size = info["model_args"].get("block_size", 512)
    model_elo_token = "<elo-any>" if args.model_elo == "any" else elo_bucket_token(args.model_elo)
    lock = threading.Lock()

    def build_ids(req):
        history = req.get("historyTokens") or []
        move_tokens = req.get("moveTokens") or []
        quality = req.get("quality")
        turn = req.get("turn", "w")
        human_elo_token = elo_bucket_token(req.get("opponentRating"))
        white = model_elo_token if turn == "w" else human_elo_token
        black = model_elo_token if turn == "b" else human_elo_token
        names = ["<bos>", white, black, *history]
        if quality:
            names.append(QUALITY_TO_NERF[quality])
        names.extend(move_tokens)
        unknown = [t for t in names if t not in stoi]
        if unknown:
            raise ValueError(f"tokens not in vocab: {unknown}")
        return [stoi[t] for t in names]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet the default per-request stderr noise
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()

        def do_GET(self):
            if self.path == "/health":
                self._send(200, info)
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "unknown path"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                ids = build_ids(req)[-block_size:]
                t0 = time.time()
                with lock, torch.no_grad():
                    idx = torch.tensor([ids], dtype=torch.long)
                    logits, _ = model(idx)          # (1, 1, vocab) at the last position
                    probs = torch.softmax(logits[0, -1, :vocab_size], dim=-1)
                print(f"predict: {len(ids)} ctx tokens -> {1000 * (time.time() - t0):.0f} ms")
                self._send(200, probs.tolist())
            except Exception as e:  # noqa: BLE001 — surface the reason to the harness
                self._send(400, {"error": str(e)})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"== serving on http://127.0.0.1:{args.port}  (health: /health)")
    server.serve_forever()


if __name__ == "__main__":
    main()
