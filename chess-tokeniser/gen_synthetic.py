"""Generate a realistic synthetic corpus in the HF schema (for offline
validation/benchmarking of the pipeline where huggingface.co is unreachable).
Movetext mimics the real dump format: SAN + inline glyphs + [%eval]/[%clk]."""
import random

import pyarrow as pa
import pyarrow.parquet as pq

import tokeniser as tk

random.seed(7)

FILES = "abcdefgh"
PIECES = "KQRBN"


def rand_san():
    r = random.random()
    if r < 0.35:  # pawn push
        return random.choice(FILES) + str(random.randint(2, 7))
    if r < 0.45:  # pawn capture
        f = random.randint(0, 7)
        g = max(0, min(7, f + random.choice((-1, 1))))
        return f"{FILES[f]}x{FILES[g]}{random.randint(2, 7)}"
    if r < 0.50:  # castle
        return random.choice(("O-O", "O-O-O"))
    if r < 0.52:  # promotion
        f = random.choice(FILES)
        cap = random.random() < 0.4
        tgt = random.choice(FILES) if cap else f
        return f"{f}x{tgt}8=Q" if cap else f"{f}8={random.choice('QRN')}"
    p = random.choice(PIECES)
    dis = random.choice(("", "", "", random.choice(FILES), str(random.randint(1, 8))))
    x = "x" if random.random() < 0.3 else ""
    chk = "+" if random.random() < 0.06 else ""
    return f"{p}{dis}{x}{random.choice(FILES)}{random.randint(1, 8)}{chk}"


def gen_game():
    n = max(20, int(random.gauss(70, 25)))
    cp, cp_prev = tk.START_CP, tk.START_CP
    parts = []
    for i in range(n):
        white = i % 2 == 0
        cp_prev = cp
        swing = random.gauss(0, 35)
        if random.random() < 0.045:  # occasional big error
            swing = random.choice((-1, 1)) * random.uniform(150, 600)
        cp = int(max(-1500, min(1500, cp + swing)))
        glyph = tk.classify(cp_prev, cp, white) or ""
        num = f"{i // 2 + 1}." if white else f"{i // 2 + 1}..."
        clk = f"[%clk 0:{random.randint(0, 2)}:{random.randint(0, 59):02d}]"
        ev = f"{cp / 100:.2f}" if random.random() > 0.001 else f"#{random.randint(1, 9)}"
        parts.append(f"{num} {rand_san()}{glyph} {{ [%eval {ev}] {clk} }}")
    parts.append(random.choice(("1-0", "0-1", "1/2-1/2")))
    return " ".join(parts)


def main(n_games=100_000, out="synthetic_month.parquet"):
    cols = {
        "Site": [f"https://lichess.org/synth{i:08d}" for i in range(n_games)],
        "UTCDate": ["2025.05.15"] * n_games,
        "WhiteElo": [random.randint(1400, 2600) for _ in range(n_games)],
        "BlackElo": [random.randint(1400, 2600) for _ in range(n_games)],
        "TimeControl": [random.choice(("180+0", "300+0", "600+8")) for _ in range(n_games)],
        "Result": ["1-0"] * n_games,
        "Termination": ["Normal"] * n_games,
        "ECO": ["B01"] * n_games,
        "movetext": [gen_game() for _ in range(n_games)],
    }
    # ~6% of games in the real dump have evals; synthetic corpus is all-eval
    # because the pipeline filters on contains('[%eval') server-side anyway.
    pq.write_table(pa.table(cols), out, compression="zstd")
    print(f"wrote {out}: {n_games} games")


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
