"""Unit tests, including a verbatim fixture from the real HF dataset
(Lichess/standard-chess-games row ~6,500,000,006, 2021-06-06)."""
import tokeniser as tk

REAL_MOVETEXT = (
    "1. d4 { [%eval 0.0] [%clk 0:03:00] } 1... d5 { [%eval 0.27] [%clk 0:03:00] } "
    "2. Bf4 { [%eval 0.0] [%clk 0:03:02] } 2... Bf5 { [%eval 0.4] [%clk 0:02:59] } "
    "3. e3 { [%eval 0.13] [%clk 0:03:03] } 3... e6 { [%eval 0.2] [%clk 0:03:00] } "
    "4. Bd3 { [%eval 0.0] [%clk 0:03:04] } 4... Bxd3 { [%eval 0.1] [%clk 0:03:00] } "
    "5. Qxd3 { [%eval 0.0] [%clk 0:03:06] } 5... Bd6 { [%eval 0.26] [%clk 0:03:01] } "
    "6. Bg3 { [%eval 0.0] [%clk 0:03:07] } 6... Nf6 { [%eval 0.09] [%clk 0:03:00] } "
    "7. Nf3 { [%eval 0.0] [%clk 0:03:05] } 7... Nbd7 { [%eval 0.07] [%clk 0:02:57] } "
    "8. Nbd2 { [%eval 0.13] [%clk 0:03:06] } 8... e5?? { [%eval 5.94] [%clk 0:02:58] } "
    "9. dxe5 { [%eval 5.98] [%clk 0:03:05] } 9... Nxe5 { [%eval 6.1] [%clk 0:02:59] } "
    "10. Bxe5 { [%eval 5.91] [%clk 0:03:04] } 10... Bxe5 { [%eval 5.91] [%clk 0:02:58] } "
    "11. Nxe5 { [%eval 5.54] [%clk 0:03:05] } 11... Qd6 { [%eval 6.3] [%clk 0:02:57] } "
    "12. Nef3 { [%eval 6.06] [%clk 0:02:47] } 12... O-O-O { [%eval 6.02] [%clk 0:02:52] } "
    "13. O-O { [%eval 6.09] [%clk 0:02:48] } 13... c5 { [%eval 6.52] [%clk 0:02:48] } "
    "14. c4 { [%eval 6.4] [%clk 0:02:47] } 14... d4 { [%eval 7.51] [%clk 0:02:47] } "
    "15. exd4 { [%eval 7.83] [%clk 0:02:45] } 15... cxd4 { [%eval 7.59] [%clk 0:02:48] } "
    "16. Qf5+ { [%eval 6.79] [%clk 0:02:14] } 16... Kb8 { [%eval 6.62] [%clk 0:02:44] } "
    "17. Qe5 { [%eval 6.13] [%clk 0:02:07] } 17... Qxe5 { [%eval 6.21] [%clk 0:02:32] } "
    "18. Nxe5 { [%eval 6.12] [%clk 0:02:08] } 18... Rhf8 { [%eval 6.09] [%clk 0:02:25] } "
    "19. Rfd1 { [%eval 6.11] [%clk 0:02:01] } 1-0"
)

NO_EVAL_MOVETEXT = (
    "1. e4 e6 2. d4 b6 3. a3 Bb7 4. Nc3 Nh6 5. Bxh6 gxh6 6. Be2 Qg5 7. Bg4 h5 "
    "8. Nf3 Qg6 9. Nh4 Qg5 10. Bxh5 Qxh4 11. Qf3 Kd8 12. Qxf7 Nc6 13. Qe8# 1-0"
)


def test_real_game():
    g = tk.parse_movetext(REAL_MOVETEXT)
    assert g.has_evals and g.result == "1-0" and len(g.plies) == 37
    assert g.plies[15].san == "e5" and g.plies[15].glyph == "??" and g.plies[15].cp == 594
    tokens, cps, dis = tk.tokenise_game(g, accuracy_source="glyph")
    s = " ".join(tokens)
    assert "[??] [e5]" in s                       # inline blunder glyph kept
    assert "[x] [Be5]" in s and "[O-O-O]" in s
    assert "[+] [Qf5]" in s
    # computed label agrees with Lichess's inline glyph on the blunder
    assert tk.classify(13, 594, False) == "??"
    assert dis == 0, f"unexpected disagreements: {dis}"
    assert len(cps) == 37 and cps[0] == 0 and cps[1] == 27


def test_no_eval_game():
    g = tk.parse_movetext(NO_EVAL_MOVETEXT)
    assert not g.has_evals and len(g.plies) == 25
    tokens, cps, _ = tk.tokenise_game(g, accuracy_source="glyph")
    assert " ".join(tokens).endswith("[#] [Qe8]")
    assert all(c == tk.MATE_CP + 1 for c in cps)  # no evals -> sentinel


def test_edge_sans():
    cases = {
        ("exf8=Q+", "??"): "[??] [+] [x] [=Q] [ef8]",
        ("Nc6", None): "[Nc6]",
        ("Qa1xb2", None): "[x] [Qa1b2]",
        ("e8=N", None): "[=N] [e8]",
        ("O-O-O+", "?"): "[?] [+] [O-O-O]",
    }
    for (san, acc), want in cases.items():
        assert " ".join(tk.tokenise_san(san, acc)) == want


def test_mate_evals():
    assert tk.eval_to_cp("#4") == 29996
    assert tk.eval_to_cp("#-3") == -29997
    assert tk.classify(50, -29997, True) == "??"   # walked into mate
    tk.win_pct(tk.eval_to_cp("#1"))                # no overflow


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
