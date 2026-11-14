"""Annotated diagrams for circuits.md, built on real puzzles and responses. Predicted outputs are computed here from the
puzzle, under each hypothesis; nothing in these figures is a measurement.

uv run --no-project --with matplotlib python circuit_diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "text.hinting": "none"})  # unhinted, so measured text widths match the rendering

PROMPT, TEXT, EDIT, READ, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#7a4fc4", "0.45"

# swaps-7b-tags-long, step 200, held-out 4-swap puzzle (answered correctly).
START = "A"
SWAPS = [("D", "E"), ("E", "A"), ("E", "B"), ("B", "D")]
# swaps-7b-tags-long, step 25, training 8-swap puzzle: after writing swap 5 (A, B) the response jumps to swap 8.
LOOKUP_START = "D"
LOOKUP_SWAPS = [("B", "F"), ("E", "D"), ("E", "A"), ("E", "C"), ("A", "B"), ("E", "B"), ("A", "B"), ("F", "B")]


def move(key: str, pair: tuple[str, str]) -> str:
    a, b = pair
    return b if key == a else a if key == b else key


def track(start: str, swaps: list[tuple[str, str]]) -> list[str]:
    out = [start]
    for pair in swaps:
        out.append(move(out[-1], pair))
    return out


class Canvas:
    """A figure whose data coordinates are inches, with helpers for runs of highlighted text."""

    def __init__(self, width: float, height: float):
        self.fig = plt.figure(figsize=(width, height), dpi=150)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.height = height

    def text(self, x, y, s, **kw):
        return self.ax.text(x, self.height - y, s, **kw)

    def width(self, text: str, size: float) -> float:
        """Advance width of text, trailing spaces included (measured against a sentinel character)."""
        def measure(s):
            t = self.ax.text(0, 0, s, fontsize=size)
            w = t.get_window_extent(self.renderer).transformed(self.ax.transData.inverted()).width
            t.remove()
            return w
        return measure(text + "|") - measure("|")

    def line(self, x: float, y: float, parts: list, size: float = 11) -> list:
        """Write parts left to right; a part is text, or (text, colour) to highlight it, or (text, colour, "box") for a
        readout. Positions come from the width of everything before each part, so pieces join seamlessly. Returns each
        part's (x0, x1, y_top, y_bottom) in the same top-down coordinates."""
        parts = [(p, None, None) if isinstance(p, str) else (*p, None)[:3] for p in parts]
        spans = []
        for text, color, style in parts:
            x += 0.04 if style == "box" else 0.0
            t = self.text(x, y, text, fontsize=size, va="baseline", ha="left", zorder=3, color=color if style == "box" else "0.1")
            x1 = t.get_window_extent(self.renderer).transformed(self.ax.transData.inverted()).x1
            lead = len(text) - len(text.lstrip())
            x0 = x + (self.width(text[:lead], size) if lead else 0.0)
            top, bottom = y - size / 72 * 0.75, y + size / 72 * 0.25
            if color and style == "box":
                self.ax.add_patch(FancyBboxPatch((x0 - 0.012, self.height - bottom - 0.03), x1 - x0 + 0.024, bottom - top + 0.06,
                                                 boxstyle="round,pad=0.01", facecolor="white", edgecolor=color, linewidth=1.6, zorder=2))
            elif color:
                self.ax.add_patch(Rectangle((x0 - 0.015, self.height - bottom - 0.02), x1 - x0 + 0.03, bottom - top + 0.04,
                                            facecolor=color, alpha=0.22, edgecolor="none", zorder=2))
            spans.append((x0, x1, top, bottom))
            x = x1 + (0.04 if style == "box" else 0.0)
        return spans

    def arrow(self, start, end, color, rad=0.0, label=None, label_at=None, lw=1.6, **kw):
        self.ax.annotate("", xy=(end[0], self.height - end[1]), xytext=(start[0], self.height - start[1]),
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2,
                                         connectionstyle=f"arc3,rad={rad}", mutation_scale=14), zorder=4)
        if label:
            self.text(*label_at, label, fontsize=10.5, color=color, ha=kw.get("ha", "left"), va="center", style="italic",
                      multialignment=kw.get("ha", "left"))

    def save(self, name: str):
        Path("figures").mkdir(exist_ok=True)
        self.fig.savefig(f"figures/{name}.png", dpi=150)
        plt.close(self.fig)


def mid(span):
    return (span[0] + span[1]) / 2


def readouts():
    """The real response with the three readout tokens of Experiment 1 and the routes information could take."""
    locations = track(START, SWAPS)
    assert locations == ["A", "A", "E", "B", "D"]
    c = Canvas(14, 6.85)
    c.text(0.3, 0.35, "Three readouts on a real response", fontsize=15, va="top")
    c.text(0.3, 0.72, "OLMo 3 7B, 1,280-token run, step 200, held-out puzzle with 4 swaps (answered D, correct).\nText is verbatim, except "
           "that step 2 is cut at the start and the prompt's closing question is left out.", fontsize=10.5, va="top", color=GREY, linespacing=1.5)

    c.text(0.3, 1.4, "Prompt", fontsize=12, va="baseline", color=PROMPT)
    c.line(1.6, 1.4, ["There are 6 boxes labeled A to F. A key is in box ", ("A", PROMPT), "."])
    p2 = c.line(1.6, 1.85, ["Swap the contents of boxes ", ("D", PROMPT), " and ", ("E", PROMPT), ". Swap the contents of boxes ",
                            ("E", PROMPT), " and ", ("A", PROMPT), "."])
    p3 = c.line(1.6, 2.3, ["Swap the contents of boxes ", ("E", PROMPT), " and ", ("B", PROMPT), ". Swap the contents of boxes ",
                           ("B", PROMPT), " and ", ("D", PROMPT), "."])
    for n, span in [(1, p2[1]), (2, p2[5]), (3, p3[1]), (4, p3[5])]:
        c.text(span[0] - 0.05, span[2] - 0.08, f"swap {n}", fontsize=8.5, color=PROMPT, va="bottom")

    c.text(8.4, 3.0, "Step 2", fontsize=11, va="baseline", color=TEXT)
    c.line(8.4, 3.4, ["… so swapping boxes E and A means the key should move from"])
    s2 = c.line(8.4, 3.8, ["box A to box ", ("E", TEXT), ". So now, the key is no longer in box A but is in box ", ("E", TEXT), "."])

    c.text(0.3, 4.9, "Step 3", fontsize=12, va="baseline", color=READ)
    s3 = c.line(1.6, 4.9, ["The third swap is between boxes ", ("E", READ, "box"), " and B. The key is currently in box ", ("E", READ, "box"),
                           ", so swapping boxes E and B would move the key from box E to box ", ("B", READ, "box"), "."])
    r2, r1, r3 = s3[1], s3[3], s3[5]
    for span, name, what in [(r2, "R2", "next pair"), (r1, "R1", "carry-over"), (r3, "R3", "update")]:
        c.text(mid(span), span[3] + 0.3, name, fontsize=11, color=READ, ha="center", va="top")
        c.text(mid(span), span[3] + 0.55, what, fontsize=9.5, color=READ, ha="center", va="top")

    c.arrow((mid(s2[1]), s2[1][3] + 0.04), (mid(r1) + 0.06, r1[2] - 0.06), TEXT, rad=-0.1)
    c.arrow((mid(s2[3]), s2[3][3] + 0.04), (mid(r1) + 0.12, r1[2] - 0.04), TEXT, rad=0.0,
            label="read off its own text?", label_at=(11.3, 4.3))
    c.arrow((p2[-1][1] + 0.08, (p2[-1][2] + p2[-1][3]) / 2), (mid(r1) - 0.06, r1[2] - 0.06), PROMPT, rad=-0.3,
            label="or track the key through\nswaps 1–2 in the prompt?", label_at=(p2[-1][1] - 0.3, 3.75), ha="right")
    c.arrow((mid(p3[1]), p3[1][3] + 0.04), (mid(r2), r2[2] - 0.06), PROMPT, rad=0.15,
            label="find swap 3\nin the prompt", label_at=(mid(p3[1]) - 1.85, 3.75))

    c.text(0.3, 6.0, "Each readout is the model's prediction of the boxed token, given the prompt and the response up to just before it. "
           "Clean answers: R2 = E, R1 = E, R3 = B.", fontsize=10.5, va="top", color=GREY)
    c.text(0.3, 6.35, "R1 asks where the previous location comes from; R2 asks how the next swap is found; R3 asks how the new location is computed.",
           fontsize=10.5, va="top", color=GREY)
    c.save("circuits-readouts")


def counterfactuals():
    """Edited inputs for each readout, with the answer predicted under each hypothesis."""
    clean_prompt = track(START, SWAPS)[2]  # where the key is after swap 2: what R1 should say
    pair3 = SWAPS[2]

    def r1(text_says, prompt_swaps):
        return text_says, track(START, prompt_swaps)[2]

    def r3(text_says, prompt_swaps):
        return move(text_says, pair3), track(START, prompt_swaps)[3]

    edited_2 = [SWAPS[0], ("C", "A"), *SWAPS[2:]]
    relabel = {"E": "C", "C": "E"}
    relabelled = [tuple(relabel.get(b, b) for b in pair) for pair in SWAPS]
    rows = [
        ("R1", "Clean", ["no change"], r1("E", SWAPS), "yes"),
        ("R1", "Edit the text", ["step 2 says \"…to box ", ("C", EDIT), "… is in box ", ("C", EDIT), "\" (was E)"], r1("C", SWAPS), "no, on purpose"),
        ("R1", "Edit the prompt", ["swap 2: \"boxes ", ("E", EDIT), " and A\" → \"boxes ", ("C", EDIT), " and A\""], r1("E", edited_2), "no, on purpose"),
        ("R1", "Relabel", ["swap the names E and C everywhere"], (relabel["E"], track(START, relabelled)[2]), "yes"),
        ("R3", "Clean", ["no change"], r3("E", SWAPS), "yes"),
        ("R3", "Edit the text, other box", ["every stated location from step 2 on: ", ("B", EDIT), " (was E)"], r3("B", SWAPS), "no, on purpose"),
        ("R3", "Edit the text, other box", ["every stated location from step 2 on: ", ("C", EDIT), " (was E; off template)"], r3("C", SWAPS), "no, on purpose"),
        ("R2", "Edit the prompt at swap 3", ["swap 3: \"boxes ", ("E", EDIT), " and B\" → \"boxes ", ("F", EDIT), " and B\""], ("–", "F"), "yes"),
    ]
    assert rows[0][3] == ("E", clean_prompt) and rows[1][3] == ("C", "E") and rows[2][3] == ("E", "C")
    assert rows[4][3] == ("B", "B") and rows[5][3] == ("E", "B") and rows[6][3] == ("C", "B")
    rows[6] = ("R3", "Edit the text, outside the pair", *rows[6][2:])

    c = Canvas(14, 1.6 + 0.48 * len(rows) + 1.2)
    c.text(0.3, 0.35, "Counterfactual inputs, with the answer each hypothesis predicts", fontsize=15, va="top")
    c.text(0.3, 0.72, "Same response as above. Edits change one letter, a single token, so every position keeps its meaning. "
           "Predictions are computed from the puzzle, not measured.", fontsize=10.5, va="top", color=GREY)
    cols = [0.3, 1.0, 3.55, 9.0, 10.6, 12.2]
    heads = ["", "Input", "What changes", "If it reads\nits text", "If it tracks\nthe prompt", "Consistent?"]
    for x, h in zip(cols, heads):
        c.text(x, 1.45, h, fontsize=11, va="bottom", color="0.2")
    c.ax.plot([0.3, 13.7], [c.height - 1.52] * 2, color="0.75", lw=0.8)
    y = 1.95
    for i, (readout, name, change, (text_pred, prompt_pred), consistent) in enumerate(rows):
        if i and rows[i - 1][0] != readout:
            c.ax.plot([0.3, 13.7], [c.height - (y - 0.33)] * 2, color="0.88", lw=0.8)
        c.text(cols[0], y, readout, fontsize=11, color=READ, va="baseline")
        c.text(cols[1], y, name, fontsize=11, va="baseline")
        c.line(cols[2], y, change, size=10.5)
        differ = text_pred != prompt_pred and "–" not in (text_pred, prompt_pred)
        for x, pred in [(cols[3], text_pred), (cols[4], prompt_pred)]:
            c.text(x + 0.35, y, pred, fontsize=12, va="baseline", ha="center", color=EDIT if differ else "0.1",
                   fontweight="bold" if differ else "normal")
        c.text(cols[5], y, consistent, fontsize=10.5, va="baseline", color=GREY)
        y += 0.48
    c.text(0.3, y + 0.05, "Orange: the two hypotheses disagree, so the measured logit difference between the two letters tells them apart. "
           "Relabelling changes both predictions together;\nit is the clean counterfactual for activation patching, where the question is which "
           "positions carry the letter.\nFor R2, the response up to the readout says nothing about swap 3, so only the prompt can supply it.", fontsize=10.5, va="top", color=GREY, linespacing=1.5)
    c.save("circuits-counterfactuals")


def patching():
    """Activation patching between the relabelled (source) and clean (base) runs at one position and layer."""
    c = Canvas(14, 8.1)
    c.text(0.3, 0.35, "Activation patching on R1", fontsize=15, va="top")
    c.text(0.3, 0.72, "Run both inputs teacher-forced. Copy one activation from the source run into the base run and see whether R1 flips "
           "from E to C.", fontsize=10.5, va="top", color=GREY)
    tokens = [
        ("swap 1", ["D", "and", "E"], PROMPT), ("swap 2", ["E", "and", "A"], PROMPT), ("swap 3", ["E", "and", "B"], PROMPT),
        ("step 2", ["box", "E", "…", "box", "E"], TEXT), ("step 3", ["…", "currently", "in", "box"], READ),
    ]
    layers = 6
    w, h, gap = 0.5, 0.26, 0.28

    def run(y0, title, swap_letters):
        c.text(0.3, y0 + 0.2, title, fontsize=12, va="center")
        x, positions = 2.0, {}
        letters = iter(swap_letters)
        for group, toks, color in tokens:
            gx = x
            for tok in toks:
                shown = next(letters) if tok in "ABCDEF" and len(tok) == 1 else tok
                is_letter = len(tok) == 1 and tok in "ABCDEF"
                for li in range(layers):
                    c.ax.add_patch(Rectangle((x, c.height - (y0 + 0.45 + li * h) - h), w - 0.06, h - 0.05,
                                             facecolor=color if is_letter else "0.92", alpha=0.25 if is_letter else 1, edgecolor="none"))
                c.text(x + (w - 0.06) / 2, y0 + 0.45 + layers * h + 0.22, shown, fontsize=10 if len(shown) > 5 else 10.5, ha="center", va="center",
                       color=color if is_letter else "0.3", fontweight="bold" if is_letter and shown in "CE" and group != "step 3" else "normal")
                positions.setdefault(group, []).append(x)
                x += w
            c.text((gx + x - 0.06) / 2, y0 + 0.45 + layers * h + 0.52, group, fontsize=9.5, ha="center", va="center", color=color)
            x += gap
        readout_x = x - gap + 0.15
        c.ax.add_patch(FancyBboxPatch((readout_x, c.height - (y0 + 0.45 + layers * h + 0.34)), 0.5, 0.26, boxstyle="round,pad=0.02",
                                      facecolor="white", edgecolor=READ, linewidth=1.6))
        c.text(readout_x + 0.25, y0 + 0.45 + layers * h + 0.21, "?", fontsize=11, ha="center", va="center", color=READ)
        return positions, readout_x

    # Source: E and C swapped everywhere; base: clean.
    src, _ = run(1.2, "Source\n(E and C\nrelabelled)", ["D", "C", "C", "A", "C", "B", "C", "C"])
    base, readout_x = run(4.3, "Base\n(clean)", ["D", "E", "E", "A", "E", "B", "E", "E"])
    for li in range(layers):
        c.text(1.9, 1.2 + 0.45 + li * h + h / 2, f"{['31', '…', '16', '…', '1', '0'][li]}", fontsize=8.5, ha="right", va="center", color=GREY)
        c.text(1.9, 4.3 + 0.45 + li * h + h / 2, f"{['31', '…', '16', '…', '1', '0'][li]}", fontsize=8.5, ha="right", va="center", color=GREY)
    c.text(1.45, 1.2 + 0.45 + 2.5 * h, "layer", fontsize=9, ha="right", va="center", color=GREY, rotation=90)
    # Patch step 2's second "E" at a middle layer.
    px = src["step 2"][4] + (w - 0.06) / 2
    py_src, py_base = 1.2 + 0.45 + 2 * h + h / 2, 4.3 + 0.45 + 2 * h + h / 2
    for y in (py_src, py_base):
        c.ax.add_patch(Rectangle((px - (w - 0.06) / 2 - 0.02, c.height - y - h / 2 - 0.02), w - 0.02, h + 0.02, facecolor="none",
                                 edgecolor=EDIT, linewidth=2))
    c.arrow((px + 0.3, py_src + 0.1), (px + 0.3, py_base - 0.1), EDIT, rad=-0.3, lw=2,
            label="copy this activation into the\nbase run (same position, layer)",
            label_at=(px + 0.85, 4.0))
    c.text(readout_x + 0.25, 4.3 + 0.45 + layers * h + 0.5, "R1: measure\nlogit(C) − logit(E)", fontsize=10, ha="center", va="top", color=READ)

    c.text(0.3, 7.25, "Sweep every position and layer. Score = (patched − base) / (source − base) of the logit difference: 0 is no effect, "
           "1 is a full flip to C.\nIf restoring step 2's location tokens is enough, R1 is read off the text; if only the prompt's swap tokens work, "
           "the model tracks the key internally.", fontsize=10.5, va="top", color=GREY, linespacing=1.5)
    c.save("circuits-patching")


def das():
    """DAS at the MLP output of one layer at the R3 readout position, with two orthogonal subspaces."""
    c = Canvas(14, 7.6)
    c.text(0.3, 0.35, "Distributed alignment search at an MLP output", fontsize=15, va="top")
    c.text(0.3, 0.72, "Find a subspace of the MLP's output at the R3 position whose value acts like a high-level variable:\n"
           "swapping it in from another input changes the output as the causal model predicts.", fontsize=10.5, va="top", color=GREY,
           linespacing=1.5)

    # Causal model.
    c.text(0.3, 1.35, "High-level causal model of one step", fontsize=12, va="top")
    nodes = {"L": (1.25, 2.3, "previous\nlocation  L"), "P": (3.35, 2.3, "swap pair\n(a, b)"),
             "T": (2.3, 3.5, "touches key?\nT = L ∈ {a, b}"), "N": (2.3, 4.75, "new location: the other\nbox if T, else L")}
    for x, y, label in nodes.values():
        c.ax.add_patch(FancyBboxPatch((x - 0.95, c.height - y - 0.33), 1.9, 0.66, boxstyle="round,pad=0.03", facecolor="white",
                                      edgecolor="0.3", linewidth=1.1))
        c.text(x, y, label, fontsize=10, ha="center", va="center", linespacing=1.3)
    for a, b in [("L", "T"), ("P", "T"), ("T", "N")]:
        (x0, y0, _), (x1, y1, _) = nodes[a], nodes[b]
        c.arrow((x0, y0 + 0.36), (x1, y1 - 0.36), "0.3", lw=1.2)
    c.arrow((1.0, 2.66), (1.6, 4.42), "0.3", rad=0.3, lw=1.2)

    # Network side.
    x0 = 4.6
    c.text(x0, 1.35, "In the model (one layer l, MLP output h ∈ ℝ⁴⁰⁹⁶, R3 position)", fontsize=12, va="top")
    rows = [
        ("base", "…currently in box E, so swapping boxes E and B would move the key from box E to box", "B", READ, 1.95, 2.33),
        ("source", "…currently in box D, so swapping boxes C and F would move the key from box D to box", "D", PROMPT, 3.6, 3.22),
    ]
    for name, text, letter, color, text_y, bar_y in rows:
        c.text(x0, text_y, name, fontsize=11, va="baseline", color=color)
        c.line(x0 + 0.85, text_y, [text, (f" {letter}", color, "box")], size=10)
        c.text(x0 + 0.85, bar_y, "MLP out h →  R h  =", fontsize=10.5, va="center", color="0.25")
        bx = x0 + 2.65
        for label, fill, width in [("L", TEXT, 0.9), ("T", EDIT, 0.45), ("rest", "0.9", 3.2)]:
            c.ax.add_patch(Rectangle((bx, c.height - bar_y - 0.13), width - 0.05, 0.26, facecolor=fill, alpha=0.35 if label != "rest" else 1,
                                     edgecolor="none"))
            c.text(bx + (width - 0.05) / 2, bar_y, label, fontsize=9.5, ha="center", va="center")
            bx += width
    c.arrow((x0 + 3.08, 3.22 - 0.15), (x0 + 3.08, 2.33 + 0.15), TEXT, rad=0.0, lw=2,
            label="interchange: replace the base's L coordinates with the source's,\nrotate back with Rᵀ, and let the forward pass continue",
            label_at=(x0 + 3.35, 2.78))

    # Predictions.
    y = 4.35
    c.text(x0, y, "What the causal model predicts at R3 after each interchange", fontsize=12, va="top")
    preds = [
        ("Interchange L from a source with L = D", f"T becomes false, the key stays: {move('D', ('E', 'B'))}", TEXT),
        ("Interchange L from a source with L = B", f"T stays true, the key moves the other way: {move('B', ('E', 'B'))}", TEXT),
        ("Interchange T from a source with T false", "the base key E stays put: E", EDIT),
        ("Interchange both from the source", f"the source's own answer: {move('D', ('C', 'F'))}", "0.2"),
    ]
    for i, (what, out, color) in enumerate(preds):
        c.text(x0 + 0.1, y + 0.45 + 0.36 * i, what, fontsize=10.5, va="top", color=color)
        c.text(x0 + 4.4, y + 0.45 + 0.36 * i, "→  " + out, fontsize=10.5, va="top")
    c.text(0.3, 6.45, "R is a learned orthogonal rotation; the L and T coordinates are disjoint blocks of it, so the two subspaces are orthogonal "
           "by construction. Train R on interchange pairs\nfor L and for T, then report interchange accuracy on held-out pairs: "
           "the share where the patched model's top box is the predicted one.", fontsize=10.5, va="top", color=GREY, linespacing=1.5)
    c.save("circuits-das")


def lookup():
    """The real skip, and the counterfactual that measures how much a later copy of the last pair pulls the lookup."""
    after = track(LOOKUP_START, LOOKUP_SWAPS)
    assert after[-1] == "E"
    c = Canvas(14, 6.6)
    c.text(0.3, 0.35, "How the next swap is found", fontsize=15, va="top")
    c.text(0.3, 0.72, "A real error: OLMo 3 7B, 1,280-token run, step 25, training puzzle with 8 swaps. The response wrote swaps 1–5, "
           f"then swap 8\n(writing its pair in reverse order). It answered F; the key ends in {after[-1]}.", fontsize=10.5, va="top", color=GREY, linespacing=1.5)
    x0, w = 1.6, 1.38
    c.text(0.3, 1.75, "Prompt", fontsize=12, va="center", color=PROMPT)
    xs = []
    for i, (a, b) in enumerate(LOOKUP_SWAPS):
        x = x0 + i * w
        hot = i in (4, 6)
        c.ax.add_patch(FancyBboxPatch((x, c.height - 1.95), w - 0.18, 0.42, boxstyle="round,pad=0.02",
                                      facecolor=PROMPT if hot else "white", alpha=0.22 if hot else 1, edgecolor=PROMPT, linewidth=1))
        c.text(x + (w - 0.18) / 2, 1.74, f"{a} ↔ {b}", fontsize=11.5, ha="center", va="center")
        c.text(x + (w - 0.18) / 2, 1.38, f"swap {i + 1}", fontsize=9, ha="center", va="center", color=PROMPT)
        xs.append(x + (w - 0.18) / 2)

    c.text(0.3, 3.05, "Response", fontsize=12, va="center", color=TEXT)
    written = [0, 1, 2, 3, 4, 7]
    for j, i in enumerate(written):
        x = x0 + i * w
        a, b = LOOKUP_SWAPS[i]
        last = j == len(written) - 1
        c.ax.add_patch(FancyBboxPatch((x, c.height - 3.25), w - 0.18, 0.42, boxstyle="round,pad=0.02",
                                      facecolor=EDIT if last else TEXT, alpha=0.22, edgecolor=EDIT if last else TEXT, linewidth=1))
        c.text(x + (w - 0.18) / 2, 3.04, f"{j + 1}. {b} and {a}" if last else f"{j + 1}. {a} and {b}", fontsize=11, ha="center", va="center")
    c.text((xs[5] + xs[6]) / 2, 3.04, "swaps 6 and 7 skipped", fontsize=10, ha="center", va="center", color=GREY, style="italic")
    top, bottom = 2.82, 1.98
    c.arrow((xs[4] - 0.1, top), (xs[5] - 0.3, bottom), TEXT, rad=-0.2, lw=1.5)
    c.text(xs[4] - 0.15, 2.4, "correct", fontsize=10, color=TEXT, style="italic", ha="right", va="center")
    c.arrow((xs[4] + 0.35, top), (xs[6] - 0.2, bottom), EDIT, rad=-0.2, lw=1.8)
    c.text(xs[5] + 0.45, 2.52, "1", fontsize=11, color=EDIT, ha="center", va="center", fontweight="bold")
    c.arrow((xs[7], bottom), (xs[7], top), EDIT, lw=1.8)
    c.text(xs[7] + 0.15, 2.4, "2", fontsize=11, color=EDIT, ha="left", va="center", fontweight="bold")
    c.line(x0, 3.62, [("correct", TEXT), ": the swap after swap 5 is swap 6.   ", ("1", EDIT), " match the last pair written (A, B) to its later copy, "
                      "swap 7.   ", ("2", EDIT), " copy the swap after that copy, swap 8."], size=10.5)

    c.text(0.3, 4.0, "Counterfactual on a late checkpoint (the prefix written in its own template, read out at the first letter of step 6)",
           fontsize=12, va="top")
    rows = [
        ("With the repeat (as above)", "swap 7 = A ↔ B", "E", "F"),
        ("Without it", "swap 7 = A ↔ D  (one token changed)", "E", "no pull toward F"),
    ]
    c.text(1.0, 4.55, "Input", fontsize=10.5, color="0.2", va="top")
    c.text(4.5, 4.55, "Change", fontsize=10.5, color="0.2", va="top")
    c.text(8.6, 4.55, "Correct next", fontsize=10.5, color="0.2", va="top")
    c.text(10.6, 4.55, "Content lookup pulls toward", fontsize=10.5, color="0.2", va="top")
    for i, (name, change, correct, pull) in enumerate(rows):
        y = 4.95 + 0.38 * i
        c.text(1.0, y, name, fontsize=10.5, va="top")
        c.text(4.5, y, change, fontsize=10.5, va="top")
        c.text(9.1, y, correct, fontsize=11, va="top", ha="center")
        c.text(11.6, y, pull, fontsize=11, va="top", ha="center", color=EDIT)
    c.text(0.3, 5.85, "Measure logit(F) − logit(E) at the readout with and without the repeat, over many puzzles: late checkpoints rarely make the error, "
           "but a\npositive shift shows the pull is still there. Patching then asks whether the shift is carried by the last written pair "
           "(the query) or the step number.", fontsize=10.5, va="top", color=GREY, linespacing=1.5)
    c.save("circuits-lookup")


if __name__ == "__main__":
    readouts()
    counterfactuals()
    patching()
    das()
    lookup()
