"""Diagrams for circuits.md, built on real puzzles and responses (sources are given in circuits.md). Predicted outputs are
computed here from the puzzle under each hypothesis; nothing in these figures is a measurement.

uv run --no-project --with matplotlib python circuit_diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "text.hinting": "none"})  # unhinted, so measured text widths match the rendering

PROMPT, TEXT, EDIT, READ, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#7a4fc4", "0.45"
BODY, LABEL, HEAD = 17, 14, 19

# swaps-7b-tags-long, step 200, held-out 4-swap puzzle (answered correctly).
START = "A"
SWAPS = [("D", "E"), ("E", "A"), ("E", "B"), ("B", "D")]
# swaps-7b-tags-v2, step 5, training 8-swap puzzle: after writing swap 5 (E, D) the response jumps to swap 8, which follows
# the later copy of (E, D) at swap 7. The landing pair is unique in the prompt and starts with a different letter from swap 6.
LOOKUP_START = "E"
LOOKUP_SWAPS = [("E", "F"), ("A", "D"), ("D", "F"), ("A", "E"), ("E", "D"), ("B", "A"), ("E", "D"), ("E", "C")]


def move(key: str, pair: tuple[str, str]) -> str:
    a, b = pair
    return b if key == a else a if key == b else key


def track(start: str, swaps: list[tuple[str, str]]) -> list[str]:
    out = [start]
    for pair in swaps:
        out.append(move(out[-1], pair))
    return out


def tint(color, amount: float = 0.18):
    return tuple(1 - amount * (1 - v) for v in to_rgb(color))


class Canvas:
    """A figure in inches with y measured from the top."""

    def __init__(self, width: float, height: float):
        self.fig = plt.figure(figsize=(width, height), dpi=150)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()
        self.height = height

    def text(self, x, y, s, size=BODY, **kw):
        kw.setdefault("va", "center")
        return self.ax.text(x, self.height - y, s, fontsize=size, **kw)

    def width(self, s: str, size: float) -> float:
        t = self.ax.text(0, 0, s, fontsize=size)
        w = t.get_window_extent(self.renderer).transformed(self.ax.transData.inverted()).width
        t.remove()
        return w

    def box(self, x0, y0, x1, y1, face="0.97", edge="0.75", lw=1.2, radius=0.12):
        self.ax.add_patch(FancyBboxPatch((x0, self.height - y1), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={radius}",
                                         facecolor=face, edgecolor=edge, linewidth=lw))

    def chip(self, x, y, s, color, size=BODY, filled=True, width=None, height=0.5, text_color="0.1"):
        """A rounded token chip centred at (x, y). Returns (x0, x1, y0, y1)."""
        w = width or self.width(s, size) + 0.32
        self.box(x - w / 2, y - height / 2, x + w / 2, y + height / 2, face=tint(color) if filled else "white", edge=color, lw=1.8, radius=0.1)
        self.text(x, y, s, size, ha="center", color=text_color)
        return x - w / 2, x + w / 2, y - height / 2, y + height / 2

    def arrow(self, start, end, color, rad=0.0, lw=2.2):
        self.ax.annotate("", xy=(end[0], self.height - end[1]), xytext=(start[0], self.height - start[1]),
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                         connectionstyle=f"arc3,rad={rad}", mutation_scale=20), zorder=4)

    def note(self, x, y, s, color, ha="left"):
        self.text(x, y, s, LABEL, ha=ha, color=color, style="italic", linespacing=1.25)

    def save(self, name: str):
        Path("figures").mkdir(exist_ok=True)
        self.fig.savefig(f"figures/{name}.png", dpi=150)
        plt.close(self.fig)


def readouts():
    """Where does the location at R1 come from: the model's own step 2, or the prompt? R3 is the update."""
    assert track(START, SWAPS) == ["A", "A", "E", "B", "D"]
    c = Canvas(11.9, 5.0)
    c.text(0.35, 0.4, "Response", HEAD)
    c.text(8.3, 0.4, "Prompt", HEAD)
    c.box(0.3, 0.75, 6.75, 4.75)
    c.box(8.25, 0.75, 11.6, 4.75)

    lines = [("Key in box A", None), ("Swap D and E", PROMPT), ("Swap E and A", PROMPT), ("Swap E and B", None), ("Swap B and D", None)]
    rows = [1.3, 2.0, 2.65, 3.3, 3.95]
    for (s, color), y in zip(lines, rows):
        if color:
            c.chip(9.92, y, s, color, width=2.6, height=0.52)
        else:
            c.text(9.92, y, s, ha="center")

    right, cx = 5.15, 5.6
    c.text(0.55, 1.05, "step 2", LABEL, color=GREY)
    c.text(right, 1.55, "… so now the key is in box", ha="right")
    c.chip(cx, 1.55, "E", TEXT, width=0.52)
    c.text(0.55, 2.45, "step 3", LABEL, color=GREY)
    c.text(right, 2.95, "The key is currently in box", ha="right")
    r1 = c.chip(cx, 2.95, "?", READ, filled=False, width=0.52, text_color=READ)
    c.text(cx, 3.37, "R1", LABEL, ha="center", color=READ)
    c.text(right, 4.0, "… move the key from box E to box", ha="right")
    c.chip(cx, 4.0, "?", READ, filled=False, width=0.52, text_color=READ)
    c.text(cx, 4.42, "R3", LABEL, ha="center", color=READ)

    c.arrow((cx, 1.8), (cx, r1[2]), TEXT)
    c.note(cx - 0.25, 2.2, "read its\nown text?", TEXT, ha="right")
    c.arrow((8.6, 2.33), (r1[1] + 0.02, 2.95), PROMPT, rad=0.15)
    c.note(7.5, 3.3, "or track\nthe prompt?", PROMPT, ha="center")
    c.save("circuits-readouts")


def counterfactuals():
    """One-letter edits for R1 and R3, and the answer each hypothesis predicts."""
    pair3 = SWAPS[2]
    edited = [SWAPS[0], ("C", "A"), *SWAPS[2:]]
    rows = [
        ("R1", "clean", [], ("E", track(START, SWAPS)[2])),
        ("R1", "edit step 2", [("step 2: in box ", None), ("C", EDIT)], ("C", track(START, SWAPS)[2])),
        ("R1", "edit the prompt", [("swap 2: ", None), ("C", EDIT), (" and A", None)], ("E", track(START, edited)[2])),
        ("R3", "edit step 2", [("step 2: in box ", None), ("B", EDIT)], (move("B", pair3), track(START, SWAPS)[3])),
    ]
    assert [r[3] for r in rows] == [("E", "E"), ("C", "E"), ("E", "C"), ("E", "B")]
    c = Canvas(11.6, 4.6)
    cols = [0.45, 1.35, 4.0, 8.4, 10.3]
    for x, h in zip(cols[1:], ["Input", "Edit", "Reads\nits text", "Tracks\nthe prompt"]):
        c.text(x, 0.6, h, HEAD, ha="left" if x < 8 else "center", linespacing=1.15)
    c.ax.plot([0.35, 11.3], [c.height - 1.15] * 2, color="0.7", lw=1)
    for i, (readout, name, edit, (text_pred, prompt_pred)) in enumerate(rows):
        y = 1.7 + 0.75 * i
        c.text(cols[0], y, readout, color=READ)
        c.text(cols[1], y, name)
        x = cols[2]
        if not edit:
            c.text(x, y, "–", color=GREY)
        for s, color in edit:
            if color:
                b = c.chip(x + 0.26, y, s, color, width=0.52)
                x = b[1] + 0.02
            else:
                c.text(x, y, s)
                x += c.width(s, BODY)
        differ = text_pred != prompt_pred
        for col, pred in [(cols[3], text_pred), (cols[4], prompt_pred)]:
            c.text(col, y, pred, 21, ha="center", color=EDIT if differ else "0.1", fontweight="bold" if differ else "normal")
        if i == 2:
            c.ax.plot([0.35, 11.3], [c.height - (y + 0.38)] * 2, color="0.88", lw=1)
    c.save("circuits-counterfactuals")


def patching():
    """Copy one activation (one position, one layer) from the relabelled run into the clean run, and read out R1."""
    c = Canvas(11.6, 5.6)
    groups = [("swap 2", ["E", "and", "A"], PROMPT), ("step 2", ["box", "E"], TEXT), ("step 3", ["currently", "in", "box"], READ)]
    layers, cell_h, gap = 4, 0.2, 0.1

    def run(y, label, swap):
        c.text(0.35, y, label, HEAD)
        x, centres = 2.1, {}
        for name, toks, color in groups:
            gx = x
            for t, tok in enumerate(toks):
                shown = swap.get(tok, tok) if len(tok) == 1 else tok
                w = max(c.width(shown, BODY) + 0.32, 0.6)
                letter = len(tok) == 1 and tok in "ABCDEF"
                c.chip(x + w / 2, y, shown, color if letter else "0.6", filled=letter, width=w)
                for li in range(layers):
                    top = y - 0.45 - (li + 1) * (cell_h + 0.06)
                    c.box(x + 0.08, top, x + w - 0.08, top + cell_h, face=tint(color, 0.35) if letter else "0.9", edge="none", radius=0.03)
                centres[(name, t)] = (x + w / 2, w)
                x += w + gap
            c.text((gx + x - gap) / 2, y + 0.5, name, LABEL, ha="center", color=color)
            x += 0.35
        readout = c.chip(x + 0.3, y, "?", READ, filled=False, width=0.55, text_color=READ)
        return centres, readout

    src, _ = run(1.75, "source", {"E": "C"})
    _, readout = run(4.55, "base", {})
    sx, w = src[("step 2", 1)]

    def layer_y(y):
        return y - 0.45 - 2 * (cell_h + 0.06) - cell_h / 2

    for y in (1.75, 4.55):
        ly = layer_y(y)
        c.box(sx - w / 2 + 0.04, ly - cell_h / 2 - 0.05, sx + w / 2 - 0.04, ly + cell_h / 2 + 0.05, face="none", edge=EDIT, lw=2.4, radius=0.04)
    gap_x = sx + w / 2 + 0.17
    c.ax.plot([sx + w / 2 - 0.04, gap_x, gap_x], [c.height - layer_y(1.75)] * 2 + [c.height - layer_y(4.55)], color=EDIT, lw=2.2,
              solid_joinstyle="round")
    c.arrow((gap_x + 0.01, layer_y(4.55)), (sx + w / 2 - 0.04, layer_y(4.55)), EDIT)
    c.note(gap_x + 0.15, 2.85, "copy one activation", EDIT)
    c.text((readout[0] + readout[1]) / 2, 5.05, "C or E?", LABEL, ha="center", color=READ)
    c.text(1.85, layer_y(1.75), "layers", LABEL, ha="right", color=GREY)
    c.save("circuits-patching")


def das():
    """DAS: rotate the MLP output, swap the block that stands for L in from another input, and check the output."""
    assert move("D", ("E", "B")) == "D"
    c = Canvas(11.6, 5.4)
    c.text(0.35, 0.4, "Causal model", HEAD)
    c.text(4.6, 0.4, "In the model, at the readout", HEAD)
    nodes = {"L": (1.0, 1.45, "L\nprevious box"), "P": (3.0, 1.45, "pair\n(a, b)"), "T": (2.0, 2.95, "T\ntouches key?"),
             "N": (2.0, 4.45, "new box")}
    for x, y, s in nodes.values():
        c.box(x - 0.8, y - 0.42, x + 0.8, y + 0.42, face="white", edge="0.35", lw=1.4)
        c.text(x, y, s, LABEL + 1, ha="center", linespacing=1.3)
    for a, b in [("L", "T"), ("P", "T"), ("T", "N")]:
        (x0, y0, _), (x1, y1, _) = nodes[a], nodes[b]
        c.arrow((x0 + (x1 - x0) * 0.12, y0 + 0.44), (x1 - (x1 - x0) * 0.12, y1 - 0.44), "0.35", lw=1.6)
    c.arrow((0.55, 1.88), (1.35, 4.05), "0.35", rad=0.3, lw=1.6)

    for name, sentence, answer, y, color in [("base", "Swap E and B. Key was in E. Now in", "B", 1.3, READ),
                                             ("source", "Swap C and F. Key was in D. Now in", "D", 4.5, PROMPT)]:
        c.text(4.6, y, name, LABEL, color=color)
        c.text(5.55, y, sentence)
        c.chip(5.55 + c.width(sentence, BODY) + 0.35, y, answer, color, width=0.5)
    bars = {}
    for name, y in [("base", 2.2), ("source", 3.6)]:
        c.text(5.45, y, "R·h", ha="right", color="0.3")
        x = 5.6
        for label, color, w in [("L", TEXT, 1.0), ("T", EDIT, 0.6), ("rest", "0.6", 2.9)]:
            c.chip(x + w / 2, y, label, color, filled=True, width=w - 0.06, height=0.44)
            bars[(name, label)] = x + w / 2
            x += w
    c.arrow((bars[("source", "L")], 3.36), (bars[("base", "L")], 2.44), TEXT)
    c.note(bars[("base", "L")] + 0.3, 2.9, "swap in the source's L,\nrotate back, run on", TEXT)
    c.text(10.4, 2.2, "→", ha="center")
    c.chip(10.95, 2.2, "D", TEXT, width=0.55)
    c.text(10.95, 2.68, "predicted", LABEL, ha="center", color=GREY)
    c.save("circuits-das")


def lookup():
    """The real skip: the response matches the last pair written to a later copy and copies what follows it."""
    assert track(LOOKUP_START, LOOKUP_SWAPS)[-1] == "D"
    c = Canvas(11.6, 4.2)
    x0, step, w = 0.35, 1.4, 1.26
    xs = [x0 + i * step + w / 2 for i in range(8)]
    c.text(0.35, 0.4, "Prompt", HEAD, color=PROMPT)
    c.text(0.35, 2.75, "Response", HEAD, color=TEXT)
    for i, (a, b) in enumerate(LOOKUP_SWAPS):
        c.chip(xs[i], 1.35, f"{a} ↔ {b}", PROMPT, filled=i in (4, 6), width=w)
        c.text(xs[i], 0.88, str(i + 1), LABEL, ha="center", color=PROMPT)
    for i in (0, 1, 2, 3, 4, 7):
        a, b = LOOKUP_SWAPS[i]
        c.chip(xs[i], 3.4, f"{a} and {b}", EDIT if i == 7 else TEXT, width=w)
    c.note((xs[5] + xs[6]) / 2, 3.4, "skipped", GREY, ha="center")
    c.arrow((xs[4] - 0.2, 3.15), (xs[5] - 0.3, 1.6), TEXT, rad=-0.15)
    c.note(xs[4] - 0.4, 2.35, "correct", TEXT, ha="right")
    c.arrow((xs[4] + 0.3, 3.15), (xs[6] - 0.25, 1.6), EDIT, rad=-0.2)
    c.note(xs[5] + 0.55, 2.85, "1  match", EDIT)
    c.arrow((xs[7], 1.6), (xs[7], 3.15), EDIT)
    c.note(xs[7] - 0.15, 2.2, "2  copy what\nfollows", EDIT, ha="right")
    c.save("circuits-lookup")


if __name__ == "__main__":
    readouts()
    counterfactuals()
    patching()
    das()
    lookup()
