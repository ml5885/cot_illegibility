"""Diagrams for circuits.md, built on real puzzles and responses (sources are given in circuits.md). Predicted outputs are
computed here from the puzzle under each hypothesis; nothing in these figures is a measurement.

uv run --no-project --with matplotlib python circuit_diagrams.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "text.hinting": "none"})  # unhinted, so measured text widths match the rendering

PROMPT, TEXT, EDIT, READ, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#7a4fc4", "0.45"
BODY, LABEL, HEAD = 17, 14, 19

# swaps-7b-tags-long, step 200, held-out 8-swap puzzle (answered A, correct). The response is verbatim up to the readout in step 4.
START = "B"
SWAPS = [("F", "D"), ("E", "A"), ("B", "F"), ("F", "A"), ("E", "D"), ("C", "E"), ("B", "C"), ("F", "D")]
QUESTION = ("There are 6 boxes labeled A to F. A key is in box B. Swap the contents of boxes F and D. Swap the contents of boxes E and A. "
            "Swap the contents of boxes {s3a} and {s3b}. Swap the contents of boxes F and A. Swap the contents of boxes E and D. "
            "Swap the contents of boxes C and E. Swap the contents of boxes B and C. Swap the contents of boxes F and D. "
            "Which box contains the key? Answer with the letter of the box.")
RESPONSE = ("Let's break down the problem step by step to determine where the key ends up after all the swaps. Initially, the key is in "
            "box B. We need to track the key's movements through each swap operation.\n"
            "1. Swap the contents of boxes F and D: Since the key is currently in box B, this swap doesn't affect the key. "
            "So the key remains in box B.\n"
            "2. Swap the contents of boxes E and A: Again, the key is in box B, so this swap also doesn't affect the key. "
            "The key remains in box B.\n"
            "3. Swap the contents of boxes B and F: The key is currently in box B, so swapping boxes B and F would move the key from "
            "box B to box {to}. So now the key is in box {now}.\n"
            "4. Swap the contents of boxes F and A: The key is currently in box")
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

    def flow(self, x, y, width, parts, size=13, line=0.29, para=1.3, breaks=None, force=()) -> tuple[float, list, set]:
        """Wrap parts (text, or (text, colour) for a highlighted letter, or (text, colour, "box") for a readout) into a column
        starting at (x, y). "\n" starts a new paragraph. With breaks (word indices), wrap exactly there instead, so two
        texts that differ only in single letters wrap alike. Returns the last line's y, each highlighted part's
        (x0, x1, y) in order, and the word indices that started a wrapped line."""
        cx, spans, wrapped, n = x, [], set(), 0
        for part in parts:
            text, color, style = (part, None, None) if isinstance(part, str) else (*part, None)[:3]
            for word in re.findall(r"\n|[ ]*\S+|[ ]+", text):
                if word == "\n":
                    cx, y = x, y + line * para
                    continue
                n += 1
                wrap = n in breaks if breaks is not None else (n in force or cx + self.width(word, size) > x + width) and cx > x
                if wrap:
                    cx, y, word = x, y + line, word.lstrip()
                    wrapped.add(n)
                if cx == x:
                    word = word.lstrip()
                lead = self.width(word[: len(word) - len(word.lstrip())], size) if word.startswith(" ") else 0.0
                w = self.width(word.lstrip(), size)
                x0 = cx + lead
                if color:
                    pad = 0.035
                    face = "white" if style == "box" else tint(color, 0.5 if style == "mark" else 0.35)
                    edge = "none" if style == "mark" else color
                    half = 0.0108 * size
                    self.box(x0 - pad, y - half, x0 + w + pad, y + half, face=face, edge=edge, lw=1.6, radius=0.04)
                    if style != "mark":
                        spans.append((x0, x0 + w, y))
                self.text(x0, y, word.lstrip(), size, ha="left", color=color if style == "box" else "0.1", zorder=5,
                          path_effects=[patheffects.withStroke(linewidth=3, foreground="white")] if not color else None)
                cx = x0 + w
        return y, spans, wrapped

    def note(self, x, y, s, color, ha="left"):
        self.text(x, y, s, LABEL, ha=ha, color=color, style="italic", linespacing=1.25)

    def save(self, name: str):
        Path("figures").mkdir(exist_ok=True)
        self.fig.savefig(f"figures/{name}.png", dpi=150)
        plt.close(self.fig)


def relabel(text: str, a: str, b: str) -> str:
    """Swap two box names wherever they name a box (not in "labeled A to F")."""
    return re.sub(r"\b(box|boxes|and) ([A-F])\b", lambda m: f"{m.group(1)} {({a: b, b: a}).get(m.group(2), m.group(2))}", text)


DIFF = "#eda100"


def marked(text: str) -> list:
    """Split text into parts, highlighting the box letters F and C (the letters the source run swaps)."""
    out, last = [], 0
    for m in re.finditer(r"\b(?:box|boxes|and) ([FC])\b", text):
        out += [text[last:m.start(1)], (m.group(1), DIFF, "mark")]
        last = m.end(1)
    return out + [text[last:]]


def localization():
    """One patch, shown on the full runs: copy the activation at step 3's last statement of the key's box from the source
    run (F and C swapped everywhere) into the base run, and read out the box in step 4."""
    after = track(START, SWAPS)
    assert after[3] == "F" and after[-1] == "A"
    c = Canvas(17.6, 13.6)
    col, gutter, top = 7.9, 1.2, 0.45
    xs = {"source": 0.3, "base": 0.3 + col + gutter}

    def parts(source: bool):
        rel = (lambda t: relabel(t, "F", "C")) if source else (lambda t: t)
        loc = "C" if source else "F"
        prompt = marked(rel(QUESTION.format(s3a="B", s3b="F")))
        before, after_now = RESPONSE.split("{now}")
        response = marked(rel(before.format(to="F"))) + [(loc, EDIT)] + marked(rel(after_now)) + [" ", ("?", READ, "box")]
        return prompt, response

    titles = {"source": ("Source run", "F and C swapped everywhere"), "base": ("Base run", "the real response")}
    breaks, tokens, readouts = {}, {}, {}
    for name in ("base", "source"):
        x = xs[name]
        c.text(x, top, titles[name][0], HEAD)
        c.text(x + c.width(titles[name][0], HEAD) + 0.15, top + 0.02, titles[name][1], LABEL, color=GREY, style="italic")
        prompt, response = parts(name == "source")
        loc = "C" if name == "source" else "F"
        y, _, b1 = c.flow(x, top + 0.7, col, prompt, size=17, line=0.4, breaks=breaks.get("prompt"))
        c.ax.plot([x, x + col], [c.height - (y + 0.32)] * 2, color="0.8", lw=1)
        token = 1 + sum(len(re.findall(r"[ ]*\S+|[ ]+", t if isinstance(t, str) else t[0])) for t in response[:response.index((loc, EDIT))])
        y, spans, b2 = c.flow(x, y + 0.68, col, response, size=17, line=0.4, para=1.45, breaks=breaks.get("response"), force={token})
        breaks.setdefault("prompt", b1)
        breaks.setdefault("response", b2)
        tokens[name], readouts[name] = spans[0], spans[1]

    # A straight arrow from the source token to the same token in the base run, drawn under the text.
    (s0, s1, sy), (b0, b1, by) = tokens["source"], tokens["base"]
    c.ax.annotate("", xy=(b0 - 0.06, c.height - by), xytext=(s1 + 0.12, c.height - sy),
                  arrowprops=dict(arrowstyle="-|>", color=EDIT, lw=2.4, shrinkA=0, shrinkB=0, mutation_scale=20), zorder=4)
    c.text(xs["base"] - gutter / 2, sy - 0.3, "patch", HEAD, ha="center", color=EDIT, style="italic")

    for name, letter in (("source", "C"), ("base", "F")):
        r0, r1, ry = readouts[name]
        c.text(r1 + 0.15, ry, f"← predicted next token: {letter}", BODY, ha="left", color=READ)

    y = readouts["base"][2] + 0.85
    x = xs["base"]
    c.text(x, y, "Base run after the patch, predicted next token:", BODY)
    for j, (letter, color, why) in enumerate([("C", EDIT, "if step 4 reads the box only from step 3's text"),
                                              ("C/F", READ, "in between if it uses the text and other routes"),
                                              ("F", GREY, "if it gets the box another way, e.g. from the prompt")]):
        yy = y + 0.55 + 0.52 * j
        c.chip(x + 0.33, yy, letter, color, width=0.6 if len(letter) > 1 else 0.45, height=0.42)
        c.text(x + 0.72, yy, why, BODY, color=color if color != GREY else "0.3")
    c.text(xs["source"], y, "Highlighted: the letters that differ between the runs.", BODY, color=GREY)
    c.save("circuits-localization")


def counterfactuals():
    """One-letter edits for R1 (\"currently in box ?\" in step 4) and R3 (\"to box ?\"), and the answer each hypothesis predicts."""
    pair4 = SWAPS[3]
    edited = [*SWAPS[:2], ("B", "C"), *SWAPS[3:]]
    rows = [
        ("R1", "clean", [], ("F", track(START, SWAPS)[3])),
        ("R1", "edit step 3", [("step 3: in box ", None), ("C", EDIT)], ("C", track(START, SWAPS)[3])),
        ("R1", "edit the prompt", [("swap 3: B and ", None), ("C", EDIT)], ("F", track(START, edited)[3])),
        ("R3", "edit step 3", [("step 3: in box ", None), ("A", EDIT)], (move("A", pair4), track(START, SWAPS)[4])),
    ]
    assert [r[3] for r in rows] == [("F", "F"), ("C", "F"), ("F", "C"), ("F", "A")]
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
    localization()
    counterfactuals()
    das()
    lookup()
