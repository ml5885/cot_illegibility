"""Interactive completion-length plot with a selectable right-axis measure (share, or mean advantage with and without), written to figures/length-interactive.html."""

import json
import re
import sys
from pathlib import Path

from plots import BIN, MIN_SAMPLES, WINDOW, first_word, group_advantages, load, newline, swap_steps, tracking

LIST = re.compile(r"(?:^|\s)1[.)]\s.*\s2[.)]\s")


def ratio(hit) -> callable:
    return lambda row: (float(bool(hit(row))), 1.0)


def per_swap(count) -> callable:
    def measure(row: dict) -> tuple[float, float]:
        steps = swap_steps(row)
        return float(sum(map(count, steps))), float(len(steps))

    return measure


FEATURES = {
    "a newline": newline,
    "a blank line": lambda row: "\n\n" in row["completion"],
    "a standalone location statement": lambda row: any(standalone for _, standalone in swap_steps(row)),
    "every swap written out": lambda row: "skipped" not in tracking(row),
    "a numbered list": lambda row: LIST.search(" ".join(row["completion"].split())),
    '"Step N:" headers': lambda row: re.search(r"Step \d+:", row["completion"]),
    'the opening "Let\'s"': lambda row: first_word(row) == "Let's",
    'the opening "To"': lambda row: first_word(row) == "To",
}

MEASURES = {
    "Completions with a newline": ratio(newline),
    "Completions with a blank line": ratio(FEATURES["a blank line"]),
    "Correct": ratio(lambda row: row["reward"]),
    "Swaps where the stated key location is right": per_swap(lambda step: step[0] == "right"),
    "Swaps followed by a standalone location statement": per_swap(lambda step: step[1]),
    "Responses that write out every swap": ratio(FEATURES["every swap written out"]),
    "Numbered list": ratio(FEATURES["a numbered list"]),
    '"Step N:" headers': ratio(FEATURES['"Step N:" headers']),
    'Opens with "Let\'s"': ratio(FEATURES['the opening "Let\'s"']),
    'Opens with "To"': ratio(FEATURES['the opening "To"']),
}
ADVANTAGE_WIDTH = 10


def advantage_split(by_step: dict[int, list[dict]], has) -> dict[str, list[float | None]]:
    """Per 10-step bin, mean advantage (and standard error) of responses with and without a feature, in groups with mixed rewards."""
    starts = range(0, max(by_step) + 1, ADVANTAGE_WIDTH)
    scored = {s: group_advantages(rows) for s, rows in by_step.items()}
    out = {"x": [start + ADVANTAGE_WIDTH / 2 for start in starts]}
    for side, key in [(True, "with"), (False, "without")]:
        means, errors = [], []
        for start in starts:
            values = [a for s in range(start, start + ADVANTAGE_WIDTH) for row, a, group in scored.get(s, []) if bool(has(row, group)) == side]
            mean = sum(values) / len(values) if len(values) >= MIN_SAMPLES else None
            means.append(mean)
            errors.append((sum((a - mean) ** 2 for a in values) / (len(values) - 1) / len(values)) ** 0.5 if mean is not None else None)
        out[key], out[f"{key}_error"] = means, errors
    return out


def smoothed(points: dict[int, tuple[float, float]]) -> list[float | None]:
    steps = sorted(points)
    shares = [points[s][0] / points[s][1] if points[s][1] else None for s in steps]
    out = []
    for i in range(len(shares)):
        window = [x for x in shares[max(0, i - WINDOW + 1) : i + 1] if x is not None]
        out.append(sum(window) / len(window) if window else None)
    return out


def summarize(run: str) -> dict:
    rows = load(run)["train"]
    limit = json.loads(Path("runs", run, "args.json").read_text())["max_tokens"]
    half = BIN // 2
    levels = {}
    for level, by_step in sorted(rows.items()):
        steps = sorted(by_step)
        starts = range(0, steps[-1] + 1, half)
        bins = [sorted(r["tokens"] for s in range(start, start + half) for r in by_step.get(s, [])) for start in starts]
        measures = {}
        for name, measure in MEASURES.items():
            points = {}
            for s in steps:
                pairs = [measure(r) for r in by_step[s]]
                points[s] = (sum(p[0] for p in pairs), sum(p[1] for p in pairs))
            measures[name] = smoothed(points)
        splits = {
            "Longer than its group's mean": lambda row, group: row["tokens"] > sum(r["tokens"] for r in group) / len(group),
            "Cut off at the token limit": lambda row, group: row["tokens"] >= limit - 2,
        } | {f"With {name}": (lambda f: lambda row, group: f(row))(f) for name, f in FEATURES.items()}
        levels[level] = {
            "steps": steps,
            "centers": [start + half / 2 for start in starts],
            "low": [b[len(b) // 20] for b in bins],
            "median": [b[len(b) // 2] for b in bins],
            "high": [b[-len(b) // 20] for b in bins],
            "longest": [b[-1] for b in bins],
            "cut": [sum(t >= limit - 2 for t in b) / len(b) for b in bins],
            "measures": measures,
            "advantage": {name: advantage_split(by_step, has) for name, has in splits.items()},
        }
    return {"limit": limit, "levels": levels}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Completion Length Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
  :root { --ink: #1f1f1f; --muted: #666; --line: #d0d0d0; --surface: #ffffff; --accent: #2a78d6; }
  body { margin: 0; padding: 16px; background: var(--surface); color: var(--ink); font-family: "DejaVu Serif", Georgia, serif; }
  h1 { font-weight: normal; font-size: 22px; margin: 0 0 12px; }
  .controls { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 8px; }
  .toggle button { font: inherit; padding: 6px 12px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; }
  .toggle button:first-child { border-radius: 6px 0 0 6px; }
  .toggle button:last-child { border-radius: 0 6px 6px 0; border-left: none; }
  .toggle button.on { background: var(--accent); border-color: var(--accent); color: white; }
  .note { color: var(--muted); font-size: 14px; margin: 4px 0 0; max-width: 900px; }
  select { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); max-width: 100%; }
  label { color: var(--muted); margin-right: 6px; }
  #plot { width: 100%; height: min(860px, 120vw); }
</style>
</head>
<body>
<h1>Completion length and the token limit</h1>
<div class="controls">
  <div class="toggle" id="runs"></div>
  <div class="toggle" id="modes"></div>
  <div><label for="measure">Right axis</label><select id="measure"></select></div>
</div>
<p class="note" id="note"></p>
<div id="plot"></div>
<script>
const DATA = __DATA__;
const runNames = Object.keys(DATA);
const sample = Object.values(Object.values(DATA)[0].levels)[0];
const MODES = {
  "Share of completions": Object.keys(sample.measures).concat(["Cut off at the token limit"]),
  "Mean advantage": Object.keys(sample.advantage),
};
const NOTES = {
  "Share of completions": "Right axis: share of training completions (5-step mean).",
  "Mean advantage": "Right axis: mean GRPO advantage of completions with and without the feature, among groups with mixed rewards, in 10-step bins (error bars: standard error). Bins with fewer than 5 completions on a side are left out.",
};
let run = Object.keys(DATA)[0], mode = "Share of completions", measure = MODES[mode][0];

function toggle(id, names, label, pick) {
  const box = document.getElementById(id);
  for (const name of names) {
    const b = document.createElement("button");
    b.textContent = label(name);
    b.onclick = () => pick(name);
    box.appendChild(b);
  }
  return box;
}
const runs = toggle("runs", runNames, name => `${DATA[name].limit.toLocaleString()}-token limit`, name => { run = name; draw(); });
const modes = toggle("modes", Object.keys(MODES), name => name, name => { mode = name; measure = MODES[mode][0]; fill(); draw(); });
const select = document.getElementById("measure");
function fill() {
  select.innerHTML = "";
  for (const name of MODES[mode]) select.add(new Option(name, name));
  select.value = measure;
}
select.onchange = () => { measure = select.value; draw(); };
fill();

const FONT = { family: '"DejaVu Serif", Georgia, serif', color: "#1f1f1f", size: 13 };
const ORANGE = "#eb6834", BLUE = "#2a78d6", GREEN = "#1baf7a";

function draw() {
  [...runs.children].forEach((b, i) => b.classList.toggle("on", runNames[i] === run));
  [...modes.children].forEach((b, i) => b.classList.toggle("on", Object.keys(MODES)[i] === mode));
  document.getElementById("note").textContent = NOTES[mode];
  const advantage = mode === "Mean advantage";
  const { limit, levels } = DATA[run];
  const traces = [], layout = {
    font: FONT, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", hovermode: "x unified", showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.13, bordercolor: "#000", borderwidth: 1 },
    margin: { l: 70, r: 70, t: 40, b: 110 }, annotations: [],
  };
  const domains = [[0, 0.46], [0.54, 1]], rows = [[0.56, 1], [0, 0.44]];
  Object.entries(levels).forEach(([level, d], i) => {
    const n = i + 1, share = `y${n}`, tokens = `y${n + 4}`, x = `x${n}`;
    const col = domains[i % 2], row = rows[Math.floor(i / 2)];
    layout[`xaxis${n}`] = { domain: col, anchor: share, range: [0, 200], title: i >= 2 ? "Training step" : "", showgrid: false, zeroline: false };
    layout[`yaxis${n}`] = { domain: row, anchor: x, side: "right", showgrid: false, tickfont: { color: ORANGE }, showticklabels: i % 2 === 1,
      ...(advantage
        ? { range: [-1.5, 1.5], tickformat: ".1f", zeroline: true, zerolinecolor: "#9a9a9a", title: i % 2 ? { text: "Mean advantage", font: { color: ORANGE, size: 12 } } : "" }
        : { range: [0, 1.06], tickformat: ".0%", zeroline: false, title: i % 2 ? { text: measure, font: { color: ORANGE, size: 12 } } : "" }) };
    layout[`yaxis${n + 4}`] = { overlaying: share, anchor: x, side: "left", range: [0, limit * 1.06], gridcolor: "#ececec", zeroline: false,
      title: i % 2 ? "" : "Completion tokens" };
    layout.annotations.push({ text: `${level} swaps`, x: (col[0] + col[1]) / 2, y: row[1] + 0.02, xref: "paper", yref: "paper", showarrow: false, font: { size: 15 } });
    const first = i === 0;
    if (advantage) {
      const a = d.advantage[measure];
      for (const [key, color, name] of [["with", ORANGE, measure], ["without", GREEN, "Without"]]) {
        traces.push({ x: a.x, y: a[key], error_y: { type: "data", array: a[`${key}_error`], color, thickness: 1, width: 2 }, mode: "lines+markers",
          line: { color, width: 2 }, marker: { size: 5 }, xaxis: x, yaxis: share, name, legendgroup: key, showlegend: first, hovertemplate: "%{y:.2f}" });
      }
    } else {
      const right = measure === "Cut off at the token limit"
        ? { type: "bar", x: d.centers, y: d.cut, marker: { color: "#eda100" }, width: 8 }
        : { type: "scatter", mode: "lines", x: d.steps, y: d.measures[measure], line: { color: ORANGE, width: 2 } };
      traces.push({ ...right, xaxis: x, yaxis: share, name: measure, legendgroup: "m", showlegend: first, hovertemplate: "%{y:.0%}" });
    }
    traces.push({ x: d.centers, y: d.low, xaxis: x, yaxis: tokens, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup: "b" });
    traces.push({ x: d.centers, y: d.high, xaxis: x, yaxis: tokens, mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: "rgba(42,120,214,0.25)", name: "5th to 95th percentile", legendgroup: "b", showlegend: first, hoverinfo: "skip" });
    traces.push({ x: d.centers, y: d.median, xaxis: x, yaxis: tokens, mode: "lines", line: { color: BLUE, width: 2 }, name: "Median",
      legendgroup: "med", showlegend: first, hovertemplate: "%{y:.0f} tokens" });
    traces.push({ x: d.centers, y: d.longest, xaxis: x, yaxis: tokens, mode: "markers", marker: { color: "#595959", size: 6 }, name: "Longest",
      legendgroup: "max", showlegend: first, hovertemplate: "%{y:.0f} tokens" });
    traces.push({ x: [0, 200], y: [limit, limit], xaxis: x, yaxis: tokens, mode: "lines", line: { color: "#000", dash: "dash", width: 1 },
      name: "Token limit", legendgroup: "lim", showlegend: first, hoverinfo: "skip" });
  });
  Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
}
draw();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    runs = sys.argv[1:] or ["swaps-7b-tags-v2", "swaps-7b-tags-long"]
    data = {run: summarize(run) for run in runs}
    Path("figures").mkdir(exist_ok=True)
    Path("figures/length-interactive.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
