"""Interactive completion-length plot with a selectable right-axis measure, written to figures/length-interactive.html."""

import json
import re
import sys
from pathlib import Path

from plots import BIN, WINDOW, first_word, load, newline, swap_steps, tracking

LIST = re.compile(r"(?:^|\s)1[.)]\s.*\s2[.)]\s")


def ratio(hit) -> callable:
    return lambda row: (float(bool(hit(row))), 1.0)


def per_swap(count) -> callable:
    def measure(row: dict) -> tuple[float, float]:
        steps = swap_steps(row)
        return float(sum(map(count, steps))), float(len(steps))

    return measure


MEASURES = {
    "Completions with a newline": ratio(newline),
    "Completions with a blank line": ratio(lambda row: "\n\n" in row["completion"]),
    "Correct": ratio(lambda row: row["reward"]),
    "Swaps where the stated key location is right": per_swap(lambda step: step[0] == "right"),
    "Swaps followed by a standalone location statement": per_swap(lambda step: step[1]),
    "Responses that write out every swap": ratio(lambda row: "skipped" not in tracking(row)),
    "Numbered list": ratio(lambda row: LIST.search(" ".join(row["completion"].split()))),
    '"Step N:" headers': ratio(lambda row: re.search(r"Step \d+:", row["completion"])),
    'Opens with "Let\'s"': ratio(lambda row: first_word(row) == "Let's"),
    'Opens with "To"': ratio(lambda row: first_word(row) == "To"),
}


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
        levels[level] = {
            "steps": steps,
            "centers": [start + half / 2 for start in starts],
            "low": [b[len(b) // 20] for b in bins],
            "median": [b[len(b) // 2] for b in bins],
            "high": [b[-len(b) // 20] for b in bins],
            "longest": [b[-1] for b in bins],
            "cut": [sum(t >= limit - 2 for t in b) / len(b) for b in bins],
            "measures": measures,
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
  .runs button { font: inherit; padding: 6px 12px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; }
  .runs button:first-child { border-radius: 6px 0 0 6px; }
  .runs button:last-child { border-radius: 0 6px 6px 0; border-left: none; }
  .runs button.on { background: var(--accent); border-color: var(--accent); color: white; }
  select { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); max-width: 100%; }
  label { color: var(--muted); margin-right: 6px; }
  #plot { width: 100%; height: min(860px, 120vw); }
</style>
</head>
<body>
<h1>Completion length and the token limit</h1>
<div class="controls">
  <div class="runs" id="runs"></div>
  <div><label for="measure">Right axis</label><select id="measure"></select></div>
</div>
<div id="plot"></div>
<script>
const DATA = __DATA__;
const runNames = Object.keys(DATA);
const measureNames = Object.keys(Object.values(Object.values(DATA)[0].levels)[0].measures).concat(["Cut off at the token limit"]);
let run = runNames[0], measure = measureNames[0];

const runs = document.getElementById("runs");
for (const name of runNames) {
  const b = document.createElement("button");
  b.textContent = `${DATA[name].limit.toLocaleString()}-token limit`;
  b.onclick = () => { run = name; draw(); };
  runs.appendChild(b);
}
const select = document.getElementById("measure");
for (const name of measureNames) select.add(new Option(name, name));
select.onchange = () => { measure = select.value; draw(); };

const FONT = { family: '"DejaVu Serif", Georgia, serif', color: "#1f1f1f", size: 13 };
const ORANGE = "#eb6834", BLUE = "#2a78d6";

function draw() {
  [...runs.children].forEach((b, i) => b.classList.toggle("on", runNames[i] === run));
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
    layout[`yaxis${n}`] = { domain: row, anchor: x, side: "right", range: [0, 1.06], tickformat: ".0%", showgrid: false, zeroline: false,
      tickfont: { color: ORANGE }, title: i % 2 ? { text: measure, font: { color: ORANGE, size: 12 } } : "", showticklabels: i % 2 === 1 };
    layout[`yaxis${n + 4}`] = { overlaying: share, anchor: x, side: "left", range: [0, limit * 1.06], gridcolor: "#ececec", zeroline: false,
      title: i % 2 ? "" : "Completion tokens" };
    layout.annotations.push({ text: `${level} swaps`, x: (col[0] + col[1]) / 2, y: row[1] + 0.02, xref: "paper", yref: "paper", showarrow: false, font: { size: 15 } });
    const first = i === 0;
    const right = measure === "Cut off at the token limit"
      ? { type: "bar", x: d.centers, y: d.cut, marker: { color: "#eda100" }, width: 8 }
      : { type: "scatter", mode: "lines", x: d.steps, y: d.measures[measure], line: { color: ORANGE, width: 2 } };
    traces.push({ ...right, xaxis: x, yaxis: share, name: measure, legendgroup: "m", showlegend: first, hovertemplate: "%{y:.0%}" });
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
