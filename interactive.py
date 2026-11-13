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


def advantage_split(by_step: dict[int, list[dict]], has, yes: str, no: str) -> dict:
    """Per 10-step bin, mean advantage (and standard error) of responses with and without a feature, in groups with mixed rewards."""
    starts = range(0, max(by_step) + 1, ADVANTAGE_WIDTH)
    scored = {s: group_advantages(rows) for s, rows in by_step.items()}
    out = {"x": [start + ADVANTAGE_WIDTH / 2 for start in starts], "yes": yes, "no": no}
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
            "Length": (
                lambda row, group: row["tokens"] > sum(r["tokens"] for r in group) / len(group), "Longer than its group's mean", "Shorter than its group's mean",
            ),
            "Cut off at the token limit": (lambda row, group: row["tokens"] >= limit - 2, "Cut off", "Not cut off"),
        } | {
            name[0].upper() + name[1:]: ((lambda f: lambda row, group: f(row))(f), f"With {name}", f"Without {name}")
            for name, f in FEATURES.items()
        }
        levels[level] = {
            "steps": steps,
            "centers": [start + half / 2 for start in starts],
            "low": [b[len(b) // 20] for b in bins],
            "median": [b[len(b) // 2] for b in bins],
            "high": [b[-len(b) // 20] for b in bins],
            "longest": [b[-1] for b in bins],
            "cut": [sum(t >= limit - 2 for t in b) / len(b) for b in bins],
            "measures": measures,
            "advantage": {name: advantage_split(by_step, *split) for name, split in splits.items()},
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
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 0; }
  .chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 6px 4px 10px; border: 1px solid var(--line); border-radius: 999px; font-size: 14px; }
  .chip i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .chip button { font: inherit; border: none; background: none; cursor: pointer; color: var(--muted); padding: 0 4px; font-size: 16px; line-height: 1; }
  select { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); max-width: 100%; }
  label { color: var(--muted); margin-right: 6px; }
  #plot { width: 100%; height: min(860px, 120vw); }
</style>
</head>
<body>
<h1>Completion length and the token limit</h1>
<div class="controls">
  <div class="toggle" id="runs"></div>
  <div><label for="add">Right axis</label><select id="add"></select></div>
</div>
<div class="chips" id="chips"></div>
<div id="plot"></div>
<script>
const DATA = __DATA__;
const runNames = Object.keys(DATA);
const sample = Object.values(Object.values(DATA)[0].levels)[0];
const KINDS = {
  share: { label: "Share of completions", names: Object.keys(sample.measures).concat(["Cut off at the token limit"]) },
  advantage: { label: "Mean advantage", names: Object.keys(sample.advantage) },
};
const COLORS = ["#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#008300", "#8a5cd6"];
const MAX_LINES = COLORS.length;
let run = runNames[0];
let lines = [{ kind: "share", name: "Completions with a newline", color: COLORS[0] }];

const runs = document.getElementById("runs");
for (const name of runNames) {
  const b = document.createElement("button");
  b.textContent = `${DATA[name].limit.toLocaleString()}-token limit`;
  b.onclick = () => { run = name; draw(); };
  runs.appendChild(b);
}

const add = document.getElementById("add");
function fillPicker() {
  add.innerHTML = "";
  add.add(new Option(lines.length < MAX_LINES ? "Add a line…" : `At most ${MAX_LINES} lines`, ""));
  for (const [kind, { label, names }] of Object.entries(KINDS)) {
    const group = document.createElement("optgroup");
    group.label = label;
    for (const name of names) {
      const option = new Option(name, `${kind}|${name}`);
      option.disabled = lines.length >= MAX_LINES || lines.some(l => l.kind === kind && l.name === name);
      group.appendChild(option);
    }
    add.appendChild(group);
  }
}
add.onchange = () => {
  const [kind, name] = add.value.split("|");
  const color = COLORS.find(c => !lines.some(l => l.color === c));
  if (name && color) lines.push({ kind, name, color });
  draw();
};

function label(line) { return line.kind === "advantage" ? `Advantage: ${line.name}` : line.name; }

function renderChips() {
  const box = document.getElementById("chips");
  box.innerHTML = "";
  for (const line of lines) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<i style="background:${line.color}"></i>`;
    chip.append(label(line));
    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.title = "Remove";
    remove.onclick = () => { lines = lines.filter(l => l !== line); draw(); };
    chip.appendChild(remove);
    box.appendChild(chip);
  }
}

const FONT = { family: '"DejaVu Serif", Georgia, serif', color: "#1f1f1f", size: 13 };
const BLUE = "#2a78d6", GAP = 0.055;

function draw() {
  [...runs.children].forEach((b, i) => b.classList.toggle("on", runNames[i] === run));
  fillPicker();
  renderChips();
  const { limit, levels } = DATA[run];
  const extra = Math.max(lines.length - 1, 0);
  const domains = [[0, 0.46], [0.54, 1 - GAP * extra]], rows = [[0.56, 1], [0, 0.44]];
  const traces = [], layout = {
    font: FONT, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", hovermode: "x unified", showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", y: -0.13, bordercolor: "#000", borderwidth: 1 },
    margin: { l: 70, r: 40, t: 40, b: 120 }, annotations: [],
  };
  Object.entries(levels).forEach(([level, d], i) => {
    const n = i + 1, x = `x${n}`, tokens = `y${60 + n}`, first = i === 0, shown = i % 2 === 1;
    const col = domains[i % 2], row = rows[Math.floor(i / 2)];
    layout[`xaxis${n}`] = { domain: col, anchor: `y${n}`, range: [0, 200], title: i >= 2 ? "Training step" : "", showgrid: false, zeroline: false };
    layout[`yaxis${n}`] = { domain: row, anchor: x, side: "right", showgrid: false, zeroline: false, showticklabels: false, ticks: "", range: [0, 1] };
    lines.forEach((line, j) => {
      const id = j === 0 ? n : 10 + (n - 1) * MAX_LINES + j, axis = `y${id}`;
      const scale = line.kind === "advantage"
        ? { range: [-1.5, 1.5], tickformat: ".1f", zeroline: true, zerolinecolor: line.color, zerolinewidth: 1 }
        : { range: [0, 1.06], tickformat: ".0%", zeroline: false };
      layout[`yaxis${id}`] = {
        ...(j === 0 ? { domain: row, anchor: x } : { overlaying: `y${n}`, anchor: "free", position: Math.min(col[1] + GAP * j, 1) }),
        side: "right", showgrid: false, showline: shown, linecolor: line.color, showticklabels: shown, ticks: shown ? "outside" : "",
        tickfont: { color: line.color, size: 11 }, title: shown ? { text: line.kind === "advantage" ? "Mean advantage" : "Share", font: { color: line.color, size: 11 }, standoff: 2 } : "",
        ...scale,
      };
      if (line.kind === "advantage") {
        const a = d.advantage[line.name];
        for (const [key, dash, side] of [["with", "solid", "yes"], ["without", "dash", "no"]]) {
          traces.push({ x: a.x, y: a[key], error_y: { type: "data", array: a[`${key}_error`], color: line.color, thickness: 1, width: 2 },
            mode: "lines+markers", line: { color: line.color, width: 2, dash }, marker: { size: 5 }, xaxis: x, yaxis: axis,
            name: a[side], legendgroup: `${line.name}${key}`, showlegend: first, hovertemplate: "%{y:.2f}" });
        }
      } else if (line.name === "Cut off at the token limit") {
        traces.push({ type: "bar", x: d.centers, y: d.cut, marker: { color: line.color, opacity: 0.8 }, width: 8, xaxis: x, yaxis: axis,
          name: line.name, legendgroup: line.name, showlegend: first, hovertemplate: "%{y:.0%}" });
      } else {
        traces.push({ x: d.steps, y: d.measures[line.name], mode: "lines", line: { color: line.color, width: 2 }, xaxis: x, yaxis: axis,
          name: line.name, legendgroup: line.name, showlegend: first, hovertemplate: "%{y:.0%}" });
      }
    });
    layout[`yaxis${60 + n}`] = { overlaying: `y${n}`, anchor: x, side: "left", range: [0, limit * 1.06], gridcolor: "#ececec", zeroline: false,
      title: i % 2 ? "" : "Completion tokens" };
    layout.annotations.push({ text: `${level} swaps`, x: (col[0] + col[1]) / 2, y: row[1] + 0.02, xref: "paper", yref: "paper", showarrow: false, font: { size: 15 } });
    traces.push({ x: d.centers, y: d.low, xaxis: x, yaxis: tokens, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup: "band" });
    traces.push({ x: d.centers, y: d.high, xaxis: x, yaxis: tokens, mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: "rgba(42,120,214,0.25)", name: "5th to 95th percentile", legendgroup: "band", showlegend: first, hoverinfo: "skip" });
    traces.push({ x: d.centers, y: d.median, xaxis: x, yaxis: tokens, mode: "lines", line: { color: BLUE, width: 2 }, name: "Median",
      legendgroup: "median", showlegend: first, hovertemplate: "%{y:.0f} tokens" });
    traces.push({ x: d.centers, y: d.longest, xaxis: x, yaxis: tokens, mode: "markers", marker: { color: "#595959", size: 6 }, name: "Longest",
      legendgroup: "longest", showlegend: first, hovertemplate: "%{y:.0f} tokens" });
    traces.push({ x: [0, 200], y: [limit, limit], xaxis: x, yaxis: tokens, mode: "lines", line: { color: "#000", dash: "dash", width: 1 },
      name: "Token limit", legendgroup: "limit", showlegend: first, hoverinfo: "skip" });
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
