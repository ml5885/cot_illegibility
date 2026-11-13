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
  .check { color: var(--ink); cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 0; }
  .chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 6px 4px 10px; border: 1px solid var(--line); border-radius: 999px; font-size: 14px; }
  .chip i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .chip button { font: inherit; border: none; background: none; cursor: pointer; color: var(--muted); padding: 0 4px; font-size: 16px; line-height: 1; }
  select { font: inherit; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); max-width: 100%; }
  label { color: var(--muted); margin-right: 6px; }
  #plot { width: 100%; }
</style>
</head>
<body>
<h1>Completion length and the token limit</h1>
<div class="controls">
  <div class="toggle" id="runs"></div>
  <div><label for="add">Right axis</label><select id="add"></select></div>
  <label class="check"><input type="checkbox" id="length" checked> Completion length</label>
</div>
<div class="chips" id="chips"></div>
<div id="plot"></div>
<script>
const DATA = __DATA__;
const runNames = Object.keys(DATA);
const sample = Object.values(Object.values(DATA)[0].levels)[0];
const KINDS = {
  share: { label: "Share of completions", names: Object.keys(sample.measures).concat(["Cut off at the token limit"]) },
  advantage: { label: "Advantage gap (with minus without)", names: Object.keys(sample.advantage) },
};
const COLORS = ["#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#008300", "#8a5cd6"];
const MAX_LINES = COLORS.length;
let run = runNames[0], showLength = true;
document.getElementById("length").onchange = event => { showLength = event.target.checked; draw(); };
let lines = [{ kind: "share", name: "Completions with a newline", color: COLORS[0] }];

const BOTH = "both", STYLES = [{ dash: "solid", symbol: "circle" }, { dash: "dash", symbol: "circle-open" }];
const runLabel = name => `${DATA[name].limit.toLocaleString()}-token limit`;
const runOptions = runNames.concat([BOTH]);
const runs = document.getElementById("runs");
for (const name of runOptions) {
  const b = document.createElement("button");
  b.textContent = name === BOTH ? "Both runs" : runLabel(name);
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

function label(line) { return line.kind === "advantage" ? `Advantage gap: ${line.name}` : line.name; }

// Mean advantage with the feature minus without it, and the standard error of that difference.
function gap(a) {
  const both = i => a.with[i] !== null && a.without[i] !== null;
  return {
    x: a.x,
    y: a.x.map((_, i) => both(i) ? a.with[i] - a.without[i] : null),
    error: a.x.map((_, i) => both(i) ? Math.hypot(a.with_error[i], a.without_error[i]) : null),
  };
}
function gapLabel(name) {
  if (name === "Length") return "longer minus shorter";
  if (name === "Cut off at the token limit") return "cut off minus not";
  return `with minus without ${name[0].toLowerCase()}${name.slice(1)}`;
}

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
let panels = [], titles = [];

function at(xs, ys, x) {
  let best = null, distance = 5.5;
  xs.forEach((value, i) => {
    if (ys[i] !== null && Math.abs(value - x) < distance) { best = ys[i]; distance = Math.abs(value - x); }
  });
  return best;
}
const percent = v => v === null ? "–" : `${Math.round(v * 100)}%`;
const signed = v => v === null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

function tooltip(p, step) {
  const overlay = p.runs.length > 1, rows = [`<b>Step ${step}</b>`];
  for (const { run: name, d } of p.runs) {
    if (overlay) rows.push(`<i>${runLabel(name)}</i>`);
    if (showLength) {
      rows.push(`Median: ${at(d.centers, d.median, step) ?? "–"} tokens`);
      if (!overlay) rows.push(`5th to 95th percentile: ${at(d.centers, d.low, step) ?? "–"} to ${at(d.centers, d.high, step) ?? "–"}`);
      rows.push(`Longest: ${at(d.centers, d.longest, step) ?? "–"}`);
    }
    for (const line of lines) {
      const dot = `<span style="color:${line.color}">●</span>`;
      if (line.kind === "advantage") {
        const g = gap(d.advantage[line.name]);
        rows.push(`${dot} Advantage gap, ${gapLabel(line.name)}: ${signed(at(g.x, g.y, step))}`);
      } else if (line.name === "Cut off at the token limit") {
        rows.push(`${dot} Cut off: ${percent(at(d.centers, d.cut, step))}`);
      } else {
        rows.push(`${dot} ${line.name}: ${percent(at(d.steps, d.measures[line.name], step))}`);
      }
    }
  }
  const right = step > 110;
  return { text: rows.join("<br>"), xref: p.x, x: step, yref: "paper", y: p.row[1] - 0.01, xanchor: right ? "right" : "left", yanchor: "top",
    xshift: right ? -8 : 8, showarrow: false, align: "left", bgcolor: "rgba(255,255,255,0.95)", bordercolor: "#8a8a8a", borderwidth: 1, borderpad: 4,
    font: { size: 11 } };
}

function draw() {
  [...runs.children].forEach((b, i) => b.classList.toggle("on", runOptions[i] === run));
  fillPicker();
  renderChips();
  const active = run === BOTH ? runNames : [run], overlay = active.length > 1;
  const limit = Math.max(...active.map(name => DATA[name].limit)), levels = DATA[active[0]].levels;
  const kinds = [...new Set(lines.map(line => line.kind))];
  const extra = Math.max(kinds.length - 1, 0);
  const domains = [[0, 0.46], [0.54, 1 - GAP * extra]], rows = [[0.56, 1], [0, 0.44]];
  panels = [];
  const traces = [], layout = {
    font: FONT, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", hovermode: "x", showlegend: true,
    legend: { orientation: "h", x: 0.5, xanchor: "center", yref: "container", y: 0, yanchor: "bottom", bordercolor: "#000", borderwidth: 1 },
    margin: { l: 70, r: 40, t: 40, b: 110 }, annotations: [], autosize: true,
    height: Math.max(520, window.innerHeight - document.getElementById("plot").getBoundingClientRect().top - 8),
  };
  Object.keys(levels).forEach((level, i) => {
    const n = i + 1, x = `x${n}`, tokens = `y${60 + n}`, first = i === 0, shown = i % 2 === 1;
    const col = domains[i % 2], row = rows[Math.floor(i / 2)];
    layout[`xaxis${n}`] = { domain: col, anchor: `y${n}`, range: [0, 200], title: i >= 2 ? "Training step" : "", showgrid: false, zeroline: false };
    layout[`yaxis${n}`] = { domain: row, anchor: x, side: "right", showgrid: false, zeroline: false, showticklabels: false, ticks: "", range: [0, 1] };
    traces.push({ x: [null], y: [null], xaxis: x, yaxis: `y${n}`, showlegend: false, hoverinfo: "skip" });
    kinds.forEach((kind, j) => {
      const id = j === 0 ? n : 10 + n, members = lines.filter(line => line.kind === kind);
      const color = members.length === 1 ? members[0].color : "#555555", shown = i % 2 === 1 || (j === 0 && !showLength);
      const scale = kind === "advantage"
        ? { range: [-1.5, 1.5], tickformat: ".1f", zeroline: true, zerolinecolor: "#9a9a9a", zerolinewidth: 1 }
        : { range: [0, 1.06], tickformat: ".0%", zeroline: false };
      layout[`yaxis${id}`] = {
        ...(j === 0 ? { domain: row, anchor: x } : { overlaying: `y${n}`, anchor: "free", position: Math.min(col[1] + GAP * j, 1) }),
        side: "right", showgrid: j === 0 && !showLength, gridcolor: "#ececec", showline: shown, linecolor: color, showticklabels: shown, ticks: shown ? "outside" : "",
        tickfont: { color, size: 11 }, title: shown ? { text: kind === "advantage" ? "Advantage gap" : "Share", font: { color, size: 11 }, standoff: 2 } : "",
        ...scale,
      };
    });
    const series = active.map(name => ({ run: name, d: DATA[name].levels[level] }));
    series.forEach(({ run: name, d }, k) => {
      const { dash, symbol } = STYLES[runNames.indexOf(name)], legend = first && k === 0;
      lines.forEach(line => {
        const j = kinds.indexOf(line.kind), axis = `y${j === 0 ? n : 10 + n}`;
        if (line.kind === "advantage") {
          const g = gap(d.advantage[line.name]);
          traces.push({ x: g.x, y: g.y, error_y: { type: "data", array: g.error, color: line.color, thickness: 1, width: 2 },
            mode: "lines+markers", line: { color: line.color, width: 2, dash }, marker: { size: 5, symbol }, xaxis: x, yaxis: axis,
            name: `Advantage gap, ${gapLabel(line.name)}`, legendgroup: line.name, showlegend: legend, hoverinfo: "none" });
        } else if (line.name === "Cut off at the token limit") {
          traces.push({ type: "bar", x: d.centers, y: d.cut, width: 8, xaxis: x, yaxis: axis, offsetgroup: name,
            marker: { color: line.color, opacity: overlay ? 0.6 : 0.8, pattern: { shape: k ? "/" : "" } },
            name: line.name, legendgroup: line.name, showlegend: legend, hoverinfo: "none" });
        } else {
          traces.push({ x: d.steps, y: d.measures[line.name], mode: "lines", line: { color: line.color, width: 2, dash }, xaxis: x, yaxis: axis,
            name: line.name, legendgroup: line.name, showlegend: legend, hoverinfo: "none" });
        }
      });
    });
    panels.push({ x, row, col, runs: series, level });
    layout.annotations.push({ text: `${level} swaps`, x: (col[0] + col[1]) / 2, y: row[1] + 0.02, xref: "paper", yref: "paper", showarrow: false, font: { size: 15 } });
    if (!showLength) return;
    layout[`yaxis${60 + n}`] = { overlaying: `y${n}`, anchor: x, side: "left", range: [0, limit * 1.06], gridcolor: "#ececec", zeroline: false,
      title: i % 2 ? "" : "Completion tokens" };
    series.forEach(({ run: name, d }, k) => {
      const { dash, symbol } = STYLES[runNames.indexOf(name)], legend = first && k === 0, cap = DATA[name].limit;
      traces.push({ x: d.centers, y: d.low, xaxis: x, yaxis: tokens, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup: "band" });
      traces.push({ x: d.centers, y: d.high, xaxis: x, yaxis: tokens, mode: "lines", line: { width: 0 }, fill: "tonexty",
        fillcolor: `rgba(42,120,214,${overlay ? 0.14 : 0.25})`, name: "5th to 95th percentile", legendgroup: "band", showlegend: legend, hoverinfo: "skip" });
      traces.push({ x: d.centers, y: d.median, xaxis: x, yaxis: tokens, mode: "lines", line: { color: BLUE, width: 2, dash }, name: "Median",
        legendgroup: "median", showlegend: legend, hoverinfo: "none" });
      traces.push({ x: d.centers, y: d.longest, xaxis: x, yaxis: tokens, mode: "markers", marker: { color: "#595959", size: 6, symbol }, name: "Longest",
        legendgroup: "longest", showlegend: legend, hoverinfo: "none" });
      traces.push({ x: [0, 200], y: [cap, cap], xaxis: x, yaxis: tokens, mode: "lines", line: { color: "#000", dash: "dash", width: 1 },
        name: "Token limit", legendgroup: "limit", showlegend: legend, hoverinfo: "skip" });
      if (overlay) layout.annotations.push({ text: cap.toLocaleString(), xref: x, x: 200, yref: tokens, y: cap, xanchor: "right", yanchor: "bottom",
        showarrow: false, font: { size: 10, color: "#555" } });
    });
  });
  if (overlay) runNames.forEach((name, k) => traces.push({ x: [null], y: [null], xaxis: "x1", yaxis: "y1", mode: "lines+markers",
    line: { color: "#444", width: 2, dash: STYLES[k].dash }, marker: { color: "#444", size: 6, symbol: STYLES[k].symbol }, name: runLabel(name),
    legendgroup: `run${k}`, hoverinfo: "skip" }));
  titles = layout.annotations;
  Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
}

const plot = document.getElementById("plot");
function cursor(x) {
  if (x === null) return Plotly.relayout(plot, { shapes: [], annotations: titles });
  const step = Math.round(x);
  const shapes = panels.map(p => ({ type: "line", xref: p.x, x0: step, x1: step, yref: "paper", y0: p.row[0], y1: p.row[1],
    line: { color: "#8a8a8a", width: 1 }, layer: "above" }));
  Plotly.relayout(plot, { shapes, annotations: titles.concat(panels.map(p => tooltip(p, step))) });
}
draw();
let resizing;
window.addEventListener("resize", () => { clearTimeout(resizing); resizing = setTimeout(draw, 150); });
plot.on("plotly_hover", event => cursor(event.xvals ? event.xvals[0] : event.points[0].x));
plot.on("plotly_unhover", () => cursor(null));

// Axes that play the same role in each panel: x1-4, the first right axis y1-4, the second y11-14, the token axis y61-64.
const GROUPS = { x: n => [1, 2, 3, 4], y: n => [1, 2, 3, 4].map(i => (n <= 4 ? 0 : n <= 14 ? 10 : 60) + i) };
let syncing = false;
plot.on("plotly_relayout", event => {
  if (syncing) return;
  const update = {};
  for (const [key, value] of Object.entries(event)) {
    const m = key.match(/^([xy])axis(\\d*)\\.(range(?:\\[[01]\\])?|autorange)$/);
    if (!m) continue;
    for (const n of GROUPS[m[1]](Number(m[2] || 1))) update[`${m[1]}axis${n === 1 ? "" : n}.${m[3]}`] = value;
  }
  if (!Object.keys(update).length) return;
  syncing = true;
  Plotly.relayout(plot, update).then(() => { syncing = false; });
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    runs = sys.argv[1:] or ["swaps-7b-tags-v2", "swaps-7b-tags-long"]
    data = {run: summarize(run) for run in runs}
    Path("figures").mkdir(exist_ok=True)
    Path("figures/length-interactive.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
