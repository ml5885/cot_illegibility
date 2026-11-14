"""Save a wandb workspace view for the swap runs: accuracy and length by swap count, training and held-out, one panel
each, with one colour per swap count. Line colours are set per run, so rerun this after new runs start.

uv run --with wandb-workspaces python wandb_view.py   (where wandb is logged in, e.g. fukuyama)
"""

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY, PROJECT = "bearseascape-carnegie-mellon-university", "cot-illegibility"
# The swap-count colours of the repo's figures (viridis at 0, 0.3, 0.6, 0.85).
COLORS = {2: "#440154", 4: "#31688e", 8: "#22a884", 16: "#a2da37"}
STEP = "train/global_step"


def by_level(run_ids: list[str], split: str, metric: str, title: str, smoothing: float = 0, mark: str = "solid") -> wr.LinePlot:
    keys = {level: f"{split}_levels/{metric}/level_{level}" for level in COLORS}
    series = lambda value: {f"{run}:{keys[level]}": value(level) for run in run_ids for level in COLORS}
    return wr.LinePlot(
        title=title, x=STEP, y=list(keys.values()), smoothing_factor=smoothing, legend_position="south",
        line_colors=series(COLORS.get), line_titles=series(lambda level: f"{level} swaps"), line_marks=series(lambda _: mark),
    )


if __name__ == "__main__":
    run_ids = [run.id for run in wandb.Api().runs(f"{ENTITY}/{PROJECT}")]
    workspace = ws.Workspace(
        name="Swaps by swap count",
        entity=ENTITY,
        project=PROJECT,
        sections=[
            ws.Section(name="Accuracy by swap count", is_open=True, panels=[
                by_level(run_ids, "train", "accuracy", "Training accuracy", smoothing=0.6),
                by_level(run_ids, "eval", "accuracy", "Held-out accuracy", mark="points"),
            ]),
            ws.Section(name="Length by swap count", is_open=True, panels=[
                by_level(run_ids, "train", "tokens", "Training completion tokens", smoothing=0.6),
                wr.LinePlot(title="Token limit", x=STEP, y=["token_limit"]),
            ]),
        ],
    )
    print(workspace.save().url)
