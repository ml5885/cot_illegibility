"""Save a wandb workspace view for the swap runs: accuracy and length by swap count, training and held-out, in one panel each.

uv run --with wandb-workspaces python wandb_view.py   (where wandb is logged in, e.g. fukuyama)
"""

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY, PROJECT = "bearseascape-carnegie-mellon-university", "cot-illegibility"
LEVELS = [2, 4, 8, 16]


def by_level(split: str, metric: str, title: str, smoothing: float) -> wr.LinePlot:
    return wr.LinePlot(title=title, x="Step", y=[f"{split}_levels/{metric}/level_{level}" for level in LEVELS],
                       smoothing_factor=smoothing, legend_position="south")


if __name__ == "__main__":
    workspace = ws.Workspace(
        name="Swaps by swap count",
        entity=ENTITY,
        project=PROJECT,
        sections=[
            ws.Section(name="Accuracy by swap count", is_open=True, panels=[
                by_level("train", "accuracy", "Training accuracy", 0.6),
                by_level("eval", "accuracy", "Held-out accuracy", 0),
            ]),
            ws.Section(name="Length by swap count", is_open=True, panels=[
                by_level("train", "tokens", "Training completion tokens", 0.6),
                wr.LinePlot(title="Token limit", x="Step", y=["token_limit"]),
            ]),
        ],
    )
    print(workspace.save().url)
