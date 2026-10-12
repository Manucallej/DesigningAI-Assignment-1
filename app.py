"""
Assignment 1 — Gradio dashboard.

You're using a script for this stage. If you'd rather use a notebook,
write app.ipynb instead and delete this file — main.py refuses to run if
it finds both app.py and app.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py app`, this file must build and launch a Gradio
app (a `demo` that calls `.launch()`) using only what your training stage
already produced — it must never retrain anything itself. At minimum, the
dashboard must let you:

- Compare your three trained models via a prediction-vs-actual plot.
- See feature and/or target distributions.
- For classification: move a decision-threshold slider and watch the
  confusion matrix, and a business-cost number tied to your REPORT.md's
  framing, change with it.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""

"""
Assignment 1 - Gradio dashboard.

Run with:  uv run python main.py app

Loads the preprocessor and all THREE trained models from artifacts/, loads the
saved test frame, and runs inference here. Nothing is retrained.

Views:
  - Model comparison: test metrics + calibration (predicted prob vs actual rate).
    For a yes/no target, calibration is the honest analogue of the regression
    "prediction vs actual" plot: a perfect model sits on the diagonal.
  - Distributions: target balance and a feature distribution split by outcome.
  - Threshold & business cost: pick a model, move the threshold, watch the
    confusion matrix and a cost number (tied to REPORT.md's framing) change.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

# Must match train.py's feature lists / target.
NUMERIC = [
    "lead_time", "total_nights", "total_guests", "adr",
    "booking_changes", "days_in_waiting_list", "total_requests",
    "previous_cancellations", "previous_bookings_not_canceled",
    "required_car_parking_spaces", "adults", "children", "babies",
    "arrival_date_week_number",
]
CATEGORICAL = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
    "is_repeated_guest", "has_agent", "has_company", "is_family",
    "arrival_date_month",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "is_canceled"

# ---- Business cost assumptions (edit to match REPORT.md) ----
COST_FN = 100.0   # euros lost when a cancellation slips through (empty room)
COST_FP = 40.0    # euros of goodwill cost when a loyal guest is wrongly flagged


class LogReg(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x)


# ---- Load artifacts and run each model once on the test set ----
pre = joblib.load(ART / "preprocessor.joblib")
test = pd.read_csv(ART / "test.csv")
y_true = test[TARGET].to_numpy()
X = pre.transform(test[FEATURES])

skl = joblib.load(ART / "sklearn.joblib")
manual = torch.load(ART / "manual.pt")
std_model = LogReg(X.shape[1])
std_model.load_state_dict(torch.load(ART / "standard.pt"))
std_model.eval()

with torch.no_grad():
    Xt = torch.tensor(X, dtype=torch.float32)
    p_manual = torch.sigmoid(Xt @ manual["w"] + manual["b"]).numpy().ravel()
    p_standard = torch.sigmoid(std_model(Xt)).numpy().ravel()

PROBAS = {
    "scikit-learn": skl.predict_proba(X)[:, 1],
    "manual pytorch": p_manual,
    "standard pytorch": p_standard,
}
results_table = pd.read_csv(ART / "results.csv")
lead_time = test["lead_time"].to_numpy()


def confusion_counts(y, proba, thr):
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def calibration_figure():
    fig = go.Figure()
    bins = np.linspace(0, 1, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    for name, proba in PROBAS.items():
        idx = np.clip(np.digitize(proba, bins) - 1, 0, len(centers) - 1)
        actual = [y_true[idx == b].mean() if (idx == b).any() else np.nan
                  for b in range(len(centers))]
        fig.add_trace(go.Scatter(x=centers, y=actual, mode="lines+markers", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="perfect"))
    fig.update_layout(title="Predicted probability vs actual rate (calibration)",
                      xaxis_title="Predicted probability of cancelling",
                      yaxis_title="Actual cancellation rate", height=420)
    return fig


def distribution_figure():
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=lead_time[y_true == 0], name="showed up",
                               opacity=0.6, nbinsx=40))
    fig.add_trace(go.Histogram(x=lead_time[y_true == 1], name="canceled",
                               opacity=0.6, nbinsx=40))
    fig.update_layout(barmode="overlay", title="Lead time by outcome (test set)",
                      xaxis_title="Lead time (days before arrival)",
                      yaxis_title="Number of bookings", height=420)
    return fig


def threshold_view(model_name, thr):
    proba = PROBAS[model_name]
    tp, tn, fp, fn = confusion_counts(y_true, proba, thr)
    cm = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["pred: show", "pred: cancel"],
        y=["actual: show", "actual: cancel"],
        text=[[tn, fp], [fn, tp]], texttemplate="%{text}",
        colorscale="Blues", showscale=False))
    cm.update_layout(title=f"Confusion matrix @ threshold {thr:.2f}", height=380)
    cost = fp * COST_FP + fn * COST_FN
    caught = tp / (tp + fn) if (tp + fn) else 0
    summary = (
        f"### Business impact at threshold {thr:.2f}\n"
        f"- Cancellations caught: **{caught:.0%}** ({tp} of {tp + fn})\n"
        f"- False alarms (guests wrongly flagged): **{fp}**\n"
        f"- Missed cancellations: **{fn}**\n"
        f"- Estimated cost: **{cost:,.0f} EUR**  "
        f"(FN x {COST_FN:.0f} + FP x {COST_FP:.0f})"
    )
    return cm, summary


with gr.Blocks(title="Hotel cancellation dashboard") as demo:
    gr.Markdown("# Hotel booking cancellation model")
    gr.Markdown("Logistic regression trained three ways. The preprocessor and "
                "all three models are loaded from `artifacts/` and run here; "
                "nothing is retrained.")

    with gr.Tab("Model comparison"):
        gr.Dataframe(results_table, label="Test-set metrics")
        gr.Plot(calibration_figure())

    with gr.Tab("Distributions"):
        gr.Plot(distribution_figure())

    with gr.Tab("Threshold & business cost"):
        model_pick = gr.Dropdown(list(PROBAS.keys()), value="scikit-learn",
                                 label="Model")
        thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold")
        cm_plot = gr.Plot()
        cost_md = gr.Markdown()
        for ctrl in (model_pick, thr):
            ctrl.change(threshold_view, [model_pick, thr], [cm_plot, cost_md])
        demo.load(threshold_view, [model_pick, thr], [cm_plot, cost_md])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)