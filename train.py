"""
Assignment 1 — training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file — main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via `uv run python main.py train`, this file must, start to finish, with no
manual steps in between:

- Load your committed dataset.
- Apply the feature engineering, preprocessing, and train/val/test split
  consistent with the business framing you wrote in REPORT.md.
- Train the same model — linear regression for a regression target,
  logistic regression for a binary one — three ways, on the same split:
  scikit-learn, a from-scratch PyTorch loop (Session 5's manual approach:
  raw tensors, autograd, a manual gradient-step update), and the standard
  torch.nn.Module + torch.optim workflow.
- Evaluate all three against each other and against a naive baseline.
- Save whatever app.py needs to build its dashboard without retraining
  anything.

How you structure the code beyond that — file layout, function
boundaries, naming — is your call to make and be able to defend.
"""
"""
Assignment 1 - training pipeline.

Run with:  uv run python main.py train

Start to finish, no manual steps:
  1. Load the committed hotel_bookings.csv.
  2. Drop leakage columns, clean bad rows, engineer features.
  3. Split by arrival date (time-based) into train / val / test.
  4. Fit preprocessing on train only.
  5. Use the VALIDATION set to pick the L2 strength and the torch learning
     rates (sparing tuning, as the brief allows).
  6. Train logistic regression three ways on the SAME split, under the SAME
     regularization, so the three are directly comparable:
       - scikit-learn LogisticRegression
       - a from-scratch PyTorch loop (raw tensors + autograd + manual step)
       - the standard torch.nn.Module + torch.optim workflow
  7. Evaluate all three plus a naive baseline on the test set.
  8. Save the preprocessor, all three models, and the test frame so app.py can
     load the models and run inference itself (it never retrains).

Regularization alignment:
  All three minimize  mean_BCE + LAMBDA * sum(w^2)  (intercept unpenalized).
  sklearn minimizes   0.5*||w||^2 + C*sum_BCE, which per-sample equals
  (1/(2*C*n))*||w||^2 + mean_BCE. Matching the two gives C = 1/(2*LAMBDA*n).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hotel_bookings.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12,
}

# Known only at or after check-in -> using them leaks the future.
LEAK_COLS = ["reservation_status", "reservation_status_date", "assigned_room_type"]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"],
        month=df["arrival_date_month"].map(MONTHS),
        day=df["arrival_date_day_of_month"],
    ))
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("UNK")
    df = df[(df["adr"] >= 0) & (df["adr"] <= 1000)].copy()
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0].copy()
    df = df.drop(columns=LEAK_COLS)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_agent"] = df["agent"].notna().astype(int)
    df["has_company"] = df["company"].notna().astype(int)
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["total_requests"] = df["total_of_special_requests"]
    return df


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


def time_split(df: pd.DataFrame):
    df = df.sort_values("arrival_date")
    cut1 = df["arrival_date"].quantile(0.70)
    cut2 = df["arrival_date"].quantile(0.85)
    train = df[df["arrival_date"] <= cut1]
    val = df[(df["arrival_date"] > cut1) & (df["arrival_date"] <= cut2)]
    test = df[df["arrival_date"] > cut2]
    return train, val, test


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def auc(y, proba):
    return roc_auc_score(y, proba)


# ---- sklearn ----
def c_from_lambda(lam, n):
    return 1.0 / (2.0 * lam * n)


def fit_sklearn(Xtr, ytr, lam):
    C = c_from_lambda(lam, len(ytr))
    m = LogisticRegression(max_iter=2000, C=C)
    m.fit(Xtr, ytr)
    return m


# ---- manual PyTorch loop ----
def train_manual(Xtr, ytr, lr, lam, epochs=600):
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)
    d = X.shape[1]
    w = torch.zeros((d, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for _ in range(epochs):
        p = torch.sigmoid(X @ w + b)
        bce = -(y * torch.log(p + 1e-7) + (1 - y) * torch.log(1 - p + 1e-7)).mean()
        loss = bce + lam * (w ** 2).sum()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    return w.detach(), b.detach()


def proba_manual(X, w, b):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(X @ w + b).numpy().ravel()


# ---- standard nn.Module + optim ----
class LogReg(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = torch.nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x)


def train_standard(Xtr, ytr, lr, lam, epochs=600):
    X = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32).reshape(-1, 1)
    model = LogReg(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # penalty added by hand below
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(X)
        loss = bce(out, y) + lam * (model.linear.weight ** 2).sum()
        loss.backward()
        opt.step()
    return model


def proba_standard(X, model):
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy().ravel()


def evaluate(name, y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"model": name,
            "accuracy": round(accuracy_score(y, pred), 4),
            "auc": round(roc_auc_score(y, proba), 4)}


def main():
    print("Loading and cleaning...")
    df = engineer(load_and_clean())
    train, val, test = time_split(df)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    pre = build_preprocessor()
    Xtr = pre.fit_transform(train[FEATURES])
    Xval = pre.transform(val[FEATURES])
    Xte = pre.transform(test[FEATURES])
    ytr, yval, yte = (train[TARGET].to_numpy(), val[TARGET].to_numpy(),
                      test[TARGET].to_numpy())

    # ---- Tune L2 strength on validation (using the cheap sklearn fit) ----
    print("Tuning L2 strength on the validation set...")
    LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2]
    best_lam, best_auc = None, -1
    for lam in LAMBDAS:
        a = auc(yval, fit_sklearn(Xtr, ytr, lam).predict_proba(Xval)[:, 1])
        print(f"  lambda={lam:<7} val AUC={a:.4f}")
        if a > best_auc:
            best_lam, best_auc = lam, a
    print(f"  chosen lambda={best_lam}")

    # ---- Tune learning rate on validation for each torch method ----
    print("Tuning learning rates on the validation set...")
    LRS = [0.05, 0.1, 0.5]
    best_lr_man = max(LRS, key=lambda lr: auc(
        yval, proba_manual(Xval, *train_manual(Xtr, ytr, lr, best_lam))))
    best_lr_std = max(LRS, key=lambda lr: auc(
        yval, proba_standard(Xval, train_standard(Xtr, ytr, lr, best_lam))))
    print(f"  manual lr={best_lr_man}   standard lr={best_lr_std}")

    # ---- Final fits on train, same lambda, evaluated on test ----
    results = []
    results.append({"model": "baseline (majority)",
                    "accuracy": round(accuracy_score(yte, np.zeros(len(yte), int)), 4),
                    "auc": 0.5})

    print("Training final models...")
    skl = fit_sklearn(Xtr, ytr, best_lam)
    results.append(evaluate("sklearn", yte, skl.predict_proba(Xte)[:, 1]))

    w, b = train_manual(Xtr, ytr, best_lr_man, best_lam)
    results.append(evaluate("manual pytorch", yte, proba_manual(Xte, w, b)))

    model = train_standard(Xtr, ytr, best_lr_std, best_lam)
    results.append(evaluate("standard pytorch", yte, proba_standard(Xte, model)))

    res_df = pd.DataFrame(results)
    print("\nTest-set results:")
    print(res_df.to_string(index=False))

    # ---- Save everything the app needs (app loads the models, not cached preds) ----
    joblib.dump(pre, ART / "preprocessor.joblib")
    joblib.dump(skl, ART / "sklearn.joblib")
    torch.save({"w": w, "b": b}, ART / "manual.pt")
    torch.save(model.state_dict(), ART / "standard.pt")
    test[FEATURES + [TARGET]].to_csv(ART / "test.csv", index=False)
    res_df.to_csv(ART / "results.csv", index=False)
    print(f"\nSaved artifacts to {ART}")


if __name__ == "__main__":
    main()