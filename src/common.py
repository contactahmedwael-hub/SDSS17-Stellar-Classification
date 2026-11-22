"""Shared constants and helpers for the SDSS17 project (used by the pipeline and the Streamlit app)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "star_classification.csv"
ART = ROOT / "artifacts"
SEED = 42
CLASSES = ["GALAXY", "QSO", "STAR"]           # fixed label order -> 0, 1, 2
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
MAGS = ["u", "g", "r", "i", "z"]
COLORS = ["u-g", "g-r", "r-i", "i-z"]
# Identifiers / observation metadata: never used as model inputs
ID_COLS = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID",
           "plate", "MJD", "fiber_ID"]
SKY_COLS = ["alpha", "delta"]                 # sky position = survey geometry, not object physics
FEATURE_SETS = {
    "with_redshift": MAGS + COLORS + ["redshift"],
    "without_redshift": MAGS + COLORS,
}


def add_color_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["u-g"] = df["u"] - df["g"]
    df["g-r"] = df["g"] - df["r"]
    df["r-i"] = df["r"] - df["i"]
    df["i-z"] = df["i"] - df["z"]
    return df


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame):
    """Row-level cleaning that needs no fitted statistics (safe before the split).
    SDSS uses -9999 as a 'no measurement' sentinel in u/g/z; those rows are removed."""
    bad = (df[MAGS] < -1000).any(axis=1)
    return df.loc[~bad].reset_index(drop=True), int(bad.sum())


def make_splits(df: pd.DataFrame):
    """Stratified 70/15/15 split. Returns dict name -> DataFrame (with engineered features + 'y')."""
    df = add_color_indices(df)
    df["y"] = df["class"].map(CLASS_TO_ID)
    train, tmp = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["y"], random_state=SEED)
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def compute_metrics(y_true, y_pred) -> dict:
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro"),
           "weighted_f1": f1_score(y_true, y_pred, average="weighted")}
    for avg in ("macro", "weighted"):
        out[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        out[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
    for c, v in zip(CLASSES, f1_score(y_true, y_pred, average=None)):
        out[f"f1_{c}"] = float(v)
    return {k: float(v) for k, v in out.items()}


def conf_mat(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))


def features_from_raw(u, g, r, i, z, redshift) -> pd.DataFrame:
    """Build the model input frame from raw photometry (used by the app for predictions)."""
    df = pd.DataFrame({"u": u, "g": g, "r": r, "i": i, "z": z, "redshift": redshift})
    return add_color_indices(df)


def mlflow_experiment(name: str):
    """set_experiment, but create new experiments with artifacts under <project>/mlruns (independent of the cwd)."""
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns" / name).as_uri())
    return mlflow.set_experiment(name)


# ----------------------------------------------------------------------------- plotting helpers
# (kept here so the notebooks stay focused on the ML logic)
PALETTE = {"GALAXY": "#4c78a8", "QSO": "#e45756", "STAR": "#f2b134"}


def plot_cm(cm, title, path=None):
    """Confusion matrix as counts and row-normalised (= per-class recall)."""
    import matplotlib.pyplot as plt, seaborn as sns
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[0], cbar=False)
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax[1], cbar=False)
    ax[0].set_title(f"{title} - counts"); ax[1].set_title("row-normalised (recall)")
    for a in ax: a.set_xlabel("predicted"); a.set_ylabel("true")
    plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    plt.show()


def plot_history(df, title, path=None):
    """Optuna optimisation history from study.trials_dataframe()."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["number"], df["value"], s=25, alpha=.7, label="trial")
    ax.plot(df["number"], df["value"].cummax(), color="C3", label="best so far")
    ax.set(xlabel="trial", ylabel="validation Macro-F1", title=title); ax.legend()
    lo = max(df["value"].min(), df["value"].max() - 0.05); ax.set_ylim(lo - 0.002, df["value"].max() + 0.002)
    plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    plt.show()


def plot_param_importance(study, title, path=None):
    import matplotlib.pyplot as plt, optuna
    try:
        imp = optuna.importance.get_param_importances(study)
    except Exception:
        return None
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); s.plot.barh(ax=ax, color="#4c78a8")
    ax.set_title(title); ax.set_xlabel("importance (fANOVA)"); plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    plt.show()
    return s.to_dict()


def plot_feature_importance(imp, title, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4)); imp[::-1].plot.barh(ax=ax, color="#59a14f")
    ax.set_title(title); plt.tight_layout()
    if path: plt.savefig(path, dpi=130)
    plt.show()
