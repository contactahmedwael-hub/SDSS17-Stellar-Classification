"""SDSS17 Astronomical Object Classification - interactive Streamlit app.

Run the pipeline first (./run_all.sh), then:   streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from common import CLASSES, FEATURE_SETS, MAGS, add_color_indices  # noqa: E402

ART = ROOT / "artifacts"
ASSETS = ROOT / "assets"
EDA, EM, CMP, MODELS = ART / "eda", ART / "emeans", ART / "comparison", ART / "models"

NAV_TABS = ["Overview", "EDA", "E-Means", "Models", "Predict", "MLflow"]
MONO_FONT = "CommitMono, JetBrains Mono, monospace"

st.set_page_config(
    page_title="SDSS17 Object Classifier · B2B Telemetry",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------------ theme & styling helpers
def init_state() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def is_dark() -> bool:
    return st.session_state.theme == "dark"


def get_palette() -> dict:
    if is_dark():
        return {
            "ink": "#f2f4f8",
            "muted": "#aeb7c7",
            "grid": "rgba(242, 244, 248, 0.10)",
            "classes": {"GALAXY": "#5b9bd5", "QSO": "#e06c75", "STAR": "#e5b54b"},
            "models": {"catboost": "#5b9bd5", "lightgbm": "#62c49e"},
        }
    return {
        "ink": "#141619",
        "muted": "#4b525c",
        "grid": "rgba(20, 22, 25, 0.08)",
        "classes": {"GALAXY": "#1e4e79", "QSO": "#b83b3b", "STAR": "#b57914"},
        "models": {"catboost": "#1e4e79", "lightgbm": "#1f7a5c"},
    }


def load_css_and_theme(css_path: Path) -> None:
    mode = st.session_state.theme
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    # Pure CSS :has(#sdss-mode-light | #sdss-mode-dark) marker (no <script> needed)
    st.markdown(f'<div id="sdss-mode-{mode}" style="display:none;"></div>', unsafe_allow_html=True)


def apply_chart_style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    pal = get_palette()
    layout_kwargs = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO_FONT, size=12, color=pal["ink"]),
        legend=dict(font=dict(color=pal["ink"]), bgcolor="rgba(0,0,0,0)"),
        height=height,
        margin=dict(l=18, r=18, t=44 if title else 22, b=18),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=13, color=pal["ink"]), x=0.0)
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=pal["grid"],
        zeroline=False,
        color=pal["muted"],
        title_font=dict(color=pal["ink"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=pal["grid"],
        zeroline=False,
        color=pal["muted"],
        title_font=dict(color=pal["ink"]),
    )
    return fig


def section_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="sdss-header">
            <div class="sdss-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p class="sdss-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar(final: dict) -> None:
    c_left, c_mid, c_right = st.columns([2.2, 5.6, 1.4], vertical_alignment="center")
    with c_left:
        st.markdown(
            '<div class="sdss-brand-mark"><span class="sdss-brand-dot"></span>SDSS17 · ASTRO ML</div>',
            unsafe_allow_html=True,
        )
    with c_mid:
        st.markdown(
            f'<div class="sdss-topbar-caption" style="text-align:center;">Active model: <code>{final["model"]}</code> · Split: 70 / 15 / 15</div>',
            unsafe_allow_html=True,
        )
    with c_right:
        label = "☀ Light mode" if is_dark() else "☾ Dark mode"
        if st.button(label, key="theme_toggle_btn", use_container_width=True):
            st.session_state.theme = "light" if is_dark() else "dark"
            st.rerun()


# ------------------------------------------------------------------ data & model helpers
def ready() -> bool:
    return (ART / "final_model.json").exists()


@st.cache_data
def read_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@st.cache_data
def read_csv(p: Path):
    return pd.read_csv(p)


@st.cache_resource
def load_model(cfg: str, name: str):
    if name == "catboost":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier()
        m.load_model(str(MODELS / f"{cfg}_catboost.cbm"))
        return m
    return joblib.load(MODELS / f"{cfg}_lightgbm.joblib")


@st.cache_resource
def load_emeans():
    return joblib.load(EM / "scaler_with_redshift.joblib"), np.load(EM / "centroids_with_redshift.npy")


def predict_proba(cfg: str, name: str, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(load_model(cfg, name).predict_proba(X[FEATURE_SETS[cfg]]))


def nearest_cluster(X: pd.DataFrame) -> np.ndarray:
    sc, C = load_emeans()
    Z = sc.transform(X[FEATURE_SETS["with_redshift"]]).astype("float32")
    d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def img(path: Path, caption: str | None = None) -> None:
    if Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {Path(path).name}")


# ------------------------------------------------------------------ pages
def page_overview(final: dict) -> None:
    pal = get_palette()
    section_header(
        "Sloan Digital Sky Survey · DR17",
        "Astronomical Object Classification",
        "Production-grade classification of SDSS sources into GALAXY, QSO, or STAR from photometry and redshift — "
        "benchmarking Faiss E-Means clustering against Optuna-tuned CatBoost and LightGBM tracked in MLflow.",
    )

    ov = read_json(EDA / "overview.json")
    t = final["test"]

    c = st.columns(5)
    c[0].metric("Final model", final["model"].upper())
    c[1].metric("Test Macro-F1", f"{t['macro_f1']:.4f}")
    c[2].metric("Test accuracy", f"{t['accuracy']:.4f}")
    c[3].metric("Rows used", f"{ov['rows_after_cleaning']:,}")
    c[4].metric("Features", len(final["features"]))

    st.divider()

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.subheader("Research & engineering pipeline")
        st.markdown(
            f"""
1. **Data & cleaning** — {ov['shape'][0]:,} rows and {ov['shape'][1]} columns initially, with zero missing values or exact duplicate rows;
   {ov['rows_removed_by_cleaning']} row removed due to the `-9999` sentinel in `u/g/z`.
2. **Feature engineering** — Magnitudes `u g r i z`, colour indices `u-g`, `g-r`, `r-i`, `i-z`, and `redshift` (evaluated with and without `redshift`).
   Metadata IDs (`obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `spec_obj_ID`, `plate`, `MJD`, `fiber_ID`) and sky coordinates (`alpha`, `delta`) are excluded.
3. **Stratified split** — 70 / 15 / 15 train / validation / test split; scalers and transforms are fit strictly on the training split.
4. **Unsupervised baseline** — Official Faiss E-Means (`faiss.contrib.e_means`, $k = 3$), assessed via Silhouette, Davies-Bouldin, ARI, and NMI.
5. **Supervised models** — CatBoost and LightGBM tuned independently with Optuna on validation Macro-F1, then evaluated once on the held-out test set.
6. **Experiment tracking** — All runs, nested Optuna trials, confusion matrices, timings, and artifacts logged to MLflow with the final model registered.
"""
        )

    with right:
        st.subheader("Class distribution")
        cc = pd.Series(ov["class_counts"]).reindex(CLASSES)
        fig = px.bar(
            x=cc.index,
            y=cc.values,
            color=cc.index,
            color_discrete_map=pal["classes"],
            labels={"x": "Object class", "y": "Objects"},
            text=[f"{v:,}" for v in cc.values],
        )
        fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(color=pal["ink"]))
        fig.update_layout(showlegend=False)
        st.plotly_chart(apply_chart_style(fig, height=350), use_container_width=True)


def page_eda() -> None:
    pal = get_palette()
    section_header(
        "Exploratory Analysis",
        "Dataset Structure & Photometric Signals",
        "Inspect cleaning decisions, split balance, feature distributions, correlations, outliers, and class separability.",
    )

    ov = read_json(EDA / "overview.json")
    tabs = st.tabs(["Data quality", "Distributions", "Correlations", "Outliers", "Class differences", "Interactive explorer"])

    with tabs[0]:
        c = st.columns(4)
        c[0].metric("Rows", f"{ov['shape'][0]:,}")
        c[1].metric("Columns", ov["shape"][1])
        c[2].metric("Missing values", ov["missing_values_total"])
        c[3].metric("Duplicate rows", ov["duplicate_rows"])

        st.markdown(
            f"""
* **Sentinel values (`-9999`)**: `{ov['sentinel_-9999_counts']}` → {ov['rows_removed_by_cleaning']} row removed.
* **`obj_ID` repetition**: Repeats across {ov['duplicate_obj_ID']:,} rows with distinct photometry and sometimes different classes (19-digit IDs exceed `float64` precision), so they are not duplicate observations and `obj_ID` is excluded from features.
"""
        )

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Split sizes**")
            st.json(ov["split_sizes"])
            st.markdown("**Split class share (%)**")
            st.dataframe(pd.DataFrame(ov["split_class_share_pct"]).rename_axis("class (%)"), use_container_width=True)
        with col_b:
            st.markdown("**Column dtypes**")
            st.dataframe(pd.DataFrame({"dtype": ov["dtypes"]}), use_container_width=True, height=320)

        st.divider()
        img(EDA / "class_distribution.png", "Class distribution across splits")
        st.markdown("**Summary statistics**")
        st.dataframe(read_csv(EDA / "describe.csv"), use_container_width=True)

    with tabs[1]:
        img(EDA / "feature_distributions.png", "Raw feature distributions")
        img(EDA / "color_distributions.png", "Colour index distributions")
        img(EDA / "color_color_redshift.png", "Colour-colour and redshift relationships")

    with tabs[2]:
        img(EDA / "correlation.png", "Feature correlation matrix")
        st.caption(
            "Magnitudes (u, g, r, i, z) correlate strongly with one another because they share overall source brightness; "
            "colour indices isolate relative spectral slope."
        )

    with tabs[3]:
        img(EDA / "boxplots_outliers.png", "IQR outlier boxplots")
        st.dataframe(read_csv(EDA / "outliers_iqr.csv"), use_container_width=True)
        st.caption(
            "IQR-flagged points are retained because extreme colours or high redshifts carry genuine astrophysical signal "
            "(such as high-redshift quasars), and gradient-boosted trees handle skewed tails naturally."
        )

    with tabs[4]:
        st.markdown("**Kruskal-Wallis H-test across classes** (higher $H$ indicates stronger separation across class medians):")
        st.dataframe(read_csv(EDA / "kruskal_by_class.csv"), use_container_width=True)
        st.markdown("**Class medians**")
        st.dataframe(read_csv(EDA / "class_medians.csv"), use_container_width=True)

    with tabs[5]:
        s = read_csv(ART / "data_sample.csv")
        feats = FEATURE_SETS["with_redshift"]
        c1, c2, c3 = st.columns([1, 1, 0.5])
        x = c1.selectbox("X axis", feats, index=feats.index("g-r"), key="eda_x")
        y = c2.selectbox("Y axis", feats, index=feats.index("u-g"), key="eda_y")
        logx = c3.checkbox("Log X", value=False, key="eda_logx")

        fig_scatter = px.scatter(
            s,
            x=x,
            y=y,
            color="class",
            color_discrete_map=pal["classes"],
            opacity=0.55,
            log_x=bool(logx and (s[x] > 0).all()),
        )
        st.plotly_chart(apply_chart_style(fig_scatter, height=500, title=f"{y} vs. {x}"), use_container_width=True)

        f = st.selectbox("Histogram feature", feats, index=feats.index("redshift"), key="eda_hist_f")
        fig_hist = px.histogram(
            s,
            x=f,
            color="class",
            color_discrete_map=pal["classes"],
            barmode="overlay",
            opacity=0.62,
            nbins=80,
        )
        st.plotly_chart(apply_chart_style(fig_hist, height=370, title=f"Distribution of {f}"), use_container_width=True)
        st.caption(f"Interactive plots use a stratified sample of {len(s):,} rows for responsive rendering.")


def page_emeans() -> None:
    section_header(
        "Unsupervised Learning",
        "E-Means Clustering (k = 3)",
        "Evaluate Faiss exponential-moving-average mini-batch clustering against ground-truth astronomical classes.",
    )

    S = read_json(EM / "summary.json")
    st.info(
        f"**Implementation:** `{S['implementation']}` · **Epochs:** {S['nepochs']} · **Scaler:** {S['scaler']}. "
        "E-Means is Faiss's exponential-moving-average variant rather than standard scikit-learn K-Means."
    )

    m = pd.DataFrame(S["metrics_with_redshift"]).T
    mz = pd.DataFrame(S["metrics_without_redshift"]).T

    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        st.subheader("Metrics (with redshift)")
        st.dataframe(m.style.format("{:.4f}"), use_container_width=True)
    with c_right:
        st.subheader("Redshift ablation (test split)")
        st.dataframe(
            pd.concat({"with redshift": m.loc[["test"]], "without redshift": mz.loc[["test"]]}).style.format("{:.4f}"),
            use_container_width=True,
        )

    st.subheader("Cluster projections & composition")
    img(EM / "clusters_pca.png", "PCA projection colored by E-Means cluster assignment")
    img(EM / "cluster_composition.png", "True class composition within each cluster")

    c_sz, c_cent = st.columns([1, 2], gap="large")
    with c_sz:
        st.markdown("**Cluster sizes**")
        st.dataframe(pd.DataFrame(S["cluster_sizes"]), use_container_width=True)
    with c_cent:
        st.markdown("**Centroids (original units)**")
        st.dataframe(read_csv(EM / "centroids_original_units.csv"), use_container_width=True)

    st.subheader("k-sweep (secondary study)")
    img(EM / "k_sweep.png", "Clustering quality across cluster counts")
    st.dataframe(read_csv(EM / "k_sweep.csv").round(4), use_container_width=True)

    st.subheader("Clustering vs. supervised classification")
    st.markdown(
        """
| Dimension | Unsupervised E-Means | Supervised CatBoost / LightGBM |
| --- | --- | --- |
| Target labels | Unused during fitting | Directly optimizes class separation |
| Output | Cluster ID with no built-in semantic label | Class prediction with probability estimates |
| Evaluation | Silhouette, Davies-Bouldin, ARI, NMI | Accuracy, Macro-F1, precision, recall, confusion matrix |
| Behaviour | Tracks dominant variance / compact neighbourhoods | Carves non-linear decision boundaries across features |

Because E-Means follows variance structure in scaled space rather than class boundaries—and galaxies, quasars, and stars overlap in colour space—its ARI/NMI scores are modest compared to supervised models that leverage non-linear `redshift` interactions.
"""
    )


def page_models(final: dict) -> None:
    pal = get_palette()
    section_header(
        "Supervised Benchmarks",
        "CatBoost vs. LightGBM",
        "Compare Optuna-tuned gradient boosting models across feature sets, splits, confusion matrices, and redshift ablations.",
    )

    mt = read_csv(CMP / "metrics_all.csv")
    tune = read_csv(CMP / "tuning_summary.csv")

    st.success(f"Selected final model: **{final['model'].title()}** (highest test Macro-F1 with `redshift` enabled).")
    for line in final["explanation"]:
        st.markdown(f"- {line}")

    tabs = st.tabs(["Test & val metrics", "Confusion matrices", "Optuna tuning", "Redshift impact", "Feature importance"])
    cfg_names = {"with_redshift": "With redshift", "without_redshift": "Without redshift"}

    with tabs[0]:
        c1, c2 = st.columns(2)
        cfg = c1.radio("Feature set", list(cfg_names), format_func=cfg_names.get, horizontal=True, key="m_cfg")
        split = c2.radio("Split", ["test", "val"], horizontal=True, key="m_split")

        t = mt[(mt.config == cfg) & (mt.split == split)].drop(columns=["config", "split"]).set_index("model")
        max_bg = "#1f3d2b" if is_dark() else "#d8ebd8"
        st.dataframe(t.style.format("{:.4f}").highlight_max(axis=0, color=max_bg), use_container_width=True)

        if cfg == "with_redshift" and split == "test":
            img(CMP / "metrics_comparison.png", "Side-by-side test metrics")

        long = t[["accuracy", "macro_f1", "weighted_f1", "precision_macro", "recall_macro"]].reset_index().melt("model")
        fig_m = px.bar(
            long,
            x="variable",
            y="value",
            color="model",
            barmode="group",
            color_discrete_map=pal["models"],
        )
        fig_m.update_yaxes(range=[max(0.0, float(long["value"].min()) - 0.02), 1.0])
        st.plotly_chart(apply_chart_style(fig_m, height=360, title=f"{cfg_names[cfg]} · {split} split"), use_container_width=True)

    with tabs[1]:
        cfg = st.radio("Feature set", list(cfg_names), format_func=cfg_names.get, horizontal=True, key="cmcfg")
        cols = st.columns(2, gap="large")
        for col, name in zip(cols, ["catboost", "lightgbm"]):
            with col:
                st.markdown(f"### {name.title()}")
                img(CMP / f"{cfg}_{name}" / "confusion_matrix_test.png", f"{name.title()} test confusion matrix")
                r = read_json(CMP / f"{cfg}_{name}" / "result.json")
                st.dataframe(pd.DataFrame(r["classification_report_test"]).T.round(4), use_container_width=True)

    with tabs[2]:
        st.markdown("**Optuna best hyperparameters & scores**")
        st.dataframe(tune.set_index(["config", "model"]).T, use_container_width=True)

        cfg = st.radio("Optuna study", list(cfg_names), format_func=cfg_names.get, horizontal=True, key="optcfg")
        cols = st.columns(2, gap="large")
        for col, name in zip(cols, ["catboost", "lightgbm"]):
            with col:
                st.markdown(f"### {name.title()}")
                d = CMP / f"{cfg}_{name}"
                img(d / "optuna_history.png", "Optimization history")
                if (d / "optuna_param_importance.png").exists():
                    img(d / "optuna_param_importance.png", "Hyperparameter importance")
                with st.expander("View all Optuna trials"):
                    st.dataframe(read_csv(d / "optuna_trials.csv"), use_container_width=True)

        st.caption("Objective: validation Macro-F1. The test set remained untouched throughout hyperparameter search.")

    with tabs[3]:
        img(CMP / "redshift_impact.png", "Performance with vs. without redshift")
        z = mt[mt.split == "test"].pivot(index="model", columns="config", values="macro_f1")
        z["macro_f1_drop"] = z["with_redshift"] - z["without_redshift"]
        st.dataframe(z.style.format("{:.4f}"), use_container_width=True)
        st.markdown("**Per-class F1 breakdown**")
        st.dataframe(read_csv(CMP / "per_class_f1.csv").round(4), use_container_width=True)
        st.caption("Without redshift, photometric colours alone leave substantial overlap—especially for QSOs across redshift ranges.")

    with tabs[4]:
        cols = st.columns(2, gap="large")
        for col, name in zip(cols, ["catboost", "lightgbm"]):
            with col:
                st.markdown(f"### {name.title()}")
                img(CMP / f"with_redshift_{name}" / "feature_importance.png", f"{name.title()} feature importance")


def page_predict(final: dict) -> None:
    pal = get_palette()
    section_header(
        "Inference Workbench",
        "Predict Object Class",
        "Score individual astronomical sources or upload a batch CSV to generate class predictions and probabilities.",
    )

    ctrl_left, ctrl_right = st.columns([1, 1.4], gap="large")
    use_z = ctrl_left.toggle(
        "Include redshift (recommended)",
        value=True,
        help="Disable to run the photometry-only models.",
        key="pred_use_z",
    )
    cfg = "with_redshift" if use_z else "without_redshift"
    default = final["model"] if use_z else final["photometry_only_best"]
    choice = ctrl_right.selectbox(
        "Classifier",
        ["catboost", "lightgbm"],
        index=["catboost", "lightgbm"].index(default),
        format_func=lambda n: f"{n.title()} (selected final)" if n == default else n.title(),
        key="pred_choice",
    )

    tab1, tab2 = st.tabs(["Single object", "Batch CSV"])
    sample = read_csv(ART / "data_sample.csv")

    with tab1:
        if "vals" not in st.session_state:
            st.session_state.vals = dict(u=22.0, g=21.0, r=20.5, i=20.2, z=20.0, redshift=0.4)

        preset_cols = st.columns([1, 1, 1, 2])
        for cls, col in zip(CLASSES, preset_cols[:3]):
            if col.button(f"Sample {cls}", key=f"sample_{cls}", use_container_width=True):
                row = sample[sample["class"] == cls].sample(1).iloc[0]
                st.session_state.vals = {k: float(row[k]) for k in MAGS + ["redshift"]}

        v = st.session_state.vals
        input_cols = st.columns(6)
        new_vals = {}
        for col, k in zip(input_cols, MAGS + ["redshift"]):
            new_vals[k] = col.number_input(
                k,
                value=float(v[k]),
                format="%.4f",
                step=0.1 if k != "redshift" else 0.01,
                key=f"field_{k}",
            )
        st.session_state.vals = new_vals

        if any(not (10.0 < new_vals[k] < 35.0) for k in MAGS):
            st.warning("One or more magnitudes sit outside typical SDSS range (10–35).")
        if new_vals["redshift"] < -0.01:
            st.warning("Negative redshift is typically only observed as minor measurement noise on nearby stars.")

        X = add_color_indices(pd.DataFrame([new_vals]))
        probs = predict_proba(cfg, choice, X)[0]
        pred = CLASSES[int(probs.argmax())]

        m_cols = st.columns(3 if use_z else 2)
        m_cols[0].metric("Predicted class", pred)
        m_cols[1].metric("Top confidence", f"{probs.max():.1%}")
        if use_z:
            m_cols[2].metric("Nearest E-Means cluster", f"Cluster {int(nearest_cluster(X)[0])}")

        fig_p = go.Figure(
            go.Bar(
                x=CLASSES,
                y=probs,
                marker_color=[pal["classes"][c] for c in CLASSES],
                text=[f"{x:.1%}" for x in probs],
                textposition="outside",
                textfont=dict(color=pal["ink"]),
            )
        )
        fig_p.update_yaxes(range=[0, 1.08], title="Probability")
        st.plotly_chart(apply_chart_style(fig_p, height=320, title="Class probability distribution"), use_container_width=True)

        st.markdown("**Derived feature vector**")
        st.dataframe(X[FEATURE_SETS[cfg]].round(4), use_container_width=True)

        with st.expander("Compare CatBoost vs. LightGBM on this object"):
            comp = pd.DataFrame(
                {n.title(): predict_proba(cfg, n, X)[0] for n in ["catboost", "lightgbm"]},
                index=CLASSES,
            )
            st.dataframe(comp.style.format("{:.3f}"), use_container_width=True)

    with tab2:
        need = MAGS + (["redshift"] if use_z else [])
        st.markdown("Required CSV columns: " + ", ".join(f"`{n}`" for n in need))
        up = st.file_uploader("Upload CSV", type="csv", key="batch_csv_uploader")

        if up is not None:
            df = pd.read_csv(up)
            missing = [c for c in need if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                bad = (df[MAGS] < -1000).any(axis=1)
                if bad.any():
                    st.warning(f"Skipped {int(bad.sum())} row(s) containing `-9999` sentinel values.")
                ok = df.loc[~bad].copy()
                if "redshift" not in ok.columns:
                    ok["redshift"] = 0.0

                X_batch = add_color_indices(ok)
                P_batch = predict_proba(cfg, choice, X_batch)

                out = ok.copy()
                out["predicted_class"] = [CLASSES[i] for i in P_batch.argmax(1)]
                for i, c in enumerate(CLASSES):
                    out[f"p_{c}"] = P_batch[:, i].round(4)

                st.dataframe(out.head(500), use_container_width=True)

                fig_pie = px.pie(
                    out,
                    names="predicted_class",
                    color="predicted_class",
                    color_discrete_map=pal["classes"],
                )
                st.plotly_chart(apply_chart_style(fig_pie, height=340, title="Predicted class breakdown"), use_container_width=True)

                if "class" in out.columns:
                    from sklearn.metrics import accuracy_score, f1_score

                    acc = accuracy_score(out["class"], out["predicted_class"])
                    mf1 = f1_score(out["class"], out["predicted_class"], average="macro")
                    st.info(f"Ground-truth `class` column detected — Accuracy: **{acc:.4f}**, Macro-F1: **{mf1:.4f}**.")

                st.download_button(
                    "Download predictions CSV",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name="sdss17_predictions.csv",
                    mime="text/csv",
                )
        else:
            st.download_button(
                "Download sample input CSV (20 rows)",
                sample.drop(columns="class").head(20).to_csv(index=False).encode("utf-8"),
                file_name="sdss17_sample_input.csv",
                mime="text/csv",
            )


def page_mlflow() -> None:
    section_header(
        "Reproducibility & Registry",
        "MLflow Experiment Tracking",
        "Inspect logged experiments, nested Optuna trial runs, and registered model metadata.",
    )

    st.markdown(
        "Tracked experiments include `SDSS17-EDA`, `SDSS17-EMeans`, `SDSS17-CatBoost`, `SDSS17-LightGBM` "
        "(with nested Optuna trial runs), and `SDSS17-Final` (`sdss17-final-classifier`)."
    )
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")

    p = ART / "mlflow_runs.csv"
    if p.exists():
        df = read_csv(p).dropna(axis=1, how="all")
        st.dataframe(df, use_container_width=True, height=420)
    else:
        st.warning("`artifacts/mlflow_runs.csv` not found — run the pipeline first.")

    st.markdown("**Final model manifest (`final_model.json`)**")
    st.json(read_json(ART / "final_model.json"))


# ------------------------------------------------------------------ main
init_state()
load_css_and_theme(ASSETS / "style.css")

if not ready():
    st.error("Artifacts not found. Run `./run_all.sh` (or `python src/eda.py && python src/emeans_run.py && python src/train_models.py`) first.")
    st.stop()

final = read_json(ART / "final_model.json")
render_top_bar(final)

st.markdown('<div class="sdss-main-tabs">', unsafe_allow_html=True)
t_overview, t_eda, t_emeans, t_models, t_predict, t_mlflow = st.tabs(NAV_TABS)
st.markdown("</div>", unsafe_allow_html=True)

with t_overview:
    page_overview(final)
with t_eda:
    page_eda()
with t_emeans:
    page_emeans()
with t_models:
    page_models(final)
with t_predict:
    page_predict(final)
with t_mlflow:
    page_mlflow()

