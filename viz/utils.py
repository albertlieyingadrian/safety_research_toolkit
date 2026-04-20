"""
ai_research_viz.py
==================
Clean, publication-ready visualizations for AI research notebooks.

Usage
-----
from ai_research_viz import (
    plot_loss_curves, plot_gradient_norms, plot_weight_histograms,
    plot_lr_schedule, plot_pca, plot_umap, plot_tsne, plot_similarity_heatmap,
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix, plot_calibration,
    plot_radar, plot_ablation_bars, plot_parallel_coords, plot_scaling_law,
    plot_forest, plot_attention_heatmap, plot_logit_lens,
    plot_sae_feature_histogram, plot_probing_accuracy, plot_patching_heatmap,
    plot_behavior_heatmap, plot_refusal_curve, plot_beta_binomial_posteriors,
    plot_steering_effect, plot_violin, plot_correlation_heatmap,
    plot_class_imbalance, plot_throughput_curve,
)

Every function accepts an optional `save_path: str | None = None` argument.
- save_path=None  → renders inline in the notebook (default)
- save_path="out/fig.png"  → saves as PNG (300 dpi) and does NOT show inline

Dependencies
------------
pip install plotly kaleido seaborn umap-learn scikit-learn numpy pandas
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Shared theme helpers
# ---------------------------------------------------------------------------

PALETTE = px.colors.qualitative.Plotly  # vibrant, accessible

_PLOTLY_LAYOUT = dict(
    font_family="Inter, system-ui, sans-serif",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=50, b=60),
    colorway=PALETTE,
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    title_font_size=16,
    title_font_color="#1a1a1a",
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
)

sns.set_theme(style="whitegrid", font="DejaVu Sans")
_SNS_CONTEXT = dict(font_scale=1.1)
_FIG_DPI = 300


def _save_or_show_plotly(fig: go.Figure, save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.write_image(save_path, scale=2)
    else:
        fig.show()


def _save_or_show_mpl(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# 1. Training dynamics
# ---------------------------------------------------------------------------

def plot_loss_curves(
    steps: Sequence[int],
    train_loss: Sequence[float],
    val_loss: Optional[Sequence[float]] = None,
    ema_alpha: float = 0.0,
    log_scale: bool = False,
    title: str = "Loss curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training (and optional validation) loss over steps.

    Parameters
    ----------
    steps       : x-axis, e.g. list(range(len(losses)))
    train_loss  : raw training loss values
    val_loss    : optional validation loss values (same length as steps)
    ema_alpha   : EMA smoothing factor 0–1; 0 = no smoothing
    log_scale   : if True, use log y-axis
    title       : chart title
    save_path   : path to save PNG; None = show inline
    """
    def _ema(x, a):
        s, out = x[0], [x[0]]
        for v in x[1:]:
            s = a * s + (1 - a) * v
            out.append(s)
        return out

    fig = go.Figure()
    tr = list(train_loss)
    fig.add_trace(go.Scatter(x=list(steps), y=tr, name="train (raw)",
                             line=dict(color="#c8d4e3", width=1), mode="lines"))
    if ema_alpha > 0:
        fig.add_trace(go.Scatter(x=list(steps), y=_ema(tr, ema_alpha),
                                 name="train (EMA)", line=dict(color="#1f77b4", width=2), mode="lines"))
    if val_loss is not None:
        vl = list(val_loss)
        fig.add_trace(go.Scatter(x=list(steps), y=vl, name="val",
                                 line=dict(color="#ff7f0e", width=2, dash="dot"), mode="lines"))
    fig.update_layout(title=title, xaxis_title="step", yaxis_title="loss",
                      yaxis_type="log" if log_scale else "linear")
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_gradient_norms(
    steps: Sequence[int],
    grad_norms: Sequence[float] | dict[str, Sequence[float]],
    log_scale: bool = True,
    clip_value: Optional[float] = None,
    title: str = "Gradient norms",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot gradient norms over training steps.

    Parameters
    ----------
    steps       : x-axis steps
    grad_norms  : single list of global norms, or dict {layer_name: norms}
    log_scale   : y-axis log scale (usually sensible)
    clip_value  : if provided, draw a dashed reference line at this value
    save_path   : path to save PNG; None = show inline
    """
    fig = go.Figure()
    if isinstance(grad_norms, dict):
        for i, (name, norms) in enumerate(grad_norms.items()):
            fig.add_trace(go.Scatter(x=list(steps), y=list(norms),
                                     name=name, mode="lines",
                                     line=dict(width=1.5, color=PALETTE[i % len(PALETTE)])))
    else:
        fig.add_trace(go.Scatter(x=list(steps), y=list(grad_norms),
                                 name="grad norm", mode="lines",
                                 line=dict(color="#1f77b4", width=2)))
    if clip_value is not None:
        fig.add_hline(y=clip_value, line_dash="dash", line_color="#d62728",
                      annotation_text="clip", annotation_position="right")
    fig.update_layout(title=title, xaxis_title="step", yaxis_title="L2 norm",
                      yaxis_type="log" if log_scale else "linear")
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_weight_histograms(
    weights_by_layer: dict[str, np.ndarray],
    n_cols: int = 3,
    title: str = "Weight distributions",
    save_path: Optional[str] = None,
) -> None:
    """
    Grid of per-layer weight histograms.

    Parameters
    ----------
    weights_by_layer : dict {layer_name: 1D or ND weight array}
    n_cols           : columns in the subplot grid
    save_path        : path to save PNG; None = show inline
    """
    layers = list(weights_by_layer.keys())
    n = len(layers)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=layers)
    for i, (name, w) in enumerate(weights_by_layer.items()):
        row, col = divmod(i, n_cols)
        fig.add_trace(
            go.Histogram(x=w.flatten(), nbinsx=60, name=name,
                         marker_color=PALETTE[i % len(PALETTE)], showlegend=False),
            row=row + 1, col=col + 1,
        )
    fig.update_layout(title_text=title, height=280 * n_rows)
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_lr_schedule(
    steps: Sequence[int],
    lr_values: Sequence[float],
    loss_values: Optional[Sequence[float]] = None,
    title: str = "Learning rate schedule",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot LR over steps; optionally overlay loss on a secondary y-axis.

    save_path : path to save PNG; None = show inline
    """
    fig = make_subplots(specs=[[{"secondary_y": loss_values is not None}]])
    fig.add_trace(go.Scatter(x=list(steps), y=list(lr_values), name="learning rate",
                             line=dict(color="#1f77b4", width=2)), secondary_y=False)
    if loss_values is not None:
        fig.add_trace(go.Scatter(x=list(steps), y=list(loss_values), name="loss",
                                 line=dict(color="#ff7f0e", width=2, dash="dot")), secondary_y=True)
        fig.update_yaxes(title_text="loss", secondary_y=True)
    fig.update_yaxes(title_text="learning rate", secondary_y=False)
    fig.update_layout(title=title, xaxis_title="step")
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Representation geometry
# ---------------------------------------------------------------------------

def plot_pca(
    embeddings: np.ndarray,
    labels: Optional[Sequence] = None,
    hover_texts: Optional[Sequence[str]] = None,
    n_components: int = 2,
    title: str = "PCA",
    save_path: Optional[str] = None,
) -> None:
    """
    PCA scatter plot (2D or 3D).

    Parameters
    ----------
    embeddings   : (N, D) array
    labels       : (N,) categorical labels for colour
    hover_texts  : (N,) strings shown on hover
    n_components : 2 or 3
    save_path    : path to save PNG; None = show inline
    """
    from sklearn.decomposition import PCA as _PCA
    pca = _PCA(n_components=n_components)
    coords = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    df = pd.DataFrame(coords, columns=[f"PC{i+1}" for i in range(n_components)])
    if labels is not None:
        df["label"] = [str(l) for l in labels]
    if hover_texts is not None:
        df["hover"] = hover_texts
    color_col = "label" if labels is not None else None
    hover_col = "hover" if hover_texts is not None else None
    ax_titles = [f"PC{i+1} ({var[i]*100:.1f}%)" for i in range(n_components)]
    if n_components == 3:
        fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3",
                            color=color_col, hover_name=hover_col,
                            labels={"PC1": ax_titles[0], "PC2": ax_titles[1], "PC3": ax_titles[2]},
                            title=title)
    else:
        fig = px.scatter(df, x="PC1", y="PC2", color=color_col, hover_name=hover_col,
                         labels={"PC1": ax_titles[0], "PC2": ax_titles[1]}, title=title)
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_umap(
    embeddings: np.ndarray,
    labels: Optional[Sequence] = None,
    hover_texts: Optional[Sequence[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP",
    save_path: Optional[str] = None,
) -> None:
    """
    2D UMAP scatter. Requires `umap-learn`.

    save_path : path to save PNG; None = show inline
    """
    import umap as _umap
    reducer = _umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(embeddings)
    df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
    if labels is not None:
        df["label"] = [str(l) for l in labels]
    if hover_texts is not None:
        df["hover"] = hover_texts
    fig = px.scatter(df, x="UMAP1", y="UMAP2",
                     color="label" if labels is not None else None,
                     hover_name="hover" if hover_texts is not None else None,
                     title=title)
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_tsne(
    embeddings: np.ndarray,
    labels: Optional[Sequence] = None,
    hover_texts: Optional[Sequence[str]] = None,
    perplexity: float = 30.0,
    title: str = "t-SNE",
    save_path: Optional[str] = None,
) -> None:
    """
    2D t-SNE scatter.

    save_path : path to save PNG; None = show inline
    """
    from sklearn.manifold import TSNE
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
    df = pd.DataFrame(coords, columns=["tSNE1", "tSNE2"])
    if labels is not None:
        df["label"] = [str(l) for l in labels]
    if hover_texts is not None:
        df["hover"] = hover_texts
    fig = px.scatter(df, x="tSNE1", y="tSNE2",
                     color="label" if labels is not None else None,
                     hover_name="hover" if hover_texts is not None else None,
                     title=title)
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_similarity_heatmap(
    matrix: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    title: str = "Representational similarity",
    colorscale: str = "RdBu",
    save_path: Optional[str] = None,
) -> None:
    """
    Symmetric similarity / CKA matrix heatmap.

    Parameters
    ----------
    matrix     : (N, N) similarity matrix
    labels     : axis tick labels
    colorscale : Plotly colorscale name
    save_path  : path to save PNG; None = show inline
    """
    tick = list(labels) if labels is not None else [str(i) for i in range(matrix.shape[0])]
    fig = go.Figure(go.Heatmap(z=matrix, x=tick, y=tick, colorscale=colorscale,
                               zmin=-1, zmax=1))
    fig.update_layout(title=title, xaxis=dict(tickangle=45))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Model evaluation
# ---------------------------------------------------------------------------

def plot_roc_curves(
    fpr_dict: dict[str, np.ndarray],
    tpr_dict: dict[str, np.ndarray],
    auc_dict: Optional[dict[str, float]] = None,
    title: str = "ROC curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Overlaid ROC curves for multiple models.

    Parameters
    ----------
    fpr_dict  : {model_name: fpr array}
    tpr_dict  : {model_name: tpr array}
    auc_dict  : optional {model_name: auc_score} — added to legend label
    save_path : path to save PNG; None = show inline
    """
    fig = go.Figure()
    for i, name in enumerate(fpr_dict):
        auc_str = f"  (AUC={auc_dict[name]:.3f})" if auc_dict and name in auc_dict else ""
        fig.add_trace(go.Scatter(x=fpr_dict[name], y=tpr_dict[name],
                                 name=name + auc_str, mode="lines",
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="#aaa", width=1),
                             showlegend=False))
    fig.update_layout(title=title, xaxis_title="FPR", yaxis_title="TPR",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_pr_curves(
    precision_dict: dict[str, np.ndarray],
    recall_dict: dict[str, np.ndarray],
    ap_dict: Optional[dict[str, float]] = None,
    title: str = "Precision-Recall curves",
    save_path: Optional[str] = None,
) -> None:
    """
    Overlaid Precision-Recall curves.

    save_path : path to save PNG; None = show inline
    """
    fig = go.Figure()
    for i, name in enumerate(precision_dict):
        ap_str = f"  (AP={ap_dict[name]:.3f})" if ap_dict and name in ap_dict else ""
        fig.add_trace(go.Scatter(x=recall_dict[name], y=precision_dict[name],
                                 name=name + ap_str, mode="lines",
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)])))
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    Annotated confusion matrix heatmap.

    Parameters
    ----------
    cm           : (N, N) integer confusion matrix
    class_names  : list of class label strings
    normalize    : if True, normalise rows to 0–1
    save_path    : path to save PNG; None = show inline
    """
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"
    tick = list(class_names) if class_names else [str(i) for i in range(cm.shape[0])]
    with sns.axes_style("white"):
        with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
            fig, ax = plt.subplots(figsize=(max(6, len(tick)), max(5, len(tick) - 1)))
            sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                        xticklabels=tick, yticklabels=tick, ax=ax,
                        linewidths=0.5, linecolor="#e0e0e0")
            ax.set_title(title, fontsize=14, pad=12)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True", fontsize=11)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
    _save_or_show_mpl(fig, save_path)


def plot_calibration(
    prob_true_dict: dict[str, np.ndarray],
    prob_pred_dict: dict[str, np.ndarray],
    title: str = "Calibration (reliability diagram)",
    save_path: Optional[str] = None,
) -> None:
    """
    Reliability diagram comparing predicted confidence vs actual accuracy.

    Parameters
    ----------
    prob_true_dict : {model_name: fraction_of_positives per bin}
    prob_pred_dict : {model_name: mean_predicted_value per bin}
    save_path      : path to save PNG; None = show inline
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="#aaa", width=1),
                             name="perfect", showlegend=True))
    for i, name in enumerate(prob_true_dict):
        fig.add_trace(go.Scatter(x=prob_pred_dict[name], y=prob_true_dict[name],
                                 name=name, mode="lines+markers",
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                                 marker=dict(size=6)))
    fig.update_layout(title=title, xaxis_title="Mean predicted probability",
                      yaxis_title="Fraction of positives",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_radar(
    scores: dict[str, dict[str, float]],
    title: str = "Benchmark radar",
    save_path: Optional[str] = None,
) -> None:
    """
    Radar / spider chart for multi-benchmark model comparison.

    Parameters
    ----------
    scores    : {model_name: {benchmark_name: score}}
               All models must share the same benchmark keys.
    save_path : path to save PNG; None = show inline
    """
    benchmarks = list(next(iter(scores.values())).keys())
    fig = go.Figure()
    for i, (name, vals) in enumerate(scores.items()):
        v = [vals[b] for b in benchmarks]
        fig.add_trace(go.Scatterpolar(r=v + [v[0]], theta=benchmarks + [benchmarks[0]],
                                      name=name, fill="toself", opacity=0.6,
                                      line=dict(color=PALETTE[i % len(PALETTE)])))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True)))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Experiments & ablations
# ---------------------------------------------------------------------------

def plot_ablation_bars(
    conditions: Sequence[str],
    means: Sequence[float],
    errors: Optional[Sequence[float]] = None,
    baseline_value: Optional[float] = None,
    metric_name: str = "metric",
    title: str = "Ablation results",
    horizontal: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart with optional error bars and baseline reference line.

    Parameters
    ----------
    conditions      : condition / ablation names
    means           : metric values per condition
    errors          : half-widths of 95% CI (or std); optional
    baseline_value  : if provided, draw dashed reference line
    horizontal      : True = horizontal bars (easier to read with long names)
    save_path       : path to save PNG; None = show inline
    """
    df = pd.DataFrame({"condition": list(conditions), "value": list(means)})
    if errors is not None:
        df["error"] = list(errors)
    if horizontal:
        fig = go.Figure(go.Bar(
            y=df["condition"], x=df["value"],
            orientation="h",
            error_x=dict(array=df["error"].tolist(), visible=True) if errors is not None else None,
            marker_color=PALETTE[0],
        ))
        if baseline_value is not None:
            fig.add_vline(x=baseline_value, line_dash="dash", line_color="#d62728",
                          annotation_text="baseline", annotation_position="top right")
        fig.update_layout(title=title, xaxis_title=metric_name, yaxis_title="")
    else:
        fig = go.Figure(go.Bar(
            x=df["condition"], y=df["value"],
            error_y=dict(array=df["error"].tolist(), visible=True) if errors is not None else None,
            marker_color=PALETTE[0],
        ))
        if baseline_value is not None:
            fig.add_hline(y=baseline_value, line_dash="dash", line_color="#d62728",
                          annotation_text="baseline", annotation_position="right")
        fig.update_layout(title=title, yaxis_title=metric_name)
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_parallel_coords(
    df: pd.DataFrame,
    color_col: str,
    title: str = "Hyperparameter sweep",
    colorscale: str = "Viridis",
    save_path: Optional[str] = None,
) -> None:
    """
    Parallel coordinates plot for hyperparameter sweep exploration.

    Parameters
    ----------
    df         : DataFrame where each row is a run; columns are hparams + metric
    color_col  : column name to colour lines by (e.g. "val_loss")
    save_path  : path to save PNG; None = show inline
    """
    dims = []
    for col in df.columns:
        if col == color_col:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "category":
            cats = df[col].astype("category")
            dims.append(dict(label=col, values=cats.cat.codes,
                             tickvals=list(range(len(cats.cat.categories))),
                             ticktext=list(cats.cat.categories)))
        else:
            dims.append(dict(label=col, values=df[col]))
    fig = go.Figure(go.Parcoords(
        line=dict(color=df[color_col], colorscale=colorscale, showscale=True,
                  colorbar=dict(title=color_col)),
        dimensions=dims,
    ))
    fig.update_layout(title=title)
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_scaling_law(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    x_label: str = "compute (FLOPs)",
    y_label: str = "loss",
    series: Optional[dict[str, tuple[Sequence[float], Sequence[float]]]] = None,
    fit_power_law: bool = True,
    title: str = "Scaling law",
    save_path: Optional[str] = None,
) -> None:
    """
    Log-log scaling law plot with optional power-law fit.

    Parameters
    ----------
    x_vals, y_vals   : primary series
    series           : optional extra series {name: (x, y)} overlaid
    fit_power_law    : if True, fit and plot y = a * x^b
    save_path        : path to save PNG; None = show inline
    """
    fig = go.Figure()
    all_series = {"main": (list(x_vals), list(y_vals))}
    if series:
        all_series.update(series)
    for i, (name, (xs, ys)) in enumerate(all_series.items()):
        xs, ys = np.array(xs), np.array(ys)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+lines", name=name,
                                 marker=dict(size=6, color=PALETTE[i % len(PALETTE)]),
                                 line=dict(width=1.5, color=PALETTE[i % len(PALETTE)])))
        if fit_power_law:
            mask = (xs > 0) & (ys > 0)
            if mask.sum() > 2:
                coeffs = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
                b, log_a = coeffs
                x_fit = np.geomspace(xs[mask].min(), xs[mask].max(), 200)
                y_fit = np.exp(log_a) * x_fit ** b
                fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
                                         name=f"{name} fit (β={b:.2f})",
                                         line=dict(dash="dash", width=1,
                                                   color=PALETTE[i % len(PALETTE)])))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label,
                      xaxis_type="log", yaxis_type="log")
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_forest(
    conditions: Sequence[str],
    effects: Sequence[float],
    ci_low: Sequence[float],
    ci_high: Sequence[float],
    null_value: float = 0.0,
    x_label: str = "effect size",
    title: str = "Forest plot",
    save_path: Optional[str] = None,
) -> None:
    """
    Forest plot of effect sizes with confidence intervals.

    Parameters
    ----------
    conditions : condition / prompt / study names (y-axis)
    effects    : point estimates
    ci_low     : lower CI bounds
    ci_high    : upper CI bounds
    null_value : reference line position (0 for differences, 1 for ratios)
    save_path  : path to save PNG; None = show inline
    """
    n = len(conditions)
    y_pos = list(range(n))
    fig = go.Figure()
    for i, (cond, eff, lo, hi) in enumerate(zip(conditions, effects, ci_low, ci_high)):
        color = "#d62728" if lo > null_value or hi < null_value else "#1f77b4"
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[i, i], mode="lines",
            line=dict(color=color, width=2), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[eff], y=[i], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=cond, showlegend=False,
        ))
    fig.add_vline(x=null_value, line_dash="dash", line_color="#888", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis=dict(tickvals=y_pos, ticktext=list(conditions), autorange="reversed"),
        height=max(300, 30 * n + 80),
    )
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Mechanistic interpretability
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: Optional[Sequence[str]] = None,
    layer: Optional[int] = None,
    head: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Attention weight heatmap for a single (layer, head).

    Parameters
    ----------
    attention : (seq_len, seq_len) attention matrix, or (n_heads, seq, seq) —
                in which case you must specify `head`
    tokens    : list of token strings for axis labels
    layer     : label only (used in auto-title)
    head      : if attention is (n_heads, seq, seq), select this head
    save_path : path to save PNG; None = show inline
    """
    if attention.ndim == 3:
        if head is None:
            raise ValueError("pass `head` index when attention has shape (n_heads, seq, seq)")
        attention = attention[head]
    seq = attention.shape[0]
    tick = list(tokens) if tokens is not None else [str(i) for i in range(seq)]
    auto_title = title or f"Attention {'L'+str(layer)+' ' if layer is not None else ''}{'H'+str(head) if head is not None else ''}"
    fig = go.Figure(go.Heatmap(z=attention, x=tick, y=tick, colorscale="Blues",
                               zmin=0, zmax=1))
    fig.update_layout(title=auto_title,
                      xaxis=dict(title="key", tickangle=45),
                      yaxis=dict(title="query", autorange="reversed"))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_logit_lens(
    logit_lens_matrix: np.ndarray,
    tokens: Sequence[str],
    layer_names: Optional[Sequence[str]] = None,
    title: str = "Logit lens",
    save_path: Optional[str] = None,
) -> None:
    """
    Logit lens / residual stream heatmap showing top-token predictions per layer.

    Parameters
    ----------
    logit_lens_matrix : (n_layers, seq_len) — each cell is the top-1 token index
                        OR pass a string matrix directly (n_layers, seq_len) of token strings
    tokens            : vocabulary (for index→string lookup) OR the full string matrix
    layer_names       : y-axis labels
    save_path         : path to save PNG; None = show inline
    """
    if logit_lens_matrix.dtype.kind in ("U", "S", "O"):
        text_matrix = logit_lens_matrix.tolist()
    else:
        vocab = list(tokens)
        text_matrix = [[vocab[int(idx)] for idx in row] for row in logit_lens_matrix]
    n_layers, seq_len = len(text_matrix), len(text_matrix[0])
    y_labels = list(layer_names) if layer_names else [f"L{i}" for i in range(n_layers)]
    x_labels = [str(i) for i in range(seq_len)]
    z_numeric = [[hash(cell) % 256 for cell in row] for row in text_matrix]
    fig = go.Figure(go.Heatmap(
        z=z_numeric, text=text_matrix, texttemplate="%{text}",
        x=x_labels, y=y_labels, colorscale="Purples", showscale=False,
    ))
    fig.update_layout(title=title, xaxis_title="token position",
                      yaxis_title="layer", yaxis_autorange="reversed",
                      height=max(300, 28 * n_layers + 80))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_sae_feature_histogram(
    activations: np.ndarray,
    feature_idx: Optional[int] = None,
    top_k_tokens: Optional[list[tuple[str, float]]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    SAE feature activation histogram with optional top-k token annotations.

    Parameters
    ----------
    activations    : 1D array of activation values for one feature across examples
    feature_idx    : feature index (used in auto-title)
    top_k_tokens   : list of (token_str, activation_value) pairs to annotate
    save_path      : path to save PNG; None = show inline
    """
    auto_title = title or (f"SAE feature {feature_idx}" if feature_idx is not None else "SAE feature activations")
    sparsity = np.mean(activations == 0)
    fig = make_subplots(rows=1, cols=2 if top_k_tokens else 1,
                        subplot_titles=["activation distribution",
                                        "top-k activating tokens"] if top_k_tokens else ["activation distribution"])
    fig.add_trace(go.Histogram(x=activations, nbinsx=60, marker_color=PALETTE[0],
                               name="activations"), row=1, col=1)
    fig.add_annotation(text=f"sparsity: {sparsity:.2%}", xref="paper", yref="paper",
                       x=0.02, y=0.98, showarrow=False, font=dict(size=11, color="#555"))
    if top_k_tokens:
        toks, vals = zip(*top_k_tokens)
        fig.add_trace(go.Bar(x=list(vals), y=list(toks), orientation="h",
                             marker_color=PALETTE[1], name="top tokens"), row=1, col=2)
    fig.update_layout(title_text=auto_title, showlegend=False)
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_probing_accuracy(
    layer_names: Sequence[str],
    accuracies: Sequence[float] | dict[str, Sequence[float]],
    chance_level: float = 0.5,
    title: str = "Probing accuracy by layer",
    save_path: Optional[str] = None,
) -> None:
    """
    Linear probing accuracy across transformer layers.

    Parameters
    ----------
    layer_names : x-axis tick labels
    accuracies  : single list, or dict {concept_name: [acc per layer]}
    chance_level: draw a dashed reference line at this y value
    save_path   : path to save PNG; None = show inline
    """
    fig = go.Figure()
    if isinstance(accuracies, dict):
        for i, (name, accs) in enumerate(accuracies.items()):
            fig.add_trace(go.Scatter(x=list(layer_names), y=list(accs), name=name,
                                     mode="lines+markers",
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                                     marker=dict(size=6)))
    else:
        fig.add_trace(go.Scatter(x=list(layer_names), y=list(accuracies),
                                 name="probe accuracy", mode="lines+markers",
                                 line=dict(color=PALETTE[0], width=2), marker=dict(size=7)))
    fig.add_hline(y=chance_level, line_dash="dash", line_color="#aaa",
                  annotation_text="chance", annotation_position="right")
    fig.update_layout(title=title, xaxis_title="layer", yaxis_title="accuracy",
                      yaxis=dict(range=[0, 1]))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_patching_heatmap(
    patch_effects: np.ndarray,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    x_title: str = "token position",
    y_title: str = "layer",
    title: str = "Activation patching effects",
    colorscale: str = "RdBu",
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of causal intervention / activation patching effects.

    Parameters
    ----------
    patch_effects : (n_layers, n_positions) effect magnitude matrix
    colorscale    : diverging colorscale recommended (e.g. "RdBu")
    save_path     : path to save PNG; None = show inline
    """
    n_layers, n_pos = patch_effects.shape
    x_tick = list(x_labels) if x_labels else [str(i) for i in range(n_pos)]
    y_tick = list(y_labels) if y_labels else [f"L{i}" for i in range(n_layers)]
    abs_max = np.abs(patch_effects).max()
    fig = go.Figure(go.Heatmap(z=patch_effects, x=x_tick, y=y_tick,
                               colorscale=colorscale, zmin=-abs_max, zmax=abs_max))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title,
                      yaxis_autorange="reversed",
                      height=max(300, 28 * n_layers + 80))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Safety & alignment
# ---------------------------------------------------------------------------

def plot_behavior_heatmap(
    matrix: np.ndarray,
    conditions: Sequence[str],
    behaviors: Sequence[str],
    title: str = "Behavior frequency by condition",
    colorscale: str = "Blues",
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of behaviour frequencies across prompt conditions.

    Parameters
    ----------
    matrix     : (n_conditions, n_behaviors) frequency / rate matrix
    conditions : row labels
    behaviors  : column labels
    save_path  : path to save PNG; None = show inline
    """
    fig = go.Figure(go.Heatmap(
        z=matrix, x=list(behaviors), y=list(conditions),
        colorscale=colorscale, zmin=0, zmax=1,
        text=np.round(matrix, 2),
        texttemplate="%{text}",
    ))
    fig.update_layout(title=title, xaxis=dict(tickangle=30),
                      height=max(300, 35 * len(conditions) + 80))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_refusal_curve(
    perturbation_levels: Sequence[float],
    refusal_rates: Sequence[float] | dict[str, Sequence[float]],
    ci_bands: Optional[dict[str, tuple[Sequence[float], Sequence[float]]]] = None,
    x_label: str = "perturbation strength",
    title: str = "Refusal rate vs perturbation strength",
    save_path: Optional[str] = None,
) -> None:
    """
    Refusal rate vs jailbreak / perturbation intensity.

    Parameters
    ----------
    perturbation_levels : x-axis values
    refusal_rates       : single list, or dict {model: [rates]}
    ci_bands            : optional {model: (lower, upper)} CI band arrays
    save_path           : path to save PNG; None = show inline
    """
    fig = go.Figure()
    xl = list(perturbation_levels)
    if isinstance(refusal_rates, dict):
        for i, (name, rates) in enumerate(refusal_rates.items()):
            color = PALETTE[i % len(PALETTE)]
            if ci_bands and name in ci_bands:
                lo, hi = ci_bands[name]
                fig.add_trace(go.Scatter(
                    x=xl + xl[::-1], y=list(hi) + list(lo)[::-1],
                    fill="toself", fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"),
                    line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                ))
            fig.add_trace(go.Scatter(x=xl, y=list(rates), name=name, mode="lines+markers",
                                     line=dict(width=2, color=color), marker=dict(size=6)))
    else:
        fig.add_trace(go.Scatter(x=xl, y=list(refusal_rates), name="refusal rate",
                                 mode="lines+markers", line=dict(color=PALETTE[0], width=2),
                                 marker=dict(size=7)))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title="refusal rate",
                      yaxis=dict(range=[0, 1]))
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_beta_binomial_posteriors(
    conditions: Sequence[str],
    alpha_params: Sequence[float],
    beta_params: Sequence[float],
    x_range: tuple[float, float] = (0.0, 1.0),
    n_points: int = 300,
    title: str = "Beta-Binomial posteriors",
    save_path: Optional[str] = None,
) -> None:
    """
    Overlaid Beta posterior PDFs for per-condition refusal/compliance rates.

    Parameters
    ----------
    conditions   : condition names (one curve per condition)
    alpha_params : Beta α (successes + prior α)
    beta_params  : Beta β (failures + prior β)
    save_path    : path to save PNG; None = show inline
    """
    from scipy.stats import beta as beta_dist
    x = np.linspace(*x_range, n_points)
    fig = go.Figure()
    for i, (name, a, b) in enumerate(zip(conditions, alpha_params, beta_params)):
        y = beta_dist.pdf(x, a, b)
        mode_val = (a - 1) / (a + b - 2) if a > 1 and b > 1 else None
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines",
                                 line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                                 fill="tozeroy",
                                 fillcolor=PALETTE[i % len(PALETTE)].replace("rgb", "rgba").replace(")", ",0.08)")))
        if mode_val is not None:
            fig.add_vline(x=mode_val, line_dash="dot", line_width=1,
                          line_color=PALETTE[i % len(PALETTE)])
    fig.update_layout(title=title, xaxis_title="rate", yaxis_title="density")
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


def plot_steering_effect(
    coefficients: Sequence[float],
    metric_values: Sequence[float] | dict[str, Sequence[float]],
    metric_name: str = "metric",
    title: str = "Steering vector effect",
    save_path: Optional[str] = None,
) -> None:
    """
    Output metric vs steering coefficient (signed).

    Parameters
    ----------
    coefficients  : steering multiplier values (can be negative)
    metric_values : single list, or dict {metric_name: values}
    save_path     : path to save PNG; None = show inline
    """
    fig = go.Figure()
    xs = list(coefficients)
    if isinstance(metric_values, dict):
        for i, (name, vals) in enumerate(metric_values.items()):
            fig.add_trace(go.Scatter(x=xs, y=list(vals), name=name, mode="lines+markers",
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                                     marker=dict(size=7)))
    else:
        fig.add_trace(go.Scatter(x=xs, y=list(metric_values), name=metric_name,
                                 mode="lines+markers", line=dict(color=PALETTE[0], width=2),
                                 marker=dict(size=7)))
    fig.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
    fig.update_layout(title=title, xaxis_title="steering coefficient", yaxis_title=metric_name)
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Data & distributions
# ---------------------------------------------------------------------------

def plot_violin(
    data: dict[str, np.ndarray],
    x_label: str = "condition",
    y_label: str = "value",
    title: str = "Distributions",
    save_path: Optional[str] = None,
) -> None:
    """
    Violin + box + strip plot per condition.

    Parameters
    ----------
    data      : {condition_name: 1D array of values}
    save_path : path to save PNG; None = show inline
    """
    rows = []
    for cond, vals in data.items():
        for v in vals:
            rows.append({x_label: cond, y_label: float(v)})
    df = pd.DataFrame(rows)
    with sns.axes_style("whitegrid"):
        with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
            fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.5), 5))
            sns.violinplot(data=df, x=x_label, y=y_label, palette="tab10",
                           inner=None, ax=ax, alpha=0.7)
            sns.stripplot(data=df, x=x_label, y=y_label, color="0.3",
                          size=3, alpha=0.4, jitter=True, ax=ax)
            ax.set_title(title, fontsize=14)
            fig.tight_layout()
    _save_or_show_mpl(fig, save_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    cluster: bool = True,
    title: str = "Correlation heatmap",
    save_path: Optional[str] = None,
) -> None:
    """
    Annotated correlation matrix with optional hierarchical clustering.

    Parameters
    ----------
    df      : DataFrame of numeric features
    method  : 'pearson', 'spearman', or 'kendall'
    cluster : if True, use seaborn clustermap (reorders rows/cols)
    save_path : path to save PNG; None = show inline
    """
    corr = df.corr(method=method)
    with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
        if cluster:
            g = sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True,
                               fmt=".2f", linewidths=0.5, figsize=(max(8, len(corr)),
                                                                    max(7, len(corr))))
            g.fig.suptitle(title, y=1.02, fontsize=14)
            _save_or_show_mpl(g.fig, save_path)
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(corr)), max(7, len(corr))))
            sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                        linewidths=0.5, ax=ax)
            ax.set_title(title, fontsize=14)
            fig.tight_layout()
            _save_or_show_mpl(fig, save_path)


def plot_class_imbalance(
    class_counts: dict[str, int],
    title: str = "Class distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal sorted bar chart of class counts.

    Parameters
    ----------
    class_counts : {class_name: count}
    save_path    : path to save PNG; None = show inline
    """
    sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*sorted_items)
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    fig = go.Figure(go.Bar(
        y=list(names), x=list(counts), orientation="h",
        text=[f"{p:.1f}%" for p in pcts], textposition="outside",
        marker=dict(color=list(range(len(names))), colorscale="Teal"),
    ))
    fig.update_layout(title=title, xaxis_title="count", yaxis=dict(autorange="reversed"),
                      showlegend=False)
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Infrastructure
# ---------------------------------------------------------------------------

def plot_throughput_curve(
    batch_sizes: Sequence[int],
    throughputs: Sequence[float] | dict[str, Sequence[float]],
    memory_usage: Optional[Sequence[float]] = None,
    y_label: str = "samples / sec",
    title: str = "Throughput vs batch size",
    save_path: Optional[str] = None,
) -> None:
    """
    Throughput vs batch size, with optional memory overlay.

    Parameters
    ----------
    batch_sizes  : x-axis
    throughputs  : single list, or dict {config: [throughput]}
    memory_usage : optional memory (GB) series for secondary y-axis
    save_path    : path to save PNG; None = show inline
    """
    fig = make_subplots(specs=[[{"secondary_y": memory_usage is not None}]])
    xs = list(batch_sizes)
    if isinstance(throughputs, dict):
        for i, (name, vals) in enumerate(throughputs.items()):
            fig.add_trace(go.Scatter(x=xs, y=list(vals), name=name, mode="lines+markers",
                                     line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
                                     marker=dict(size=7)), secondary_y=False)
    else:
        fig.add_trace(go.Scatter(x=xs, y=list(throughputs), name="throughput",
                                 mode="lines+markers", line=dict(color=PALETTE[0], width=2),
                                 marker=dict(size=8)), secondary_y=False)
    if memory_usage is not None:
        fig.add_trace(go.Scatter(x=xs, y=list(memory_usage), name="memory (GB)",
                                 mode="lines+markers", line=dict(color="#d62728", width=2, dash="dot"),
                                 marker=dict(size=6, symbol="diamond")), secondary_y=True)
        fig.update_yaxes(title_text="memory (GB)", secondary_y=True)
    fig.update_yaxes(title_text=y_label, secondary_y=False)
    fig.update_layout(title=title, xaxis_title="batch size", xaxis_type="log")
    _apply_plotly_theme(fig)
    _save_or_show_plotly(fig, save_path)