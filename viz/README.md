# viz

Plotting utilities covering training dynamics, representation geometry, model evaluation, mechanistic interpretability, safety evals, and infrastructure profiling.

## Dependencies

```bash
pip install plotly kaleido seaborn umap-learn scikit-learn scipy pandas numpy matplotlib
```

`kaleido` is only needed for PNG export.

---

## The `save_path` convention

Every function has an optional `save_path` argument:

```python
# render inline in the notebook (default)
plot_loss_curves(steps, losses)

# save as 300 dpi PNG instead — directory is created if it doesn't exist
plot_loss_curves(steps, losses, save_path="figures/loss.png")
```

When `save_path` is set the figure is **not** shown inline — only written to disk.

---

## Functions by category

### 1. Training dynamics

#### `plot_loss_curves`
Training and validation loss over steps, with optional EMA smoothing.

```python
from ai_research_viz import plot_loss_curves

plot_loss_curves(
    steps=list(range(1000)),
    train_loss=train_losses,        # list of floats
    val_loss=val_losses,            # optional
    ema_alpha=0.95,                 # 0 = no smoothing
    log_scale=False,
    title="GPT-2 fine-tune",
)
```

---

#### `plot_gradient_norms`
Global or per-layer gradient L2 norm over steps. Pass a dict for per-layer breakdown.

```python
from ai_research_viz import plot_gradient_norms

# global norm
plot_gradient_norms(steps, global_norms, clip_value=1.0)

# per-layer
plot_gradient_norms(steps, {"attn": attn_norms, "mlp": mlp_norms}, log_scale=True)
```

---

#### `plot_weight_histograms`
Grid of per-layer weight distribution histograms.

```python
from ai_research_viz import plot_weight_histograms

plot_weight_histograms({
    "embed": model.embed.weight.detach().numpy(),
    "attn.W_Q": model.attn.W_Q.detach().numpy(),
    "mlp.W_in": model.mlp.W_in.detach().numpy(),
})
```

---

#### `plot_lr_schedule`
Learning rate over steps, with optional loss overlay on a secondary y-axis.

```python
from ai_research_viz import plot_lr_schedule

plot_lr_schedule(steps, lr_values, loss_values=train_losses)
```

---

### 2. Representation geometry

#### `plot_pca`
Linear PCA scatter (2D or 3D). Axis labels show explained variance.

```python
from ai_research_viz import plot_pca

plot_pca(
    embeddings,           # (N, D) np.ndarray
    labels=class_labels,  # optional (N,) for colouring
    hover_texts=tokens,   # optional (N,) strings on hover
    n_components=2,
)
```

---

#### `plot_umap`
2D UMAP scatter with configurable `n_neighbors` and `min_dist`.

```python
from ai_research_viz import plot_umap

plot_umap(embeddings, labels=labels, n_neighbors=30, min_dist=0.05)
```

---

#### `plot_tsne`
2D t-SNE scatter.

```python
from ai_research_viz import plot_tsne

plot_tsne(embeddings, labels=labels, perplexity=50)
```

---

#### `plot_similarity_heatmap`
Representational similarity / CKA matrix heatmap.

```python
from ai_research_viz import plot_similarity_heatmap

plot_similarity_heatmap(
    cka_matrix,                                      # (N, N) float matrix
    labels=[f"layer_{i}" for i in range(n_layers)],
    colorscale="RdBu",
)
```

---

### 3. Model evaluation

#### `plot_roc_curves`
Overlaid ROC curves for multiple models with AUC in the legend.

```python
from ai_research_viz import plot_roc_curves
from sklearn.metrics import roc_curve, roc_auc_score

fpr_a, tpr_a, _ = roc_curve(y_true, scores_a)
fpr_b, tpr_b, _ = roc_curve(y_true, scores_b)

plot_roc_curves(
    fpr_dict={"Model A": fpr_a, "Model B": fpr_b},
    tpr_dict={"Model A": tpr_a, "Model B": tpr_b},
    auc_dict={"Model A": roc_auc_score(y_true, scores_a),
              "Model B": roc_auc_score(y_true, scores_b)},
)
```

---

#### `plot_pr_curves`
Overlaid Precision-Recall curves with average precision in the legend.

```python
from ai_research_viz import plot_pr_curves

plot_pr_curves(
    precision_dict={"Model A": prec_a},
    recall_dict={"Model A": rec_a},
    ap_dict={"Model A": 0.81},
)
```

---

#### `plot_confusion_matrix`
Annotated confusion matrix. Row-normalised by default.

```python
from ai_research_viz import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, class_names=["safe", "unsafe", "ambiguous"], normalize=True)
```

---

#### `plot_calibration`
Reliability diagram (predicted probability vs actual fraction of positives).

```python
from ai_research_viz import plot_calibration
from sklearn.calibration import calibration_curve

frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
plot_calibration(
    prob_true_dict={"my model": frac_pos},
    prob_pred_dict={"my model": mean_pred},
)
```

---

#### `plot_radar`
Spider/radar chart for multi-benchmark comparison.

```python
from ai_research_viz import plot_radar

plot_radar({
    "GPT-4": {"MMLU": 0.86, "HumanEval": 0.67, "TruthfulQA": 0.59, "BBH": 0.83},
    "Claude": {"MMLU": 0.85, "HumanEval": 0.71, "TruthfulQA": 0.72, "BBH": 0.81},
})
```

---

### 4. Experiments & ablations

#### `plot_ablation_bars`
Horizontal bar chart with 95% CI error bars and an optional baseline reference line.

```python
from ai_research_viz import plot_ablation_bars

plot_ablation_bars(
    conditions=["full model", "- layer norm", "- pos encoding", "- attention"],
    means=[0.84, 0.79, 0.71, 0.52],
    errors=[0.02, 0.03, 0.02, 0.04],   # CI half-widths
    baseline_value=0.84,
    metric_name="accuracy",
)
```

---

#### `plot_parallel_coords`
Parallel coordinates for hyperparameter sweep results. Each line is a run; brush to filter.

```python
from ai_research_viz import plot_parallel_coords
import pandas as pd

sweep_df = pd.DataFrame({
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [16, 32, 32, 64],
    "warmup_steps": [100, 200, 100, 500],
    "val_loss": [1.82, 1.74, 1.79, 1.91],
})
plot_parallel_coords(sweep_df, color_col="val_loss")
```

---

#### `plot_scaling_law`
Log-log scaling law plot with automatic power-law fit overlay.

```python
from ai_research_viz import plot_scaling_law

plot_scaling_law(
    x_vals=compute_flops,
    y_vals=val_losses,
    x_label="compute (FLOPs)",
    y_label="cross-entropy loss",
    fit_power_law=True,
)
```

---

#### `plot_forest`
Forest plot of effect sizes with confidence intervals. Significant effects (CI excludes null) are highlighted in red.

```python
from ai_research_viz import plot_forest

plot_forest(
    conditions=["sys prompt A", "sys prompt B", "sys prompt C", "baseline"],
    effects=[0.12, -0.05, 0.31, 0.0],
    ci_low=[0.04, -0.18, 0.21, -0.06],
    ci_high=[0.20, 0.08, 0.41, 0.06],
    null_value=0.0,
    x_label="Δ refusal rate",
)
```

---

### 5. Mechanistic interpretability

#### `plot_attention_heatmap`
Attention weight heatmap for a single (layer, head).

```python
from ai_research_viz import plot_attention_heatmap

# attn shape: (n_heads, seq, seq) — specify head
plot_attention_heatmap(
    attention=cache["blocks.4.attn.hook_attn"][0],  # (n_heads, seq, seq)
    tokens=model.to_str_tokens(prompt),
    layer=4,
    head=2,
)
```

---

#### `plot_logit_lens`
Residual stream / logit lens heatmap showing top-1 predicted token at each layer.

```python
from ai_research_viz import plot_logit_lens

# string matrix approach
token_preds = np.array([
    [model.to_str_tokens([logits[layer, pos].argmax()])[0]
     for pos in range(seq_len)]
    for layer in range(n_layers)
])
plot_logit_lens(token_preds, tokens=vocab, layer_names=[f"L{i}" for i in range(n_layers)])
```

---

#### `plot_sae_feature_histogram`
SAE feature activation histogram with sparsity annotation and optional top-k token bar chart.

```python
from ai_research_viz import plot_sae_feature_histogram

plot_sae_feature_histogram(
    activations=feature_acts[:, 42],   # activations for feature 42 across all tokens
    feature_idx=42,
    top_k_tokens=[(" king", 8.3), (" queen", 7.1), (" throne", 6.9), (" crown", 6.2)],
)
```

---

#### `plot_probing_accuracy`
Linear probing accuracy per layer. Pass a dict for multiple concepts on one chart.

```python
from ai_research_viz import plot_probing_accuracy

plot_probing_accuracy(
    layer_names=[f"L{i}" for i in range(32)],
    accuracies={
        "sentiment": sentiment_accs,
        "syntax_depth": depth_accs,
        "coreference": coref_accs,
    },
    chance_level=0.5,
)
```

---

#### `plot_patching_heatmap`
Activation patching / causal intervention effect heatmap (layers × token positions).

```python
from ai_research_viz import plot_patching_heatmap

plot_patching_heatmap(
    patch_effects=patch_results,           # (n_layers, seq_len) float matrix
    x_labels=model.to_str_tokens(prompt),
    y_labels=[f"L{i}" for i in range(n_layers)],
    colorscale="RdBu",
)
```

---

### 6. Safety & alignment

#### `plot_behavior_heatmap`
Frequency matrix of model behaviours across prompt conditions.

```python
from ai_research_viz import plot_behavior_heatmap
import numpy as np

plot_behavior_heatmap(
    matrix=np.array([
        [0.92, 0.05, 0.03],
        [0.61, 0.28, 0.11],
        [0.14, 0.71, 0.15],
    ]),
    conditions=["benign", "jailbreak v1", "jailbreak v2"],
    behaviors=["refusal", "compliance", "hedge"],
)
```

---

#### `plot_refusal_curve`
Refusal rate vs perturbation/jailbreak strength, with optional confidence bands.

```python
from ai_research_viz import plot_refusal_curve

plot_refusal_curve(
    perturbation_levels=[0, 1, 2, 3, 4, 5],
    refusal_rates={
        "claude-3": [0.98, 0.95, 0.87, 0.72, 0.51, 0.30],
        "gpt-4":    [0.96, 0.90, 0.78, 0.60, 0.38, 0.19],
    },
)
```

---

#### `plot_beta_binomial_posteriors`
Overlaid Beta posterior PDFs for per-condition refusal/compliance rates. Dashed lines mark the posterior mode.

```python
from ai_research_viz import plot_beta_binomial_posteriors

# alpha = successes + prior_alpha, beta = failures + prior_beta
plot_beta_binomial_posteriors(
    conditions=["control", "CoT prompt", "adversarial"],
    alpha_params=[48 + 1, 61 + 1, 22 + 1],
    beta_params=[ 2 + 1,  9 + 1, 28 + 1],
)
```

---

#### `plot_steering_effect`
Output metric vs signed steering coefficient.

```python
from ai_research_viz import plot_steering_effect

plot_steering_effect(
    coefficients=[-3, -2, -1, 0, 1, 2, 3],
    metric_values={
        "refusal rate": [0.02, 0.08, 0.31, 0.67, 0.89, 0.96, 0.99],
        "coherence":    [0.91, 0.90, 0.88, 0.85, 0.79, 0.68, 0.50],
    },
)
```

---

### 7. Data & distributions

#### `plot_violin`
Violin + box + strip plot per condition. Good for comparing score distributions across prompt types.

```python
from ai_research_viz import plot_violin
import numpy as np

plot_violin({
    "baseline": np.random.normal(0.65, 0.12, 200),
    "augmented": np.random.normal(0.73, 0.09, 200),
    "adversarial": np.random.normal(0.48, 0.18, 200),
}, y_label="accuracy")
```

---

#### `plot_correlation_heatmap`
Feature correlation matrix with optional hierarchical clustering (seaborn clustermap).

```python
from ai_research_viz import plot_correlation_heatmap
import pandas as pd

plot_correlation_heatmap(features_df, method="spearman", cluster=True)
```

---

#### `plot_class_imbalance`
Sorted horizontal bar chart of class counts with percentage labels.

```python
from ai_research_viz import plot_class_imbalance

plot_class_imbalance({
    "safe": 14203,
    "unsafe": 1847,
    "ambiguous": 412,
})
```

---

### 8. Infrastructure

#### `plot_throughput_curve`
Samples/sec vs batch size (log x-axis), with optional memory overlay on a secondary y-axis.

```python
from ai_research_viz import plot_throughput_curve

plot_throughput_curve(
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
    throughputs={
        "A100 bf16": [48, 93, 181, 340, 612, 980, 1340, 1480],
        "A100 fp32": [24, 47,  91, 170, 302, 471,  601,  634],
    },
    memory_usage=[2.1, 2.4, 3.0, 4.2, 6.8, 12.1, 23.4, 45.9],
)
```

---

## Notes

**Library choices.** Most functions use Plotly (interactive, hover labels, zoomable). `plot_confusion_matrix`, `plot_violin`, and `plot_correlation_heatmap` use Seaborn/Matplotlib where those libraries produce cleaner results. The `save_path` convention is identical across both backends.

**PNG export.** Plotly uses `kaleido` under the hood (`fig.write_image(..., scale=2)` → 300 dpi equivalent). Matplotlib uses `savefig(..., dpi=300, bbox_inches="tight")`.

**UMAP reproducibility.** `plot_umap` and `plot_tsne` both set `random_state=42`. Pass different seeds by modifying the call if you want variance across runs.

**Beta-Binomial.** `plot_beta_binomial_posteriors` requires `scipy`. The prior is encoded directly in the alpha/beta parameters you pass — a uniform Beta(1,1) prior means `alpha = successes + 1`, `beta = failures + 1`.