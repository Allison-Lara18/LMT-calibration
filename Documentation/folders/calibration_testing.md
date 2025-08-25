# Calibration testing (notebook guide)

> **What this is.**
> These notebooks test and visualize **probability calibration** when the **true class probability** $p^*(x)$ is known (synthetic DGP). .

**Notebooks covered:**

* `known_prob_calibration.ipynb`
* `known_prob_calibration copy.ipynb` *(variant/sandbox of the same flow)*
* `visualization.ipynb`

**Exact implementation:** `lmt_pkg/calibration?functions.py`. 
**Theory note:** `Documentation/theory/calibration_implementation.md`

---

## Key Concepts

* **Why “known probability”?** With synthetic data you can write down $p^*(x)$. This lets you score calibration **without label noise**:
  * **True-Brier:** $\mathbb{E}\bigl[(\hat{p}(x)-p^*(x))^2\bigr]$
  * **True log-loss (cross-entropy to $p^*$):** $\mathbb{E}\bigl[-\,p^*(x)\,\log \hat{p}(x)\;-\;\bigl(1-\hat{p}(x)\bigr)\,\log\bigl(1-\hat{p}(x)\bigr)\bigr]$
  * **Population ECE:** bin by $\hat{p}$, compare $\overline{\hat{p}}$ vs. $\overline{p^*}$ per bin.
* **Empirical vs population metrics.** You can also compute the usual empirical metrics against labels $y\sim \operatorname{Bernoulli}\!\bigl(p^*(x)\bigr)$. Expect **higher variance** than the population (true-$p$) versions.
* **What you compare.** Any probabilistic classifier(s): trees/LMT, logistic regression. Same interface: produce $\hat{p}(x)$, compute metrics vs. $p^*$ and vs. labels, and plot.

---

## 1) `known_prob_calibration.ipynb` (ground-truth calibration)

1. **Data with known $p^*$**

   * Builds a synthetic DGP with **closed-form $p^*(x)$** (circle boundary via a sigmoid on a distance/radius function).
   * Exposes helpers to evaluate $p^*(X)$ on any split (train/val/test).

2. **Models under test**

   * Trains one or more probabilistic classifiers (baseline and/or your custom models).
     Each exposes `predict_proba(X)[:, 1]` → $\hat p(x)$.

3. **Calibration diagnosis and plots**

   * Calibration curves
   * True vs predicted probabilities scatter plots
   * Scatter plots by quantile groups


### Typical outputs

Figures: **reliability plots**, **$\hat p$ histograms**, **$p^*$ vs $\hat p$** scatter.

---

## 2) `known_prob_calibration copy.ipynb` (variant)

A **copy/variant** of the first notebook used to tweak DGP parameters (e.g., sharper boundary, different radius $r_0$, new seeds) and in absolute value dataset.
Produces the **same artifacts** with a different configuration.

---

## 3) `visualization.ipynb` 

* Loads or reproduces selected **checkpoints** from the calibration runs and generates **consistent, side-by-side** visuals:
  * **KDE diagrams**.
  * **Reliability diagrams** (fixed bins, matched axes).
  * **$\hat p$ histograms** and **density overlays**.
  * **$p^*$ vs $\hat p$** scatter with identity line and optional smoothing.
* Layout utilities to standardize fonts, margins, and figure sizes for slides/papers.

---

## Typical workflow (from the notebooks)

1. **Simulate data** with known $p^*(x)$; split into train/val/test and compute `p_true_* = p_star(X_*)`.
2. **Fit models** (baseline and candidate). Get `p_hat_* = model.predict_proba(X_*)[:,1]`.
5. **Plot calibration**: reliability curves (true vs empirical), histograms, $p^*$–$\hat p$ scatter.
6. **Export figures** and (**optionally**) collect them in `visualization.ipynb` for the final composite.

---

## API cheat-sheet (as used in the notebooks)

**Example**

```python
# Data for circles dataset
n, alpha, r0 = 5000, 2, 1.5
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
r2 = X1**2 + X2**2
p = 1 / (1 + np.exp(-alpha * (r2 - r0)))
y = np.random.binomial(1, p)
X = np.column_stack((X1, X2))   # now X has shape (n,2)
X_ext = np.column_stack((X, r2))  # shape (n,3)
X_train, X_test, y_train, y_test = train_test_split(X_ext, y, test_size=0.3)

# Fitting LMT with original tree construction for original and extended dataset
original_models2 = []
configs = [
    ('shallow', False, "shallow (original)"),
    ('regular', False, "regular (original)"),
    ('overfit', False, "overfit (original)"),
    ('regular', True,  "pruned (original)")
]
for size, pruning, label in tqdm(configs, desc="Training original models"):
        clf2, nodes2 = lmt.fit_logistic_model_tree_v2(X_train[:, :2], y_train, size=size, pruning=pruning)
        original_models2.append((clf2, nodes2))
extended_models2=[]
for size, pruning, label in tqdm(configs, desc="Training extended models"):
        clf2, nodes2 = lmt.fit_logistic_model_tree_v2(X_train, y_train, size=size, pruning=pruning)
        extended_models2.append((clf2, nodes2))

# Visualization of trees, decision borders and probability surfaces
lmt.compare_lmt_variants_multiclass(
    X_train=X_train[:, :2],   # training data used for those models
    X_test=X_test[:, :2],     # must match feature shape
    y_train=y_train,
    y_test=y_test,
    lmt=lmt,
    version='v2',
    decision=True,
    prob_surf=True,
    feature_pair=(0, 1),
    original_models=original_models2
)

# calibration curves
calibration.plot_lmt_combined_calibration_and_hists(
    X_test[:,:2], y_test,
    X_test, y_test,
    clfs=[original_models2[0][0], original_models2[1][0], original_models2[2][0], original_models2[3][0]],
    nodes=[original_models2[0][1], original_models2[1][1], original_models2[2][1], original_models2[3][1]],
    clfs_ext=[extended_models2[0][0], extended_models2[1][0], extended_models2[2][0], extended_models2[3][0]],
    nodes_ext=[extended_models2[0][1], extended_models2[1][1], extended_models2[2][1], extended_models2[3][1]],
    model_labels=["Shallow", "Regular", "Overfit", "Pruned"]
)

# true vs predicted probabilities
models = {
    "Shallow Tree (orig)": (original_models2[0][0], original_models2[0][1]),
    "Regular Tree (orig)": (original_models2[1][0], original_models2[1][1]),
    "Overfit Tree (orig)": (original_models2[2][0], original_models2[2][1]),
    "Pruned Tree (orig)": (original_models2[3][0], original_models2[3][1]),
    "Shallow Tree (ext)": (extended_models2[0][0], extended_models2[0][1]),
    "Regular Tree (ext)": (extended_models2[1][0], extended_models2[1][1]),
    "Overfit Tree (ext)": (extended_models2[2][0], extended_models2[2][1]),
    "Pruned Tree (ext)": (extended_models2[3][0], extended_models2[3][1]),
}

p_true = expit(alpha * (X_test[:, 0]**2 + X_test[:, 1]**2 - r0))

calibration.plot_pred_vs_true(
    models=models,
    p_true=p_true,
    X_test=X_test,
    model_type='lmt',
    orig_cols=2,
    figsize=(12, 5)
)

# scatter plots by quantiles
# Step 1: Get predicted probabilities from all models
all_probas = []

for (clf, nodes) in zip(
                                [original_models2[0][0], original_models2[1][0], original_models2[2][0], original_models2[3][0]],
                                [original_models2[0][1], original_models2[1][1], original_models2[2][1], original_models2[3][1]]
    ):
    y_proba = lmt.predict_proba_lmt_multiclass(X_test[:, :2], clf, nodes)[:, 1]
    all_probas.append(y_proba)

for (clf, nodes) in zip(
                                [extended_models2[0][0], extended_models2[1][0], extended_models2[2][0], extended_models2[3][0]],
                                [extended_models2[0][1], extended_models2[1][1], extended_models2[2][1], extended_models2[3][1]]
    ):
    y_proba = lmt.predict_proba_lmt_multiclass(X_test, clf, nodes)[:, 1]
    all_probas.append(y_proba)

all_probas = np.concatenate(all_probas)

# Step 2: Define consistent binning, norm, and colormap
nq = 5
q_edges = np.quantile(all_probas, np.linspace(0, 1, nq + 1))
norm = mpl.colors.Normalize(vmin=0, vmax=nq - 1)
cmap_name = 'Spectral'
cmap = mpl.cm.get_cmap(cmap_name, nq)
model_labels = ["Shallow", "Regular", "Overfit", "Pruned"]

# Step 3: Create subplots
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# Top row: original (X_test[:, :2])
for idx, (clf, nodes) in enumerate(zip(
                                [original_models2[0][0], original_models2[1][0], original_models2[2][0], original_models2[3][0]],
                                [original_models2[0][1], original_models2[1][1], original_models2[2][1], original_models2[3][1]]
    )):
    y_proba = lmt.predict_proba_lmt_multiclass(X_test[:, :2], clf, nodes)[:, 1]
    ax = axes[0, idx]
    calibration.plot_scatter_by_quantile(
        X_test[:, :2], y_proba,
        n_quantiles=nq,
        title=f"{model_labels[idx]} – Original",
        ax=ax,
        feature_1=0,
        feature_2=1,
        cmap=cmap,
        norm=norm,
        q_edges=q_edges
    )

# Bottom row: extended (predict on full X_test, still plot first 2 dims)
for idx, (clf, nodes) in enumerate(zip(
                                [extended_models2[0][0], extended_models2[1][0], extended_models2[2][0], extended_models2[3][0]],
                                [extended_models2[0][1], extended_models2[1][1], extended_models2[2][1], extended_models2[3][1]]
    )):
    y_proba = lmt.predict_proba_lmt_multiclass(X_test, clf, nodes)[:, 1]
    ax = axes[1, idx]
    calibration.plot_scatter_by_quantile(
        X_test[:, :2], y_proba,
        n_quantiles=nq,
        title=f"{model_labels[idx]} – Extended",
        ax=ax,
        feature_1=0,
        feature_2=1,
        cmap=cmap,
        norm=norm,
        q_edges=q_edges
    )


# Step 4: Shared colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    orientation='vertical',
    fraction=0.02,
    pad=0.01
)
cbar.set_label("Quantile group")
cbar.set_ticks(np.arange(nq))
cbar.set_ticklabels([f"Q{i + 1}" for i in range(nq)])

fig.suptitle("Scatter Plots by Quantile Groups of Predicted Probability for LMT version 2", fontsize=16)
plt.show()
```

---

## Tips & notes

* **Use both views.** Population metrics (vs $p^*$) reveal **true calibration**; empirical metrics reflect what you’d see in the wild and are noisier. Agreement between them is a strong signal.
* **Seed sensitivity.** Re-run with different seeds; empirical curves will jitter, while population curves should stay stable.

**Code location:** `lmt_pkg/calibration_functions.py`. 
**Theory document:** `Documentation/theory/calibration_implementation.md`. 