# Alternative tree (linear spline) : notebook guide

> **What this is.**
> These notebooks prototype a **custom decision tree** whose internal splits are chosen via **linear-spline criteria** and whose predictions can be upgraded with **LogitBoost-style logistic models** (LMT). You’ll see end-to-end growth, alternative split metrics (AUC vs Brier), two pruning styles (original **AUC-δ** and **local gain**), and **cross-validation to pick δ**. The goal is to understand how spline-guided splitting + principled pruning affect decision boundaries, calibration, and model size.

**Exact implementation:** `lmt_pkg/spline_splitting.py`
**Theory note:** `Documentation/theory/linear_spline_tree.md` and `Documentation/theory/regression_pruning.md`

---

## Key concepts

* **Linear-spline splits.** For a candidate feature $x_j$, we fit/score simple **piecewise-linear** (knoted) relations that capture changes in class odds along $x_j$. Candidate “knot+direction” splits are ranked by a metric (e.g., **AUC**↑ or **Brier**↓) computed on the node’s data; the best split is applied if it clears purity/size guards.
* **Leaf models (optional LMT).** After the tree structure is set (or as it grows), replace node/leaf scores with **LogitBoosted logistic models**, which often yield smoother probability maps and better calibration.
* **Pruning.**

  * **Original AUC-δ (BFS):** keep a node only if its **node-level AUC ≥ δ** (held-out or CV). Otherwise, **collapse** it.
  * **Local-gain pruning:** keep a split only if the **children’s weighted AUC exceeds the parent’s AUC by ≥ δ** (a *per-split* improvement test).
* **Model selection (δ via CV).** Sweep δ and pick what optimizes a chosen validation objective (e.g., **Brier** or **log-loss**).

---

## 1) `splitting_construction.ipynb` (from splits to pruning)

1. **Tree growth with spline-guided splits**

   * `grow_spline_tree(X, y, features, max_depth, min_samples_leaf, purity_threshold, ...)`
     Builds a custom tree on `features` (commonly `[0, 1]`), choosing each split via a linear-spline scoring rule (metric configurable).

2. **Prediction API**

   * `predict(tree, X)` and `predict_proba(tree, X)` for the custom structure.

3. **Leaf/LMT modeling & pipeline**

   * `fit_logistic_model_tree_custom(...)` (helpers from `lmt_pkg/spline_splitting.py`)
   * `pipeline_spline_tree_lmt(X_train, y_train, X_test, y_test, ..., pruning_threshold=...)`
     End-to-end: grow → (optionally) attach LogitBoost models → prune → evaluate.

4. **Pruning (original BFS AUC-δ)**

   * `regression_pruning_spline_bfs(X, y, root_node, node_models, threshold=δ, ...)`
     Prunes by node AUC using breadth-first traversal.

5. **Visualization**

   * `plot_custom_tree(...)`, `plot_probability_surface_tree(...)`, plus **decision-boundary** and **probability-surface** plots.

### Typical outputs

* A **fitted** tree and its **pruned** counterpart (given δ).
* Figures for **tree structure**, **decision regions**, and **probability surfaces**.
* Quick hold-out **accuracy/AUC** checks to sanity-test behavior.

---

## 2) `splitting_metrics.ipynb` (MVP: split metrics, local-gain pruning, CV for δ)

1. **Competing split criteria**

   * Grow two versions using:

     * `metric='auc'` → maximize node AUC,
     * `metric='brier'` → minimize Brier score.
   * Each yields a `(tree, node_models)` pair, e.g.:

     * `spline_lmt_auc, node_models_auc`
     * `spline_lmt_brier, node_models_brier`

2. **Two pruning methods**

   * **Original (BFS) AUC-δ:** keep node if `AUC ≥ δ`.
   * **Local gain rule:** `regression_pruning_gain_local(...)` keeps a split only if

  $$
  \underbrace{\frac{n_l}{n}\,\text{AUC}_l+\frac{n_r}{n}\,\text{AUC}_r}_{\text{children (weighted)}}\;-\;\text{AUC}_{\text{parent}} \geq \delta
  $$


3. **Cross-validation to pick δ**

   * `cv_delta_pruning(X_val, y_val, tree, node_models, deltas=..., K=..., metric_eval=..., metric_prune=..., method=...)`

     * `metric_prune` = criterion used **during pruning** (often AUC).
     * `metric_eval` = objective to **choose δ** (e.g., **Brier** or **log-loss**).
     * `method` ∈ `{ "original", "local" }`.

   * **δ ranges** (illustrative, as used in-notebook):

     * Original rule: `np.linspace(0.60, 0.90, 20)`.
     * Local-gain rule: `np.linspace(0.0, 0.0025, 20)` (small deltas—these are *improvements*).

4. **Metrics & visuals**

   * **Performance:** accuracy, F1, ROC-AUC.
   * **Calibration:** Brier, log-loss, reliability curves (`CalibrationDisplay.from_predictions(...)`).
   * **Panels:** side-by-side **tree diagrams**, **decision boundaries**, and **probability surfaces** for (AUC-split vs Brier-split) × (Original vs Local-gain) pruning.

### What to look for

* Whether **Brier-split** trees produce **smoother probability maps** and **lower calibration error**.
* How **local-gain pruning** reshapes the tree (often trimming low-value splits a global threshold would keep).
* Which **δ (from CV)** under **Brier** vs **log-loss** is more stable across seeds.

---

## 3) `threshold_metrics.ipynb` (sweeping δ and reading trade-offs)

### What you’ll find

1. **Systematic δ sweeps**

   * Repeat across **multiple seeds** (e.g., 0 and 10) and across several independently grown trees (“First/Second/Third/Fourth tree”).

2. **Helper toolkit**

   * **Bookkeeping & summaries**

     * `results_from_tree(...)` → detailed `df_results` (row per δ) + compact `df_summary` (mins/maxes, best-δ by metric).
     * `annotate_min_max(ax, x, y, label)` → marks optima on curves.
   * **Calibration**

     * `compute_ece(y_true, y_prob, n_bins=10)` → **ECE** with 10-bin reliability diagrams.
   * **Size/complexity**

     * `count_nodes(node)`, `tree_depth(node)` to track complexity.
   * **Pipeline**

     * `pipeline_spline_tree_lmt(...)` to fit → prune → score over δ.

3. **Per-δ measurements**

   * **Performance:** accuracy, F1, ROC-AUC.
   * **Calibration:** Brier, log-loss, **ECE (10 bins)**.
   * **Model size:** depth, total nodes.
   * **Model-selection proxies:** **AIC/BIC-style** scores derived from **log-loss** plus an **estimated parameter count** (e.g., non-zero coefficients/intercepts from LogitBoost at leaves). *(Practical proxies rather than textbook AIC/BIC.)*

4. **Plots you’ll see**

   * Metric-vs-δ curves with **min/max annotations**.
   * **Complexity curves** (depth & node count) to spot the **elbow**.
   * Reliability diagrams + probability histograms per configuration.

### How to interpret

* Good δ values tend to align where **Brier/log-loss/ECE are low**, **AIC/BIC-style proxies** are low, and **complexity** shows an **elbow**—often clustering near the same δ for a given seed/run.
* The notebook highlights **seed sensitivity**: node-level AUCs fluctuate with sampling, so **hard thresholds** can induce **high-variance pruning**.
---

## Typical workflow

1. **Choose features** (often `[0, 1]`) and grow with `grow_spline_tree(...)`, setting `max_depth`, `min_samples_leaf`, and a split `metric` (`'auc'` or `'brier'`).
2. **Attach LMT models** with `fit_logistic_model_tree_custom(...)` or via `pipeline_spline_tree_lmt(...)`.
3. **Pick a pruning method**:

   * **Original AUC-δ (BFS)** with `regression_pruning_spline_bfs(...)`, or
   * **Local gain** with `regression_pruning_gain_local(...)`.
4. **Select δ via CV** using `cv_delta_pruning(...)` with your preferred **evaluation metric** (Brier/log-loss).
5. **Evaluate & visualize**: decision boundaries, probability surfaces, reliability diagrams, and complexity plots.
6. **Summarize** with `results_from_tree(...)` and annotate best points on curves.

---

## API cheat-sheet (as used in the notebooks)

* **Growth & inference**

  * `spline.grow_spline_tree(X_train, y_train, features, max_depth, min_samples_leaf, purity_threshold, metric='auc'|'brier', verbose=True -> spline_tree`
  * `predict(X, spline_tree) -> y_pred`
  * `predict_proba(X, spline_tree) -> y_proba`

* **LMT integration**

  * `spline.fit_logistic_model_tree_custom(X_train, y_train, festures=[0,1], max_depth, min_samples_leaf, purity_threshold, metric='auc'|'brier', verbose=True) -> spline_tree, node_models`

* **Pruning**

  * `spline.regression_pruning_spline_bfs(X_train, y_train, spline_tree, node_models, delta, multiclass=False, verbose=True) -> pruned_spline_tree, pruned_node_models` Original regression pruning based on AUC of each node independently, with manual delta selection.
  * `spline.regression_pruning_gain_local(X_train, y_train, spline_tree, node_models, delta, multiclass=False, verbose=True) -> pruned_spline_tree, pruned_node_models` Local gain based regression pruning, with manual delta selection
  * `spline.v_delta_pruning(X_val, y_val, spline_tree, node_models, deltas, K=5, metric_eval='brier'|'logloss', metric_prune='auc', method='local'|'original', multiclass=False, verbose=True) -> best_delta` Cross-validation routine for finding best delta for each of both regression pruning methods.

* **Visualization**

  * `spline.plot_custom_tree_with_models(pruned_spline_tree, pruned_node_models, X_train, title)`
  * `spline.plot_decision_regions_custom_tree(X_train, y_train, pruned_spline_tree, pruned_node_models, title)`
  * `spline.plot_probability_surface_custom_tree(pruned_spline_tree, pruned_node_models, X_train, title)`



**Example**
```python
# Data for absolute value dataset
n, alpha, r0 = 5000, 2, 1.5
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)
r2 = np.abs(X1) + np.abs(X2) 
p = 1 / (1 + np.exp(-alpha * (r2 - r0)))
y = np.random.binomial(1, p)
X = np.column_stack((X1, X2))   # now X has shape (n,2)
# Step 1: Train (60%) + Temp (40%)
X_train, X_temp, y_train, y_temp, true_probs_train, true_probs_temp = train_test_split(X, y, p, test_size=0.4, random_state=42, stratify=y)
# Step 2: Validation (20%) + Test (20%) from Temp
X_val, X_test, y_val, y_test, true_probs_val, true_probs_test = train_test_split(X_temp, y_temp, true_probs_temp, test_size=0.5, random_state=42, stratify=y_temp)


# LMT model with spline splitting minimizing brier score
from lmt_pkg import spline_splitting as spline
spline_lmt, node_models = spline.fit_logistic_model_tree_custom(X_train, y_train, [0,1], 10, 20, 0.85, metric='brier', verbose=True)

# Cross-validation routine for delta selection on local gain regression pruning
deltas_local = np.linspace(0, 0.0025, 20)
delta_star = spline.cv_delta_pruning(
    X_val, y_val, spline_lmt, node_models,
    deltas=deltas_local, K=5,
    metric_eval='brier',  # or 'log-loss'
    metric_prune='auc',  # the metric used during pruning
    method='local',  # or 'original'
    verbose=True
    ,true_probs=true_probs_val  # Pass true probabilities for evaluation
)

# Regression pruning (Local gain)
pruned_spline_lmt, pruned_node_models = spline.regression_pruning_gain_local(
    X_train, y_train, spline_lmt, node_models, delta=delta_star, verbose=True
)

# Visualization
spline.plot_custom_tree_with_models(pruned_spline_lmt, pruned_node_models, X_train, title="Tree with node models")
spline.plot_probability_surface_custom_tree(pruned_spline_lmt, pruned_node_models, X_train, [0,1], title="Probability Surface")
plt.show()
```

---

## Tips & notes

* **Split metric choice.** If **calibration** matters, the **Brier-driven** splitter often yields smoother probability fields; if you want strong ranking early, **AUC-driven** splits can help.
* **δ scales differ.** For **original AUC-δ**, try `0.55–0.75`. For **local-gain**, start **near zero** (e.g., `0–0.003`)—it measures *incremental improvement*.
* **CV target ≠ prune metric.** It’s fine to prune by **AUC** but select δ by **Brier** or **log-loss**—this often improves calibration without harming AUC much.
* **Look for elbows.** Cross-check δ at which **complexity** plateaus with δ minimizing **Brier/log-loss/ECE**; agreement is a strong selection signal.
* **Seed sensitivity.** Fix random seeds when comparing methods and report variability; **local-gain + CV** typically reduces variance in the final choice.

**Code location:** `lmt_pkg/spline_splitting.py`. 
**Theory document:** `Documentation/theory/linear_spline_tree.md` and `Documentation/theory/regression_pruning.md`. 