# Composite Tree: multiple root cuts, regular `predict()`

A **Composite Tree** makes several binary cuts at the root (potentially on different features, or even multiple cuts on the same feature), then trains a separate scikit-learn `DecisionTreeClassifier` for each resulting region. At inference, it routes each sample to its region’s tree and returns the region tree’s prediction—so it behaves like a standard classifier with a normal `predict()`.

---

## Step-by-step

### 1) Find the best root cuts

Two helper functions are provided.

#### a) One threshold per feature: `find_best_cuts(X, y, *, max_cuts=2, criterion='entropy', n_bins=10)`

* **Candidate thresholds**  
  For each feature $x_j$, generate `n_bins` evenly spaced thresholds in the closed range

$$
[\text{quantile}_{5}(x_j), \text{quantile}_{95}(x_j)]
$$


  This avoids extreme outliers and degenerate splits at the min/max.

* **Score of a single threshold**  
  For a candidate threshold $t$ on feature $j$, split the labels into

$$
\text{left}=\{\,y_i:\ x_{ij}\le t\,\},
\text{right}=\{\,y_i:\ x_{ij}>t\,\},
$$

  skipping empty sides. Let $p_L=\tfrac{|\text{left}|}{|y|}$ and $p_R=1-p_L$.  

  The function computes the **information gain**

$$
\text{gain}(t)=I(y) - \bigl(p_L I(\text{left}) + p_R I(\text{right})\bigr),
$$

  where the impurity $I(\cdot)$ is either

  * **Entropy** (base-2, with a tiny epsilon inside the log for numerical safety):

$$
H(v) = -\sum_{c\in\mathcal{C}}\hat p_c(v)\,\log_2 \bigl(\hat p_c(v)+\varepsilon\bigr),
\hat p_c(v)=\frac{\bigl|\{\,i:\ v_i=c\,\}\bigr|}{|v|},
$$

  * **Gini**:

$$
G(v) = 1-\sum_{c\in\mathcal{C}}\hat p_c(v)^2.
$$

  The best $t$ for each feature is the one with maximal gain.  
  (In the code, classes $\mathcal{C}$ are taken from `np.unique(y)`.)

* **Selecting features to cut**  
  Gather the best $(\text{gain},j,t)$ per feature, sort by gain descending, and return the top `max_cuts` as a dictionary `{feature_index: threshold}`.

**Complexity (roughly):** $O\!\bigl(d\cdot n_{\text{bins}}\cdot n\bigr)$ where $d$ is \#features and $n$ is \#samples (each candidate split scans a boolean mask of length $n$).

#### b) Multiple thresholds per (possibly the same) feature:

`find_best_cuts_multiple_per_feature(X, y, *, max_cuts=3, criterion='entropy', n_bins=10, top_k_per_feature=2)`

* Same candidate grid and same **gain** definition as above.
* For each feature $j$, keep its **top `top_k_per_feature`** thresholds by gain.
* Pool across features, take the **global top `max_cuts`** thresholds, and return a dict `{feature_index: [t1, t2, …]}`.  
  This is the mode to use when you explicitly want more than one cut on the same feature.

> **Which function should I use?**
>
> * At most one cut per feature → `find_best_cuts`.
> * Possibly multiple cuts on the same feature → `find_best_cuts_multiple_per_feature` and control density with `top_k_per_feature`.

---

### 2) Partition the space into regions

* **If you cut $m$ distinct features with one threshold each** (e.g., $x_{f_1}\leq t_1$, $x_{f_2}\leq t_2$, …): you get $2^m$ axis-aligned, disjoint regions, one for every True/False combination of the $m \leq$ tests.

* **If you allow multiple thresholds per feature** (say feature $j$ has $k_j$ thresholds), that feature is split into $k_j+1$ intervals
  $(-\infty, t_{j1}],\ (t_{j1}, t_{j2}],\ \ldots,\ (t_{j\,k_j}, \infty)$.  
  The total number of hyper-rectangular regions is $\prod_j (k_j+1)$.

---

### 3) Train one decision tree per region

Use one of the builders depending on the cut format you collected:

* **Single threshold per feature:**
  `create_root_with_joint_cuts(X, y, cut_dict, size='regular', criterion='entropy', random_state=0)`

* **Multiple thresholds per feature:**
  `create_root_with_multiple_thresholds(X, y, cut_dict, size='regular', criterion='entropy', random_state=0)`, where `cut_dict` maps a feature index to a **list** of thresholds.

For each region:

1. Build a boolean mask for that region (product of the region’s interval conditions).
2. Subset $(X,y)$ by the mask. If the subset is empty, store `None` for that region.
3. Fit a separate `sklearn.tree.DecisionTreeClassifier` with the requested `criterion` and a preset tied to `size`:

* `size='shallow'`: `{'max_depth': 1, 'min_samples_leaf': 20, 'min_samples_split': 40}`
* `size='regular'`: `{'max_depth': 3, 'min_samples_leaf': 5,  'min_samples_split': 15}`
* `size='overfit'`: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}`

Note: when a `max_depth` is used, the code subtracts 1 (so the subtrees have that depth beneath the composite root).

---

## How prediction works

`CompositeTreeClassifier` holds:

* `trees`: one trained subtree per region,
* `region_conditions`: the region definitions (per-feature intervals or binary cut outcomes),
* optional `feature_names`.

For an input matrix $X$, it:

1. Finds which region mask each row satisfies,
2. Routes rows to the corresponding subtree and calls that subtree’s `.predict`,
3. **Fallback:** rows that match no region (rare with consistent cuts) are assigned class `0`.

---

## Visualization

`visualize_root_and_subtrees_grid(...)` draws:

* a schematic root with labeled cut conditions, and
* a grid of the child region trees (each rendered via `sklearn.tree.plot_tree`), annotated with the conjunction of inequalities that define that region.

If using multiple thresholds per feature, the companion utilities display interval labels instead of single $\leq$ / $>$ tests.

---

## Practical notes

* **Choosing `criterion`:** Gini and entropy typically rank thresholds similarly; entropy is more sensitive to class probability changes near 0/1. Both are supported in the code paths above.
* **`n_bins`:** Higher values explore more thresholds but cost more time. Since thresholds are taken from $[5\%,95\%]$, very skewed or constant features may yield many uninformative candidates.
* **Region sample sizes:** If a region ends up tiny, consider `size='shallow'` (regularization) or reduce the number of root cuts.

---

## Summary

This composite approach lets you:

* make **several, jointly applied root cuts** (on distinct or repeated features),
* **modularly grow** one subtree per region with standard scikit-learn tools,
* keep a clean **`.predict()`** interface and standard plotting.

It’s a hybrid of rule-based partitioning (at the root) and classic decision trees (within regions), giving you fine control over complexity, interpretability, and decision boundaries.

**Disclaimer.** In practice, this approach may not outperform strong baselines on average; its value is chiefly in **structure, control, and interpretability**, not guaranteed accuracy gains.
