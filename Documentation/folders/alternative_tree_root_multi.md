# Alternative tree (root multi-cuts) : notebook guide

> **What this is.**
> These two notebooks prototype and demonstrate an **alternative tree** that makes **multiple cuts at the root** (joint thresholds across features) and then grows a regular subtree inside each resulting region. You get a *tree-of-trees*: the root partitions the space in parallel, and each partition is handled by its own decision tree (and, optionally, a Logistic Model Tree variant with LogitBoost at internal nodes).

**Exact implementation:** `lmt_pkg/composite_tree.py`. 
**Theory note:** `Documentation/theory/multi_root.md`. 

---

## Key concepts
* **Root multi-cuts:** choose one threshold per selected feature, then build the **Cartesian product** of “≤”/“>” decisions to form regions (e.g., with 2 features you get up to 4 regions).
* **Per-region subtrees:** train a standard `DecisionTreeClassifier` in each region with depth/regularization presets (e.g., *shallow*, *regular*, *overfit*).
* **Composite predictor:** route each sample to its region, then query that region’s subtree for class/probability.
* **Optional LMT layer:** inside each region’s subtree, replace leaf/inner predictions with **LogitBoosted** logistic models for smoother probability estimates and better calibration.

---

## 1) `tree_building.ipynb`

1. **Cut selection at the root**

   * `find_best_cuts(X, y, max_cuts=2, criterion='entropy', n_bins=10)`
     Scans features, tries percentile-based candidate thresholds, scores each by **information gain** (entropy or Gini), and returns a dictionary `{feature_index: threshold}` for the top `max_cuts` features.

2. **Joint regions + per-region trees**

   * `create_root_with_joint_cuts(X, y, cut_dict, size='regular', criterion='entropy', random_state=0)`
     Builds all **2^m** region conditions from `m=len(cut_dict)` features, fits a `DecisionTreeClassifier` **inside each region** using presets:

     * `size='shallow'`: very small trees
     * `size='regular'`: balanced defaults
     * `size='overfit'`: little regularization

   * Returns: list of region trees, region masks, region conditions, the feature indices and thresholds.

3. **Composite estimator wrapper**

   * `CompositeTreeClassifier(trees, cut_dict, feature_names=None)`

     * `.predict(X)`: route by region → subtree prediction
     * `.predict_proba(X)`: route by region → subtree probability

4. **Evaluation & visualization**

   * `evaluate_forest(trees, region_masks, X, y, cut_dict, ...)` → accuracy & assigned flags
   * `visualize_trees(...)`, `visualize_full_tree_schematic(...)`, `visualize_root_and_subtrees_grid(...)` → quick looks of the region subtrees
   * A **Testing** section using synthetic data (e.g., “Circles” / diamond-shaped boundary) to sanity-check learning dynamics.

5. **Calibration & diagnostics**

   * `plot_combined_calibration_and_hists_for_composites_side_by_side(...)`
     Overlays calibration curves for **two composite models** (e.g., Original vs Extended) and shows **two histograms** (one per model family) to compare probability distributions.

6. **Logistic Model Tree (LMT) integration (advanced)**

   * `fit_logistic_composite_tree_v2(X, y, cut_dict, size='regular', pruning=True, tree_random_state=0, lb_n_estimators=200, lb_eps=1e-5, lb_cv_splits=5, lb_random_state=0)`
     For each region’s subtree:

     * Fit a **SimpleLogistic** (LogitBoost with CV) at the subtree root to select `M*`.
     * **Warm-start** child nodes with the parent’s learners and continue LogitBoost training per node.
     * Returns the region trees plus a `node_models` dict keyed by `(region_idx, node_id)` with learners, class count `J`, `M*`, and CV errors (when applicable).

   * Inference helpers:

     * `predict_composite_lmt(X, composite_clf, node_models)`
     * `predict_proba_composite_lmt(X, composite_clf, node_models)`

   * Visual diagnostics for the composite LMT:

     * `plot_tree_with_linear_models_grid_improved(...)` (shows which nodes carry linear models)
     * `plot_decision_regions_lmt(...)` and `plot_probability_surface_lmt(...)`
     * `plot_calibration_curve_composite_lmt(...)`
     * `plot_pred_vs_true(models, p_true, X_test, ...)` (compare model scores vs. “true” probabilities in synthetic settings)

### Typical workflow 

1. **Pick root thresholds**
   `cut_dict = find_best_cuts(X_train, y_train, max_cuts=2, criterion='entropy')`
2. **Build composite model**
   `trees, masks, region_conditions, feats, thresholds = create_root_with_joint_cuts(X_train, y_train, cut_dict, size='regular')`
   `clf = CompositeTreeClassifier(trees, cut_dict, feature_names=["X1","X2"])`
3. **Evaluate & visualize**
   `acc, preds, assigned = evaluate_forest(...)` → calibration plots / schematic visualizations
4. **(Optional) Upgrade to LMT**
   `trees, node_models, region_conditions, feats, thresholds = fit_logistic_composite_tree_v2(...)`
   Then use the `predict_*_composite_lmt` and plotting helpers.

> **Why multi-cuts?**
> When the decision boundary is strongly **axis-aligned but interacts across features**, one parallel partition at the root can drastically simplify downstream trees, improving interpretability and sometimes calibration. The theory note (`Documentation/theory/multi_root.md`) formalizes the expected gain and shows when this is preferable to serial (single-split) growth.

---

## 2) `visualization_new_tree.ipynb`

### Purpose

An **interactive image browser** (built with `ipywidgets`) to review artifacts produced by the alternative tree experiments: region schematics, decision regions, probability surfaces, calibration curves, and “true” distributions.

### Key pieces

* **Folder assumptions**

  * Base path: `images/`
  * Under each dataset folder (e.g., `images/n5000_alpha2/`), create one subfolder per model (e.g., `logitboost/`, `c45/`, `lmt_v1/`, `lmt_extended/`).
  * Inside each model folder, save images with the following **section prefixes**:

    * `dataset`, `trees`, `decision`, `prob`, `calibration`, `scatter`, `true`
  * For cases with multiple variants of the same section (e.g., extended view), name them with **numeric suffixes**:
    `prob_0.png`, `prob_1.png`, `decision_0.png`, `decision_1.png`, etc.

* **Core functions**

  * `get_datasets()` → list dataset folders under `images/`.
  * `get_models(dataset)` → returns available model folders plus a special `"root"` view.
  * `get_images(dataset, model, section)` → collects files that start with the chosen section prefix (handles multi-image cases like `prob_0`, `prob_1`).
  * `show_images(dataset, model, section)` → displays images; if multiple, arranges them in a **2×2 grid**, scaling up so they’re legible.
  * `update_models()`, `update_images()` → widget callbacks to refresh the model list and the image grid as you change dataset/model/section.

* **UI controls**

  * **Dropdowns:** *Dataset*, *Model*, *Section* (section choices hard-coded as `['dataset','trees','decision','prob','calibration','scatter','true']`).
  * **Display logic:** automatically shows images when a selection changes. If nothing matches, shows **“Image not available.”**

### How to use it

1. Place your image outputs under `images/<dataset>/<model>/` with the expected prefixes.
2. Run the notebook cells; the three dropdowns will appear.
3. Select a dataset → a model (or “root”) → a section.
   If multiple images exist (e.g., `prob_0.png`, `prob_1.png`), you’ll see them **side-by-side in a 2×2 grid**.

---

## API cheat-sheet (as used in the notebooks)

* **Cut discovery & region trees**

  * `composite_tree.find_best_cuts(X, y, max_cuts=2, criterion={'entropy'|'gini'}, n_bins=10) -> dict[int->float]`
  * `composite_tree.create_root_with_joint_cuts(X, y, cut_dict, size={'shallow'|'regular'|'overfit'}, criterion, random_state) -> (trees, region_masks, region_conditions, feats, thresholds)`
  * `CompositeTreeClassifier(...).predict(X) / predict_proba(X)`

* **LMT integration**
  * `composite_tree.fit_logistic_composite_tree_v2(...) -> composite_tree, node_models`

* **Evaluation**

  * `composite_tree.evaluate_forest_with_intervals(...) -> (accuracy, predictions, assigned_mask)`


* **Visualization** 
  * `composite_tree.visualize_root_and_subtrees_grid_with_intervals(...)`
  * `lmt.plot_decision_surface_from_fitted_tree(...)`


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Finding best cuts
cut_dict_ext = composite_tree.find_best_cuts_multiple_per_feature(X_train, y_train, max_cuts=2, top_k_per_feature=1, criterion='entropy', n_bins=15)

# Growing the tree and LMT fitting
trees, node_models, region_conditions, feats, thresholds = fit_logistic_composite_tree_v2(
    X_train, y_train,
    cut_dict=cut_dict_ext,
    size='regular',
    lb_n_estimators=200,
    lb_cv_splits=5
)

clf_lmt = composite_tree.CompositeTreeClassifier(
    trees=trees,
    region_conditions=region_conditions,
    feature_names=["X1", "X2"]  # adjust if needed
)

# Predict class labels using the composite logistic model tree
y_pred = predict_composite_lmt(X_test, clf_lmt, node_models)

# Visualization
node_models_list = split_node_models_by_tree(node_models, n_trees=len(trees))

plot_tree_with_linear_models_grid_improved(
    trees=trees,
    region_conditions=region_conditions,
    node_models_list=node_models_list,  # result of your split_node_models_by_tree()
    X=X_train,
    feature_names=["X1", "X2"],
    title="Composite Tree with LMT V2",
    show_internal=False
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
lmt.plot_decision_regions_lmt(
    X_train,
    y_train,
    clf_lmt=clf_lmt,
    nodes_lmt=node_models,
    feature_pair=(0, 1),
    fill_value="mean",
    grid_steps=200,
    cmap='RdYlBu',
    ax=ax[0],
    title="Decision regions (LMT V2)",
    show_scatter=False,
    tree_model='composite'
)
lmt.plot_probability_surface_lmt(
    clf_lmt=clf_lmt,
    nodes_lmt=node_models,
    X=X_train,
    feature_pair=(0,1),
    prob_class=1,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=ax[1],
    title='Predicted probability surface (LMT V2)',
    show_scatter=True,
    tree_model='composite'
)
plt.show()

```

**Code location:** `lmt_pkg/composite_tree.py`. 
**Theory document:** `Documentation/theory/multi_root.md`. 
