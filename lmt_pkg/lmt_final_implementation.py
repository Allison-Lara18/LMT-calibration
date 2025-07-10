"""
LMT final implementation with plotting functions
"""

# --------------------------- #
# Libraries                   #
# --------------------------- #
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
#import logitboost_j_implementation as logitboost
from . import logitboost_j_implementation as logitboost
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from  sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


# ---------------------- #
# Tree construction      #
# ---------------------- #
# Tree pruning function
def prune_tree_cv(X, y, cv=5, scoring='accuracy', random_state=0):
    """
    Prune a DecisionTreeClassifier using minimal cost-complexity pruning
    with cross-validation to select the best ccp_alpha.

    Parameters:
    - X: Training features
    - y: Training labels
    - cv: Number of cross-validation folds
    - scoring: Scoring metric for cross-validation
    - random_state: Random seed for repeatability

    Returns:
    - pruned_tree: DecisionTreeClassifier fitted with the optimal ccp_alpha
    - best_alpha: The ccp_alpha value that achieved the highest CV score
    - alphas: List of tested ccp_alpha values
    - cv_scores: List of mean CV scores corresponding to each alpha
    """
    # Compute pruning path
    base_clf = DecisionTreeClassifier(random_state=random_state)
    path = base_clf.cost_complexity_pruning_path(X, y)
    alphas = path.ccp_alphas[1:]     # drop the 0.0 entry to prune something

    
    # Evaluate each candidate alpha with cross-validation
    cv_scores = []
    for alpha in alphas:
        clf = DecisionTreeClassifier(random_state=random_state, ccp_alpha=alpha, criterion='entropy')
        scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
        cv_scores.append(scores.mean())
    
    # Select the best alpha
    best_alpha = alphas[np.argmax(cv_scores)]
    
    # Train the final pruned tree on the full dataset
    pruned_tree = DecisionTreeClassifier(random_state=random_state, ccp_alpha=best_alpha, min_samples_leaf=5, min_samples_split=15, criterion='entropy')
    pruned_tree.fit(X, y)
    
    return pruned_tree, best_alpha, alphas, cv_scores


# Tree construction function
def construct_tree(X, y, size='regular', pruning=False, pruning_show_stats=False, cv=5, scoring='accuracy', random_state=0):
    """
    Constructs a decision tree classifier with specified parameters.
    Parameters:
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    size : 'shallow', 'regular', or 'overfit'
    pruning : bool, apply cost-complexity pruning if True
    pruning_show_stats: bool, show alphas and alphas vs accuracy graph
    cv, scoring, random_state: passed to prune_tree_cv

    Returns:
    clf : DecisionTreeClassifier. A fitted decision tree classifier.
    """
    # define hyperparameter presets
    presets = {
        'shallow':  {'max_depth': 2,  'min_samples_leaf': 20, 'min_samples_split': 40},
        'regular':  {'min_samples_leaf': 5,  'min_samples_split': 15}, # Paper's implementation
        'overfit':  {'max_depth': None,'min_samples_leaf': 1,  'min_samples_split': 2},
    }

    params = presets.get(size)
    if params is None:
        raise ValueError("Invalid size parameter. Choose from 'shallow', 'regular', or 'overfit'.")
    
    # Create a decision tree classifier with specified parameters
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=random_state, **params)
    clf.fit(X, y)

    # Pruning with CART-like method
    if pruning:
        pruned_tree, best_alpha, alphas, cv_scores = prune_tree_cv(X, y, cv=cv, scoring=scoring, random_state=random_state)
        clf = pruned_tree

        if pruning_show_stats:
            print("Best alpha:", best_alpha)
            print("Alphas tested:", alphas)
            print("Cross-validation scores:", cv_scores)    

            # Plot alphas vs. cross-validation scores
            plt.figure(figsize=(10, 6))
            plt.plot(alphas, cv_scores, marker='o')
            plt.title("Cross-Validation Scores vs. Alpha Values")
            plt.xlabel("Alpha (ccp_alpha)")
            plt.ylabel("Cross-Validation Score")
            plt.show()


    return clf

# ------------------------------------- #
# LMT with SimpleLogistic at every node #
# ------------------------------------- #
def _choose_cv_folds(y, k_desired=5):
    """
    Pick a stratified‐CV fold count ≤ smallest class size (or None if too small).
    """
    y = np.asarray(y).ravel()
    # minimum count among classes
    min_count = np.bincount(y, minlength=y.max()+1).min()
    if min_count < 2:
        return None   # skip CV entirely
    return max(2, min(k_desired, min_count))

def fit_logistic_model_tree(
    X, y,
    size='regular',
    pruning=True,
    tree_random_state=0,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0
):
    """
    Build a DecisionTree, then at each node fit a SimpleLogistic (LogitBoost) model
    using the samples that reach that node. Root has no warm_start, children
    inherit their parent's model as warm_start.

    Returns
    -------
    clf_tree     : fitted DecisionTreeClassifier
    node_models  : dict mapping node_id -> {
                      'learners': [...],
                      'J': int,
                      'M_star': int,
                      'cv_errors': array
                   }
    """
    # 1) Fit the base tree
    clf_tree = construct_tree(
        X, y,
        size=size,
        pruning=pruning
    )
    tree = clf_tree.tree_
    n_nodes = tree.node_count

    # 2) Pre-compute the decision_path matrix
    # decision_path gives you a sparse indicator matrix (n_samples x n_nodes)
    node_indicator = clf_tree.decision_path(X)

    # 3) Recursive traversal to fit logitboost at each node
    node_models = {}

    def recurse(node_id, warm_start):
        # mask of samples that pass through this node
        sample_mask = node_indicator[:, node_id].toarray().ravel().astype(bool)
        X_node = X[sample_mask]
        y_node = y[sample_mask]

        # choose CV folds based on y_node
        k = _choose_cv_folds(y_node, lb_cv_splits)
        if k is None:
            # too few samples for stratified CV → do plain boosting (no CV)
            learners, J = logitboost.logitboost_fit(
                X_node, y_node,
                n_estimators=lb_n_estimators,
                eps=lb_eps,
                warm_start=warm_start
            )
            M_star  = J
            cv_errs = None
        else:
            # safe to run CV
            learners, J, M_star, cv_errs = logitboost.simple_logistic_fit(
                X_node, y_node,
                n_estimators=lb_n_estimators,
                eps=lb_eps,
                cv_splits=k,
                warm_start=warm_start,
                random_state=lb_random_state
            )

        # store the fitted LogitBoost model for this node
        node_models[node_id] = {
            'learners': learners,
            'J': J,
            'M_star': M_star,
            'cv_errors': cv_errs
        }

        # propagate to children if not a leaf
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left != -1:
            recurse(left, warm_start=(learners, J))
        if right != -1:
            recurse(right, warm_start=(learners, J))

    # start at root (node 0) with no warm start
    recurse(0, warm_start=None)

    return clf_tree, node_models

# --------------------------------------------- #
# LMT with SimpleLogistic just at the root node #
# --------------------------------------------- #
def fit_logistic_model_tree_v2(
    X, y,
    size='regular',
    pruning=True,
    tree_random_state=0,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0
):
    """
    V2: SimpleLogistic only at root (no warm-start), then
    classical LogitBoost at every other node for M* rounds,
    passing warm_start=parent_model so children inherit.

    Returns
    -------
    clf_tree     : fitted DecisionTreeClassifier
    node_models  : dict[node_id] = {
                      'learners': list,
                      'J': int,
                      'M_star': int,
                      'cv_errors': array or None
                   }
    """

    # 1) Fit the base decision tree
    clf_tree = construct_tree(
        X, y,
        size=size,
        pruning=pruning,
        random_state=tree_random_state
    )
    tree_ = clf_tree.tree_
    n_nodes = tree_.node_count

    # 2) Root: run SimpleLogistic to get M_star
    root_learners, J, M_star, cv_errs = logitboost.simple_logistic_fit(
        X, y,
        n_estimators=lb_n_estimators,
        eps=lb_eps,
        cv_splits=lb_cv_splits,
        warm_start=None,
        random_state=lb_random_state
    )

    # 3) Precompute sample-to-node mapping
    node_indicator = clf_tree.decision_path(X)

    node_models = {}

    def recurse(node_id, warm_start):
        # mask of samples reaching this node
        mask = node_indicator[:, node_id].toarray().ravel().astype(bool) \
               if sparse.issparse(node_indicator) \
               else node_indicator[:, node_id].astype(bool)

        X_node, y_node = X[mask], y[mask]


        if node_id == 0:
            # root: store SimpleLogistic result
            learners_node, J_node, cv_err_node = root_learners, J, cv_errs
        else:
            # child: classical LogitBoost for M_star rounds, warm-start from parent
            learners_node, J_node = logitboost.logitboost_fit(
                X_node, y_node,
                n_estimators=M_star,
                eps=lb_eps,
                warm_start=warm_start
            )
            cv_err_node = None

        node_models[node_id] = {
            'learners': learners_node,
            'J': J_node,
            'M_star': M_star,
            'cv_errors': cv_err_node
        }

        # recurse to children, passing this node's model as warm_start
        left, right = tree_.children_left[node_id], tree_.children_right[node_id]
        if left != -1:
            recurse(left, warm_start=(learners_node, J_node))
        if right != -1:
            recurse(right, warm_start=(learners_node, J_node))

    # start at root
    recurse(0, warm_start=None)

    return clf_tree, node_models

# ----------------------------------- #
# Prediction                          #
# ----------------------------------- #
def predict_lmt(X, clf_tree, node_models):
    leaf_ids = clf_tree.apply(X)
    preds = []
    for xi, leaf in zip(X, leaf_ids):
        # grab the model fitted on that node
        mdl = node_models[leaf]
        learners, J = mdl['learners'], mdl['J']
        # predict label
        p = logitboost.logitboost_predict(xi.reshape(1, -1), learners, J)
        preds.append(p[0])
    return np.array(preds)

# def predict_proba_lmt(X, clf_tree, node_models):
#     """
#     Return P(y=1) for each row of X by routing it to its leaf, then 
#     calling logitboost_predict_proba on that leaf's model.
#     """
#     leaf_ids = clf_tree.apply(X)
#     probs = []
#     for x_i, leaf in zip(X, leaf_ids):
#         learners = node_models[leaf]['learners']
#         J        = node_models[leaf]['J']
#         # returns array[[p(0), p(1)]]
#         p01 = logitboost.logitboost_predict_proba(x_i.reshape(1, -1), learners, J)
#         probs.append(p01[0, 1])
#     return np.array(probs)
def predict_proba_lmt(X, clf_tree, node_models):
    leaf_ids = clf_tree.apply(X)
    probs = []

    for i, (x_i, leaf) in enumerate(zip(X, leaf_ids)):
        learners = node_models[leaf]['learners']
        J        = node_models[leaf]['J']
        p01      = logitboost.logitboost_predict_proba(x_i.reshape(1, -1), learners, J)

        # Handle the case where only one class is present
        if J == 1:
            # Check which class is present in this leaf (typically class 0 or class 1)
            class_in_leaf = node_models[leaf].get('class_in_leaf', 0)  # add this during training if needed
            if class_in_leaf == 1:
                probs.append(1.0)
            else:
                probs.append(0.0)
        else:
            probs.append(p01[0, 1])

    return np.array(probs)

import numpy as np

def predict_lmt_multiclass(X, clf_tree, node_models):
    """
    Predict class labels for a multiclass Logistic Model Tree.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test set.
    clf_tree : DecisionTreeClassifier
        The top‐level tree you built with construct_tree().
    node_models : dict
        Maps leaf_id -> { 'learners': [...], 'J': int, ... }.

    Returns
    -------
    preds : ndarray of shape (n_samples,)
        Predicted class indices (0..J-1).
    """
    leaf_ids = clf_tree.apply(X)
    preds = np.empty(len(X), dtype=int)

    for i, (x_i, leaf) in enumerate(zip(X, leaf_ids)):
        learners = node_models[leaf]['learners']
        J        = node_models[leaf]['J']
        # logitboost_predict returns an array of shape (n_samples_leaf,)
        p = logitboost.logitboost_predict(x_i.reshape(1, -1), learners, J)
        preds[i] = p[0]

    return preds


def predict_proba_lmt_multiclass(X, clf_tree, node_models):
    """
    Predict full probability distributions for a multiclass LMT.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    clf_tree : DecisionTreeClassifier
    node_models : dict mapping leaf_id -> { 'learners': [...], 'J': int, ... }

    Returns
    -------
    proba_matrix : ndarray of shape (n_samples, J_global)
        proba_matrix[i, j] = P(y=j | X[i]) as estimated by the leaf's LogitBoost.
        If different leaves have different J, the matrix is zero-padded on the right.
    """
    leaf_ids = clf_tree.apply(X)
    # figure out the maximum J over all leaves (so we can pad if needed)
    J_global = max(mdl['J'] for mdl in node_models.values())

    proba_matrix = np.zeros((len(X), J_global), dtype=float)

    for i, (x_i, leaf) in enumerate(zip(X, leaf_ids)):
        learners = node_models[leaf]['learners']
        J        = node_models[leaf]['J']
        # returns shape (1, J)
        p = logitboost.logitboost_predict_proba(x_i.reshape(1, -1), learners, J)
        proba_matrix[i, :J] = p[0]

    return proba_matrix


# ----------------------------------- #
# Visualization of the LMT            #
# ----------------------------------- #
def plot_tree_with_linear_models(
    clf_tree,
    node_models,
    X,
    title,
    class_names=None,
    show_internal=False,
    model_threshold=1e-6,
    ax=None  # <---- New optional axis
):
    """
    Plots a DecisionTree with appended linear models in each node.
    Can be embedded inside a subplot grid using the 'ax' argument.
    """
    import re
    n_features = X.shape[1]
    feature_labels = [f"x[{k}]" for k in range(n_features)]

    # precompute intercepts & coefs for each node
    intercepts_dict = {}
    coefs_dict      = {}
    for nid, mdl in node_models.items():
        ints, cos = logitboost.extract_linear_models(mdl['learners'], mdl['J'], n_features)
        intercepts_dict[nid] = ints
        coefs_dict[nid]      = cos

    # Create a new figure and axis only if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
        created_fig = True

    # draw tree
    plot_tree(
        clf_tree,
        feature_names=feature_labels,
        class_names=class_names,
        filled=True,
        node_ids=True,
        ax=ax,
        fontsize=8
    )

    # identify leaf node IDs
    tree_ = clf_tree.tree_
    leaf_ids = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1]

    # append formulas
    for txt in ax.texts:
        full = txt.get_text()
        first_line = full.split("\n", 1)[0]
        m = re.search(r'\b(\d+)\b', first_line)
        if not m:
            continue
        node_id = int(m.group(1))
        if not show_internal and node_id not in leaf_ids:
            continue

        ints = intercepts_dict.get(node_id)
        cos  = coefs_dict.get(node_id)
        if ints is None or cos is None:
            continue

        J = len(ints)
        lines = []
        for j in range(J):
            b = ints[j]
            parts = [f"F{j}(x)={b:.2f}"]
            for k in range(n_features):
                a = cos[j, k]
                if abs(a) > model_threshold:
                    parts.append(f"{'+' if a>=0 else '-'}{abs(a):.2f}*x[{k}]")
            lines.append(" ".join(parts))
        txt.set_text(full + "\n" + "\n".join(lines))

    ax.set_title(title)
    if created_fig:
        plt.tight_layout()
        plt.show()



# ------------------------------------------------------- #
# Visualization of decision surface of two given features #
# ------------------------------------------------------- #
def plot_tree_decision_surface(
    X, y,
    feature_pair,
    size='regular',
    pruning=False,
    feature_names=None,
    class_names=None,
    plot_colors="rb",
    plot_step=0.02,
    cmap=plt.cm.RdYlBu,
    ax=None
):
    i, j = feature_pair
    X2 = X[:, [i, j]]

    # Fit the tree
    clf = construct_tree(X2, y, size=size, pruning=pruning)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Decision surface
    DecisionBoundaryDisplay.from_estimator(
        clf, X2,
        response_method="predict",
        cmap=cmap,
        plot_method="contourf",
        grid_resolution=int(1/plot_step),
        ax=ax
    )

    # Scatter points
    classes = np.unique(y)
    has_cls = class_names is not None
    for cls, color in zip(classes, plot_colors):
        mask = (y == cls)
        lbl  = class_names[cls] if has_cls else str(cls)
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=color, edgecolor="k", s=20,
            label=lbl
        )

    # Axis labels
    if feature_names is not None:
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
    else:
        ax.set_xlabel(f"x[{i}]")
        ax.set_ylabel(f"x[{j}]")

    ax.legend(loc="lower right", fontsize="small")
    ax.set_title(f"Decision surface using features {i} & {j}")
    return ax

def plot_decision_surface_from_fitted_tree(
    clf_tree,
    X,
    feature_pair,
    y=None,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title='Decision surface (fitted tree)'
):
    """
    Plots the 2D decision surface of a fitted DecisionTreeClassifier
    and (if y is provided) overlays the true points in matching colors.

    Parameters
    ----------
    clf_tree : DecisionTreeClassifier
        A tree already fit on the full-dimensional X.
    X : array-like, shape (n_samples, n_features)
    feature_pair : tuple of two ints
        Indices of the two features to plot.
    y : array-like of shape (n_samples,), optional
        True class labels for overlaying points.
    fixed_vals : array-like of shape (n_features,), optional
        Values to fill for “unused” features. Defaults to X.mean(axis=0).
    grid_steps : int
        Number of points along each axis for the background grid.
    cmap : str or Colormap
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : matplotlib Axes
    contour : QuadContourSet
    """
    i, j = feature_pair

    # 1) build grid over the two selected features
    x_min, x_max = X[:, i].min(), X[:, i].max()
    y_min, y_max = X[:, j].min(), X[:, j].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    # 2) lift grid points back into full feature space
    if fixed_vals is None:
        fixed_vals = X.mean(axis=0)
    base = np.tile(fixed_vals, (xx.size, 1))
    base[:, i] = xx.ravel()
    base[:, j] = yy.ravel()

    # 3) predict on the grid
    Z = clf_tree.predict(base).reshape(xx.shape)

    # 4) plot decision regions
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.6)

    # 5) overlay true points (if provided), in matching colors
    if y is not None:
        norm = contour.norm
        cmap_used = contour.cmap
        for cls in np.unique(y):
            mask = (y == cls)
            color = cmap_used(norm(cls))
            ax.scatter(
                X[mask, i],
                X[mask, j],
                color=color,
                edgecolor='k',
                s=30,
                label=f'class {cls}'
            )
        ax.legend(loc='lower right')

    # 6) labels & title
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title(title)

    return ax, contour

def plot_decision_regions_lmt(
    X,
    y,
    clf_lmt,
    nodes_lmt,
    feature_pair=(0, 1),
    fill_value="mean",       # "mean" | "median" | float
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title="Decision regions (LMT)"
):
    """
    Plots the 2D decision regions of an LMT model using its class predictions.

    Parameters
    ----------
    X : array, shape (n_samples, D)
        Full dataset.
    y : array, shape (n_samples,)
        True labels (0 … J-1), used for overlay and J=number of classes.
    clf_lmt : object
        The fitted tree from fit_logistic_model_tree.
    nodes_lmt : object
        The nodes/models returned alongside clf_lmt.
    feature_pair : tuple(int i, int j)
        Which two features to plot on the x- and y-axes.
    fill_value : "mean" | "median" | float
        How to fill the other dimensions (3…D) when building the grid.
    grid_steps : int
        Resolution of the background grid.
    cmap : Colormap
        A discrete colormap, e.g. plt.cm.Paired.
    ax : matplotlib Axes, optional
    title : str
    """
    i, j = feature_pair
    D = X.shape[1]
    classes = np.unique(y)
    J = len(classes)

    # 1) build a grid over features i vs j
    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    n_grid = grid_pts.shape[0]

    # 2) lift grid back into full D-space
    if fill_value == "mean":
        default = X.mean(axis=0)
    elif fill_value == "median":
        default = np.median(X, axis=0)
    else:
        default = np.full(D, float(fill_value))

    X_grid = np.tile(default, (n_grid, 1))
    X_grid[:, i] = grid_pts[:, 0]
    X_grid[:, j] = grid_pts[:, 1]

    # 3) predict classes on the grid
    Z_flat = predict_lmt_multiclass(X_grid, clf_lmt, nodes_lmt)
    Z = Z_flat.reshape(xx.shape)

    # 4) plot the decision regions
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # discrete boundaries at class midpoints: e.g. [-0.5, 0.5, 1.5] for J=2
    levels = np.arange(J+1) - 0.5
    cf = ax.contourf(
        xx, yy, Z,
        levels=levels,
        cmap=cmap,
        alpha=0.3
    )

    # 5) overlay the true points by their true label in matching colors
    if y is not None:
        # use the colormap to get colors for each class
        norm = cf.norm
        cmap_used = cf.cmap
        for cls in classes:
            mask = (y == cls)
            color = cmap_used(norm(cls))
            # sc = ax.scatter(
            #     X[mask, i], X[mask, j],
            #     color=color,
            #     s=20,
            #     label=f'class {cls}'
            # )
            sc = ax.scatter(
                X[mask, i], X[mask, j],
                c=[cls] * np.sum(mask),
                cmap=cmap_used,
                norm=norm,
                s=20,
                edgecolor='k',
                linewidth=0.3
            )
        # ax.legend(loc='lower right')

    

    # 6) labels & title & legend
    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_title(title)

    # discrete legend
    handles, _ = sc.legend_elements()
    ax.legend(handles, [f"class {c}" for c in classes], loc="lower right")


    return ax, cf

def plot_probability_surface_tree(
    clf,
    X,
    feature_pair,
    prob_class=1,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title='Predicted probability surface'
):
    """
    Plots a 2D probability surface of a fitted classifier, colouring
    each location by P(y = prob_class) ∈ [0,1], and overlays the
    training points coloured by their own predicted probabilities.

    Parameters
    ----------
    clf : classifier with predict_proba
        Already fit on the full-dimensional X.
    X : array-like, shape (n_samples, n_features)
    feature_pair : tuple of two ints
        Indices of the two features to plot.
    prob_class : int, default=1
        Which column of predict_proba to show (e.g. 1 for P(y=1)).
    fixed_vals : array-like of shape (n_features,), optional
        Values to fill for unused features. Defaults to X.mean(axis=0).
    grid_steps : int, default=200
        Resolution of the background grid.
    cmap : str or Colormap, default='RdYlBu'
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    ax : matplotlib Axes
    contour : QuadContourSet
    """
    i, j = feature_pair

    # 1) build grid over the two selected features:
    x_min, x_max = X[:, i].min(), X[:, i].max()
    y_min, y_max = X[:, j].min(), X[:, j].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    # 2) lift grid points back into full feature space:
    if fixed_vals is None:
        fixed_vals = X.mean(axis=0)
    base = np.tile(fixed_vals, (xx.size, 1))
    base[:, i] = xx.ravel()
    base[:, j] = yy.ravel()

    # 3) predict probabilities on the grid:
    probs = clf.predict_proba(base)[:, prob_class].reshape(xx.shape)

    # 4) plot continuous probability surface:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.linspace(0, 1, 51)
    contour = ax.contourf(
        xx, yy, probs,
        levels=levels,
        cmap=cmap,
        vmin=0, vmax=1,
        alpha=0.8
    )
    plt.colorbar(contour, ax=ax, label=f'P(y={prob_class})')

    # 5) overlay training points, coloured by their predicted prob:
    point_probs = clf.predict_proba(X)[:, prob_class]
    sc = ax.scatter(
        X[:, i], X[:, j],
        c=point_probs,
        cmap=cmap,
        vmin=0, vmax=1,
        edgecolor='k',
        s=20,
        alpha=0.6
    )

    # 6) labels & title
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title(title)

    return ax, contour


def plot_probability_surface_lmt(
    clf_lmt,
    nodes_lmt,
    X,
    feature_pair,
    prob_class=1,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title='Predicted probability surface (LMT)'
):
    """
    Plots a 2D probability surface from your LMT model, colouring each
    point by P(y = prob_class) ∈ [0,1], and overlays the training points
    coloured by their own predicted probability.

    Parameters
    ----------
    clf_lmt : the object returned by fit_logistic_model_tree (e.g. clf_shallow_ext)
    nodes_lmt : the nodes structure returned alongside clf_lmt
    X : array-like, shape (n_samples, n_features)
    feature_pair : tuple of two ints (i, j), indices of features to plot
    prob_class : int, default=1
        Which class’s probability to display (0 or 1)
    fixed_vals : array-like of shape (n_features,), optional
        Values to hold the “unused” features at. Defaults to X.mean(axis=0).
    grid_steps : int, default=200
        Resolution of the background grid.
    cmap : str or Colormap, default='RdYlBu'
    ax : matplotlib Axes, optional
    title : str

    Returns
    -------
    ax : matplotlib Axes
    contour : QuadContourSet
    """
    i, j = feature_pair

    # 1) build grid in feature‐i vs feature‐j space
    x_min, x_max = X[:, i].min(), X[:, i].max()
    y_min, y_max = X[:, j].min(), X[:, j].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    # 2) lift grid back into full feature space
    if fixed_vals is None:
        fixed_vals = X.mean(axis=0)
    base = np.tile(fixed_vals, (xx.size, 1))
    base[:, i] = xx.ravel()
    base[:, j] = yy.ravel()

    # 3) get probs from your LMT
    Z_flat = predict_proba_lmt_multiclass(base, clf_lmt, nodes_lmt)[:, prob_class]
    Z_flat = np.clip(Z_flat, 0, 1)  # ensure probabilities are in [0, 1]
    Z = Z_flat.reshape(xx.shape)

    # 4) plot the continuous surface
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    
    levels = np.linspace(0, 1, 51)
    contour = ax.contourf(
        xx, yy, Z,
        levels=levels,
        cmap=cmap,
        vmin=0, vmax=1,
        alpha=0.8
    )
    #contour.set_clim(0, 1)
    cbar = plt.colorbar(contour, ax=ax, label=f'P(y={prob_class})')

    # 5) overlay the actual points, coloured by their own prob
    point_probs = predict_proba_lmt_multiclass(X, clf_lmt, nodes_lmt)[:, prob_class]
    point_probs = np.clip(point_probs, 0, 1)  # ensure probabilities are in [0, 1]
    ax.scatter(
        X[:, i], X[:, j],
        c=point_probs,
        cmap=cmap,
        vmin=0, vmax=1,
        edgecolor='k',
        s=20,
        alpha=0.6
    )

    # 6) labels & title
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title(title)

    return ax, contour


# -------------------------- #
# Preprocessing              #
# -------------------------- #
def preprocess_dataset(X: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and one-hot encode nominal features.
    
    - Numeric columns: impute missing values with the column mean.
    - Nominal (categorical) columns: impute missing values with the mode,
      then one-hot encode.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input dataset with numeric and nominal columns.
    
    Returns
    -------
    X_processed : pd.DataFrame
        Preprocessed DataFrame with no missing values and nominal
        columns replaced by one-hot encoded features.
    """
    # Identify columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    nominal_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # 1) Impute numeric
    num_imputer = SimpleImputer(strategy='mean')
    X_numeric = pd.DataFrame(
        num_imputer.fit_transform(X[numeric_cols]),
        columns=numeric_cols,
        index=X.index
    )
    
    # 2) Impute nominal
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_nominal = pd.DataFrame(
        cat_imputer.fit_transform(X[nominal_cols]),
        columns=nominal_cols,
        index=X.index
    )
    
    # 3) One-hot encode nominal
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_ohe = pd.DataFrame(
        ohe.fit_transform(X_nominal),
        columns=ohe.get_feature_names_out(nominal_cols),
        index=X.index
    )
    
    # 4) Combine numeric and encoded nominal
    X_processed = pd.concat([X_numeric, X_ohe], axis=1)
    return X_processed

# Example usage (assuming df is your DataFrame):
# df_preprocessed = preprocess_dataset(df)



# ------------------------------------------------ #
# Tree Comparison functions                        #
# ------------------------------------------------ #
def compare_tree_variants(X_train, X_test, y_train, y_test, lmt, decision=False, prob_surf=False, feature_pair=(0,1)):
    """
    Trains and evaluates four logistic model trees using different sizes and pruning strategies.
    Plots the tree structure and prints evaluation metrics.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
        Training and test feature sets.
    y_train, y_test : pd.Series or np.ndarray
        Training and test labels.
    lmt : module or object
        Module or object that provides the `construct_tree` method.
    """
    
    configs = [
        ('shallow', False, "Shallow Tree (No Pruning)"),
        ('regular', False, "Regular Tree (No Pruning)"),
        ('overfit', False, "Overfit Tree (No Pruning)"),
        ('regular', True,  "Regular Tree (With Pruning)")
    ]

    classifiers = []
    results = []

    # Tree construction
    for size, pruning, label in configs:
        clf = lmt.construct_tree(X_train, y_train, size=size, pruning=pruning)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        loss = log_loss(y_test, clf.predict_proba(X_test))

        print(f"{label} - Accuracy: {acc:.4f}, AUC: {auc:.4f}, Log Loss: {loss:.4f}")
        
        classifiers.append(clf)
        results.append((label, acc, auc, loss))

    # Plotting all four trees
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, clf, (label, _, _, _) in zip(axes.ravel(), classifiers, results):
        tree.plot_tree(clf, filled=True, ax=ax, fontsize=8)
        ax.set_title(label, fontsize=14)

    plt.tight_layout()
    plt.show()

    if decision:
        # Plotting decision surfaces of all four trees
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        for ax, clf, (size, pruning, label) in zip(axes2.ravel(), classifiers, configs):
            lmt.plot_decision_surface_from_fitted_tree(
                clf_tree=clf,
                X=X_train,
                feature_pair=feature_pair,
                y=y_train,
                fixed_vals=None,
                grid_steps=200,
                cmap='RdYlBu',
                ax=ax,
                title=label
            )

        plt.tight_layout()
        plt.show()

    if prob_surf:
        # Plotting probability surfaces for each tree
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        for ax, clf, (size, pruning, label) in zip(axes3.ravel(), classifiers, configs):
            lmt.plot_probability_surface_tree(
                clf=clf,
                X=X_train,
                feature_pair=feature_pair,
                prob_class=1,
                fixed_vals=None,
                grid_steps=200,
                cmap='RdYlBu',
                ax=ax,
                title=f'Probability Surface - {label}'
            )

        plt.tight_layout()
        plt.show()


def compare_tree_variants_multiclass(
    X_train, X_test, y_train, y_test,
    lmt, decision=False, prob_surf=False, feature_pair=(0,1)
):
    """
    Trains and evaluates four logistic model trees (3-class) with different sizes and pruning.
    Plots the tree structure and prints multiclass evaluation metrics.
    """
    configs = [
        ('shallow', False, "Shallow Tree (No Pruning)"),
        ('regular', False, "Regular Tree (No Pruning)"),
        ('overfit', False, "Overfit Tree (No Pruning)"),
        ('regular', True,  "Regular Tree (With Pruning)")
    ]

    classifiers = []

    # Construcción y evaluación
    for size, pruning, label in configs:
        clf = lmt.construct_tree(X_train, y_train, size=size, pruning=pruning)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)         # shape = (n_samples, 3)

        # Métricas multiclass
        acc   = accuracy_score(y_test, y_pred)
        loss  = log_loss(y_test, y_proba)

        print(f"{label} - Accuracy: {acc:.4f}, Log Loss: {loss:.4f}")
        classifiers.append((clf, label))

    # Plot de las cuatro estructuras
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for (clf, label), ax in zip(classifiers, axes.ravel()):
        tree.plot_tree(clf, filled=True, ax=ax, fontsize=8)
        ax.set_title(label, fontsize=14)
    plt.tight_layout()
    plt.show()

    if decision:
        # Superficies de decisión (puede funcionar para multiclass)
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        for (clf, label), ax in zip(classifiers, axes2.ravel()):
            lmt.plot_decision_surface_from_fitted_tree(
                clf_tree=clf,
                X=X_train,
                feature_pair=feature_pair,
                y=y_train,
                fixed_vals=None,
                grid_steps=200,
                cmap='RdYlBu',     # colormap con al menos 3 colores
                ax=ax,
                title=label
            )
        plt.tight_layout()
        plt.show()
    
    if prob_surf:
        # Superficies de probabilidad para cada árbol
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        for (clf, label), ax in zip(classifiers, axes3.ravel()):
            lmt.plot_probability_surface_tree(
                clf=clf,
                X=X_train,
                feature_pair=feature_pair,
                prob_class=1,  # Cambia según la clase que quieras visualizar
                fixed_vals=None,
                grid_steps=200,
                cmap='RdYlBu',
                ax=ax,
                title=f'Probability Surface - {label}'
            )
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------- #
# LMT version 1 (SimpleLogistic at every node) comparison   #
# --------------------------------------------------------- #
def compare_lmt_variants(X_train, X_test, y_train, y_test, lmt, decision=False, feature_pair=(0,1)):
    """
    Trains and evaluates four logistic model trees using different sizes and pruning strategies.
    Plots the tree structure and prints evaluation metrics.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
        Training and test feature sets.
    y_train, y_test : pd.Series or np.ndarray
        Training and test labels.
    lmt : module or object
        Module or object that provides the `construct_tree` method.
    """
    
    configs = [
        ('shallow', False, "Shallow Tree (No Pruning)"),
        ('regular', False, "Regular Tree (No Pruning)"),
        ('overfit', False, "Overfit Tree (No Pruning)"),
        ('regular', True,  "Regular Tree (With Pruning)")
    ]

    fitted = []

    # Tree construction
    for size, pruning, label in configs:
        clf_tree, node_models = lmt.fit_logistic_model_tree(X_train, y_train, size=size, pruning=pruning)
        # Hard predictions
        y_pred = lmt.predict_lmt(X_test, clf_tree, node_models)
        # extract per‐sample proba. of class 1
        y_prob = lmt.predict_proba_lmt(X_test, clf_tree, node_models)

        # 4) metrics
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        loss = log_loss(y_test, y_prob)
        print(f"{label} → Accuracy: {acc:.4f}, AUC: {auc:.4f}, Log Loss: {loss:.4f}")
        
        fitted.append((clf_tree, node_models, label))

    # Plotting all four trees
    # Prepare a 2x2 grid for the 4 decision trees
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Plotting all four trees
    for i, (clf_tree, node_models, label) in enumerate(fitted):
        lmt.plot_tree_with_linear_models(
            clf_tree=clf_tree,
            node_models=node_models,
            X=X_train,
            title=label,
            ax=axes[i]  # Use subplot axis
        )

    # Final layout and display
    plt.tight_layout()
    plt.show()


    # Plotting decision surfaces of all four trees
    if decision:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, (clf_tree, node_models, label) in zip(axes.ravel(), fitted):
            plot_decision_regions_lmt(
                X=X_train,
                y=y_train,
                clf_lmt=clf_tree,
                nodes_lmt=node_models,
                feature_pair=feature_pair,
                fill_value="mean",  # or "median" or a specific float
                grid_steps=200,
                cmap=plt.cm.RdYlBu,
                ax=ax,
                title=label
            )
        plt.tight_layout()
        plt.show()



# ------------------------------------------------------------- #
# LMT version 2 (SimpleLogistic just at root node) comparison   #
# ------------------------------------------------------------- #
def compare_lmt_variants_v2(X_train, X_test, y_train, y_test, lmt, decision=True, feature_pair=(0,1)):
    """
    Trains and evaluates four logistic model trees using different sizes and pruning strategies.
    Plots the tree structure and prints evaluation metrics.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame or np.ndarray
        Training and test feature sets.
    y_train, y_test : pd.Series or np.ndarray
        Training and test labels.
    lmt : module or object
        Module or object that provides the `construct_tree` method.
    """
    
    configs = [
        ('shallow', False, "Shallow Tree (No Pruning)"),
        ('regular', False, "Regular Tree (No Pruning)"),
        ('overfit', False, "Overfit Tree (No Pruning)"),
        ('regular', True,  "Regular Tree (With Pruning)")
    ]

    fitted = []

    # Tree construction
    for size, pruning, label in configs:
        clf_tree, node_models = lmt.fit_logistic_model_tree_v2(X_train, y_train, size=size, pruning=pruning)
        #print(node_models)
        # Hard predictions
        y_pred = lmt.predict_lmt(X_test, clf_tree, node_models)
        # 3) extract per‐sample proba. of class 1
        y_prob = lmt.predict_proba_lmt(X_test, clf_tree, node_models)

        # 4) metrics
        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        loss = log_loss(y_test, y_prob)
        print(f"{label} → Accuracy: {acc:.4f}, AUC: {auc:.4f}, Log Loss: {loss:.4f}")
        
        fitted.append((clf_tree, node_models, label))

    # Plotting all four trees
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16))
    axes = axes.flatten()

    # Plotting all four trees
    for i, (clf_tree, node_models, label) in enumerate(fitted):
        lmt.plot_tree_with_linear_models(
            clf_tree=clf_tree,
            node_models=node_models,
            X=X_train,
            title=label,
            ax=axes[i]  # Use subplot axis
        )

    # Plotting decision surfaces of all four trees
    if decision:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, (clf_tree, node_models, label) in zip(axes.ravel(), fitted):
            plot_decision_regions_lmt(
                X=X_train,
                y=y_train,
                clf_lmt=clf_tree,
                nodes_lmt=node_models,
                feature_pair=feature_pair,
                fill_value="mean",  # or "median" or a specific float
                grid_steps=200,
                cmap=plt.cm.RdYlBu,
                ax=ax,
                title=label
            )
        plt.tight_layout()
        plt.show()

# Multiclass and combined version
# def compare_lmt_variants_multiclass(
#     X_train, X_test, y_train, y_test,
#     lmt,
#     version='v1',
#     decision=False,
#     prob_surf=False,
#     feature_pair=(0,1)
# ):
#     """
#     Train & evaluate four Logistic Model Tree variants (multiclass).

#     Parameters
#     ----------
#     X_train, X_test : array-like, shape (n_samples, n_features)
#     y_train, y_test : array-like, shape (n_samples,)
#         Class labels 0..J-1.
#     lmt : module-like
#         Must provide:
#           - fit_logistic_model_tree   (for version='v1')
#           - fit_logistic_model_tree_v2(for version='v2')
#           - predict_lmt       (or predict_lmt_multiclass)
#           - predict_proba_lmt (or predict_proba_lmt_multiclass)
#           - plot_tree_with_linear_models
#           - plot_decision_surface_from_fitted_tree
#     version : {'v1','v2'}, default='v1'
#         Which fitting routine to call.
#     decision : bool, default=False
#         If True, also plot decision surfaces.
#     feature_pair : tuple(int,int)
#         Which two features to use for decision‐surface plots.

#     Notes
#     -----
#     This routine now handles multiclass metrics:
#       - Accuracy
#       - Log loss
#     """
#     configs = [
#         ('shallow', False, "Shallow Tree (No Pruning)"),
#         ('regular', False, "Regular Tree (No Pruning)"),
#         ('overfit', False, "Overfit Tree (No Pruning)"),
#         ('regular', True,  "Regular Tree (With Pruning)")
#     ]

#     fitted = []

#     # 1) Train & collect
#     for size, pruning, label in configs:
#         # choose the fitting function
#         if version == 'v2':
#             fit_fn = lmt.fit_logistic_model_tree_v2
#         else:
#             fit_fn = lmt.fit_logistic_model_tree

#         clf_tree, node_models = fit_fn(
#             X_train, y_train,
#             size=size, pruning=pruning
#         )

#         # 2) Predict
#         y_pred = lmt.predict_lmt_multiclass(X_test, clf_tree, node_models)
#         y_proba = lmt.predict_proba_lmt_multiclass(X_test, clf_tree, node_models)
#         # y_proba shape = (n_samples, J)

#         # 3) Metrics
#         acc = accuracy_score(y_test, y_pred)
#         loss = log_loss(y_test, y_proba)

#         print(f"{label} → Accuracy: {acc:.4f}, Log Loss: {loss:.4f}")

#         fitted.append((clf_tree, node_models, label))

#     # 4) Plot the four tree structures
#     fig, axes = plt.subplots(2, 2, figsize=(24, 16))
#     axes = axes.flatten()
#     for ax, (clf_tree, node_models, label) in zip(axes, fitted):
#         lmt.plot_tree_with_linear_models(
#             clf_tree=clf_tree,
#             node_models=node_models,
#             X=X_train,
#             title=label,
#             ax=ax
#         )
#     plt.tight_layout()
#     plt.show()

#     # 5) (Optional) Decision surfaces
#     # Plotting decision surfaces of all four trees
#     if decision:
#         fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#         for ax, (clf_tree, node_models, label) in zip(axes.ravel(), fitted):
#             plot_decision_regions_lmt(
#                 X=X_train,
#                 y=y_train,
#                 clf_lmt=clf_tree,
#                 nodes_lmt=node_models,
#                 feature_pair=feature_pair,
#                 fill_value="mean",  # or "median" or a specific float
#                 grid_steps=200,
#                 cmap=plt.cm.RdYlBu,
#                 ax=ax,
#                 title=label
#             )
#         plt.tight_layout()
#         plt.show()
    
#     # 6) (Optional) Probability surfaces
#     if prob_surf:
#         fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
#         for ax, (clf_tree, node_models, label) in zip(axes2.ravel(), fitted):
#             lmt.plot_probability_surface_lmt(
#                 clf_lmt=clf_tree,
#                 nodes_lmt=node_models,
#                 X=X_train,
#                 feature_pair=feature_pair,
#                 prob_class=1,  # Change as needed
#                 fixed_vals=None,
#                 grid_steps=200,
#                 cmap='RdYlBu',
#                 ax=ax,
#                 title=f'Probability Surface - {label}'
#             )  
#         plt.tight_layout()
#         plt.show()

def compare_lmt_variants_multiclass(
    X_train, X_test, y_train, y_test,
    lmt,
    version='v1',
    decision=False,
    prob_surf=False,
    feature_pair=(0, 1),
    original_models=None,
    tag="Original"
):
    """
    Evaluate and visualize 4 LMT variants for multiclass classification.

    Parameters
    ----------
    original_models : list of (clf_tree, node_models) or None
    """

    configs = [
        ('shallow', False, "Shallow Tree (No Pruning)"),
        ('regular', False, "Regular Tree (No Pruning)"),
        ('overfit', False, "Overfit Tree (No Pruning)"),
        ('regular', True,  "Regular Tree (With Pruning)")
    ]

    def evaluate_models(X_eval, models, tag):
        fitted = []
        print(f"{tag} Models Evaluation")
        for (size, pruning, label), (clf_tree, node_models) in zip(configs, models):
            y_pred = lmt.predict_lmt_multiclass(X_test, clf_tree, node_models)
            y_proba = lmt.predict_proba_lmt_multiclass(X_test, clf_tree, node_models)
            acc = accuracy_score(y_test, y_pred)
            loss = log_loss(y_test, y_proba)
            print(f"{label} → Accuracy: {acc:.4f}, Log Loss: {loss:.4f}")
            fitted.append((clf_tree, node_models, f"{tag} – {label}"))

        # Plot tree structures
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        for ax, (clf_tree, node_models, label) in zip(axes.ravel(), fitted):
            lmt.plot_tree_with_linear_models(
                clf_tree=clf_tree,
                node_models=node_models,
                X=X_eval,
                title=label,
                ax=ax
            )
        plt.tight_layout()
        plt.show()

        # Optional: Decision surfaces
        if decision:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            for ax, (clf_tree, node_models, label) in zip(axes.ravel(), fitted):
                plot_decision_regions_lmt(
                    X=X_eval,
                    y=y_train,
                    clf_lmt=clf_tree,
                    nodes_lmt=node_models,
                    feature_pair=feature_pair,
                    fill_value="mean",
                    grid_steps=200,
                    cmap=plt.cm.RdYlBu,
                    ax=ax,
                    title=label
                )
            plt.tight_layout()
            plt.show()

        # Optional: Probability surfaces
        if prob_surf:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            for ax, (clf_tree, node_models, label) in zip(axes.ravel(), fitted):
                lmt.plot_probability_surface_lmt(
                    clf_lmt=clf_tree,
                    nodes_lmt=node_models,
                    X=X_eval,
                    feature_pair=feature_pair,
                    prob_class=1,
                    fixed_vals=None,
                    grid_steps=200,
                    cmap='RdYlBu',
                    ax=ax,
                    title=f"{label} – Prob Surface"
                )
            plt.tight_layout()
            plt.show()

    # --- Fit original models if not given
    if original_models is None:
        print("⚙️ Training original models...")
        fit_fn = lmt.fit_logistic_model_tree_v2 if version == 'v2' else lmt.fit_logistic_model_tree
        original_models = []
        for size, pruning, _ in configs:
            clf_tree, node_models = fit_fn(X_train, y_train, size=size, pruning=pruning)
            original_models.append((clf_tree, node_models))

    evaluate_models(X_train, original_models, tag=tag)