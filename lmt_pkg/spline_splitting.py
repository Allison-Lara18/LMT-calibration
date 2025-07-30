"""
LMT implementation of spline splitting for decision trees.
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
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
#import logitboost_j_implementation as logitboost
from . import logitboost_j_implementation as logitboost
from . import composite_tree
from . import calibration_functions as calibration

# ------------------------
# Node Classes
# ------------------------

class LeafNode:
    def __init__(self, y, node_id):
        self.is_leaf = True
        self.n_samples = len(y)
        self.value = np.bincount(y, minlength=2)
        self.prediction = np.argmax(self.value)
        self.node_id = node_id

class DecisionNode:
    def __init__(self, feature, tau, left, right, y, node_id):
        self.is_leaf = False
        self.feature = feature
        self.tau = tau
        self.left = left
        self.right = right
        self.n_samples = len(y)
        self.value = np.bincount(y, minlength=2)
        self.node_id = node_id
        #self.prediction = np.argmax(self.value)

# ------------------------
# Utility Functions
# ------------------------

def purity(y):
    """Returns the maximum class proportion"""
    counts = np.bincount(y)
    return counts.max() / counts.sum()

def fit_logistic_spline(X, y):
    """Fits logistic regression on a 2D array with linear spline terms"""
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    return model

def compute_auc(model, X, y):
    """Computes AUC given model and input"""
    prob = model.predict_proba(X)[:, 1]

    # Just one class present
    if len(np.unique(y)) < 2:
        return 1
    return roc_auc_score(y, prob)

# ------------------------
# Recursive Tree Builder
# ------------------------

def grow_spline_tree(X, y, features, depth=0,
                     max_depth=3, min_samples_leaf=10,
                     purity_threshold=0.95, verbose=False, node_id_counter=[0]):
    """
    node_id_counter : list of 1 element acting as mutable counter (default [0])
    """
    n = len(y)

    # Stop conditions
    if depth >= max_depth or n < 2 * min_samples_leaf or purity(y) >= purity_threshold:
        node_id = node_id_counter[0]
        node_id_counter[0] += 1
        return LeafNode(y, node_id)

    # Initialize best AUC and split
    best_auc = -np.inf
    best_split = None

    # Iterate over features and potential splits
    for feature in features:
        X_j = X[:, feature]
        # Get quantiles for potential splits (taus)
        taus = np.quantile(X_j, np.linspace(0.1, 0.9, 9))

        # For each quantile, create a linear spline term and fit the model
        for tau in taus:
            # Create linear spline term
            spline_term = np.maximum(0, X_j - tau).reshape(-1, 1)
            X_feat = X_j.reshape(-1, 1)
            # Combine original feature with spline term
            X_model = np.hstack([X_feat, spline_term])

            try:
                # Fit logistic regression model
                model = fit_logistic_spline(X_model, y)
            except ValueError:
                continue
            
            # Separate data into left and right based on the split
            mask_left = X_j <= tau
            mask_right = ~mask_left

            # Check if both splits have enough samples
            if mask_left.sum() < min_samples_leaf or mask_right.sum() < min_samples_leaf:
                continue
            
            # Compute AUC for left and right splits
            auc_L = compute_auc(model, X_model[mask_left], y[mask_left])
            auc_R = compute_auc(model, X_model[mask_right], y[mask_right])
            weighted_auc = (mask_left.sum() / n) * auc_L + (mask_right.sum() / n) * auc_R

            if verbose:
                print(f"Feature {feature}, tau = {tau:.2f}, AUC_L = {auc_L:.3f}, AUC_R = {auc_R:.3f}, Weighted AUC = {weighted_auc:.3f}")

            # Update best split if this one is better
            if weighted_auc > best_auc:
                best_auc = weighted_auc
                best_split = {
                    'feature': feature,
                    'tau': tau,
                    'X_left': X[mask_left],
                    'y_left': y[mask_left],
                    'X_right': X[mask_right],
                    'y_right': y[mask_right]
                }

    # If no valid split found, return a leaf node
    if best_split is None:
        # Use the current node_id_counter to create a leaf node
        node_id = node_id_counter[0]
        node_id_counter[0] += 1
        return LeafNode(y, node_id)
    
    # Recursively grow left and right subtrees
    left_node = grow_spline_tree(best_split['X_left'], best_split['y_left'], features,
                                  depth + 1, max_depth, min_samples_leaf, purity_threshold, verbose, node_id_counter)
    right_node = grow_spline_tree(best_split['X_right'], best_split['y_right'], features,
                                   depth + 1, max_depth, min_samples_leaf, purity_threshold, verbose, node_id_counter)
    
    # Use the current node_id_counter to create a decision node
    node_id = node_id_counter[0]
    node_id_counter[0] += 1

    return DecisionNode(best_split['feature'], best_split['tau'], left_node, right_node, best_split['y_left'].tolist() + best_split['y_right'].tolist(), node_id)

# Prediction function for the tree
def predict_single(x, node):
    """
    Predicts class for a single observation x (as a pandas Series)
    using the decision tree rooted at `node`.
    """
    while not node.is_leaf:
        if x[node.feature] <= node.tau:
            node = node.left
        else:
            node = node.right
    return node.prediction

def predict(X, root):
    """
    Predicts classes for a DataFrame X using the tree rooted at `root`.
    Returns a numpy array of predictions.
    """
    return np.array([predict_single(row, root) for row in X])

# Predict probabilities functions
def predict_proba_single(x, node):
    """
    Predicts class probabilities for a single observation x (as a pandas Series)
    using the decision tree rooted at `node`.
    """
    while not node.is_leaf:
        if x[node.feature] <= node.tau:
            node = node.left
        else:
            node = node.right

    value = node.value
    total = value.sum()
    proba = value / total
    return proba

def predict_proba(X, root):
    """
    Predicts class probabilities for a DataFrame X using the tree rooted at `root`.
    Returns a numpy array of probabilities.
    """
    return np.array([predict_proba_single(row, root) for row in X])


# -------------------------- #
# Visualization Functions
# -------------------------- #
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines


def plot_custom_tree(node, depth=0, pos_x=0.5, pos_y=1.0, dx=0.2, dy=0.15, ax=None, parent_pos=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plot_custom_tree(node, depth, pos_x, pos_y, dx, dy, ax)
        plt.show()
        return

    # Línea desde el padre
    if parent_pos is not None:
        ax.add_line(mlines.Line2D([parent_pos[0], pos_x], [parent_pos[1], pos_y], color='black'))

    # Nodo
    if node.is_leaf:
        label = (
        f"Leaf\n"
        f"Predict: {node.prediction}\n"
        f"value = {node.value.tolist()}\n"
        f"n = {node.n_samples}"
        )
        color = 'lightgreen'
    else:
        label = (
        f"[{node.feature} ≤ {node.tau:.2f}]\n"
        f"depth = {depth} | n = {node.n_samples}\n"
        f"value = {node.value.tolist()}"
    )
        color = 'lightblue'

    bbox = dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor=color)
    ax.text(pos_x, pos_y, label, ha='center', va='center', bbox=bbox, fontsize=10)

    # Recursión
    if not node.is_leaf:
        dx_child = dx * 0.8  # más cerrado en profundidad
        dy_child = dy
        plot_custom_tree(node.left, depth + 1, pos_x - dx, pos_y - dy, dx_child, dy_child, ax, (pos_x, pos_y))
        plot_custom_tree(node.right, depth + 1, pos_x + dx, pos_y - dy, dx_child, dy_child, ax, (pos_x, pos_y))


def plot_decision_surface_from_custom_tree(
    clf_tree,
    X,
    feature_pair,
    y=None,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title='Decision surface (custom spline tree)',
    plot_splits=False
):
    """
    Plots 2D decision surface and optionally split lines for a custom spline-based decision tree.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    i, j = feature_pair

    x_min, x_max = X[:, i].min(), X[:, i].max()
    y_min, y_max = X[:, j].min(), X[:, j].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    if fixed_vals is None:
        fixed_vals = X.mean(axis=0)

    base = np.tile(fixed_vals, (xx.size, 1))
    base[:, i] = xx.ravel()
    base[:, j] = yy.ravel()

    Z = predict(base, clf_tree).reshape(xx.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    contour = ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.6)

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

    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title(title)

    # Plot split lines if requested
    if plot_splits:
        def add_splits(node):
            if node.is_leaf:
                return
            if node.feature == i:
                ax.axvline(x=node.tau, color='black', linestyle='--', linewidth=1)
            elif node.feature == j:
                ax.axhline(y=node.tau, color='black', linestyle='--', linewidth=1)
            add_splits(node.left)
            add_splits(node.right)

        add_splits(clf_tree)

    return ax, contour

def plot_probability_surface_tree(
    clf_tree,
    X,
    feature_pair,
    prob_class=1,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title='Predicted probability surface',
    plot_splits=False
):
    """
    Plots a 2D probability surface for a custom decision tree,
    showing P(y = prob_class) ∈ [0,1], with optional split lines.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    i, j = feature_pair

    # 1) Build grid
    x_min, x_max = X[:, i].min(), X[:, i].max()
    y_min, y_max = X[:, j].min(), X[:, j].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    # 2) Fill unused features with fixed values
    if fixed_vals is None:
        fixed_vals = X.mean(axis=0)
    base = np.tile(fixed_vals, (xx.size, 1))
    base[:, i] = xx.ravel()
    base[:, j] = yy.ravel()

    # 3) Predict probabilities
    probs = predict_proba(base, clf_tree)[:, prob_class].reshape(xx.shape)

    # 4) Plot probability surface
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

    # 5) Overlay points colored by their predicted prob
    point_probs = predict_proba(X, clf_tree)[:, prob_class]
    ax.scatter(
        X[:, i], X[:, j],
        c=point_probs,
        cmap=cmap,
        vmin=0, vmax=1,
        edgecolor='k',
        s=20,
        alpha=0.6
    )

    # 6) Labels and title
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title(title)

    # 7) Optional: Draw split lines
    if plot_splits:
        def draw_splits(node):
            if node.is_leaf:
                return
            if node.feature == i:
                ax.axvline(x=node.tau, color='black', linestyle='--', linewidth=1)
            elif node.feature == j:
                ax.axhline(y=node.tau, color='black', linestyle='--', linewidth=1)
            draw_splits(node.left)
            draw_splits(node.right)

        draw_splits(clf_tree)

    return ax, contour


# -------------------------- #
# LMT with Spline Splitting
# -------------------------- #
def fit_logistic_model_tree_custom(
    X, y,
    features,
    max_depth=3,
    min_samples_leaf=10,
    purity_threshold=0.95,
    verbose=False,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0
):
    """
    Entrena un árbol custom con LogitBoost en cada nodo (raíz con SimpleLogistic).

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    features : list of feature indices
    max_depth : int
    min_samples_leaf : int
    purity_threshold : float
    verbose : bool
    lb_* : parámetros para LogitBoost

    Returns
    -------
    root_node : raíz del árbol (TreeNode)
    node_models : dict con node_id -> {learners, J, M_star, cv_errors}
    """

    node_models = {}
    node_counter = [0]  # contador mutable para asignar node_id

    # Paso 1: construir el árbol base (con node_id)
    root_node = grow_spline_tree(
        X, y,
        features=features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        purity_threshold=purity_threshold,
        verbose=verbose,
        node_id_counter=node_counter
    )

    # Paso 2: Entrenar modelo en la raíz con validación cruzada (SimpleLogistic)
    learners_root, J_root, M_star, cv_errs = logitboost.simple_logistic_fit(
        X, y,
        n_estimators=lb_n_estimators,
        eps=lb_eps,
        cv_splits=lb_cv_splits,
        warm_start=None,
        random_state=lb_random_state
    )

    # Guardar modelo de la raíz
    node_models[root_node.node_id] = {
        'learners': learners_root,
        'J': J_root,
        'M_star': M_star,
        'cv_errors': cv_errs
    }

    # Paso 3: función recursiva para entrenar hijos
    def recurse(node, X_sub, y_sub, warm_start):
        if node.is_leaf:
            return

        # Dividir datos según el split del nodo actual
        mask_left = X_sub[:, node.feature] <= node.tau
        mask_right = ~mask_left

        for child_node, child_mask in zip([node.left, node.right], [mask_left, mask_right]):
            X_child = X_sub[child_mask]
            y_child = y_sub[child_mask]

            # Entrenar LogitBoost clásico con M_star y warm_start
            learners, J = logitboost.logitboost_fit(
                X_child, y_child,
                n_estimators=M_star,
                eps=lb_eps,
                warm_start=warm_start
            )

            node_models[child_node.node_id] = {
                'learners': learners,
                'J': J,
                'M_star': M_star,
                'cv_errors': None
            }

            # Recursión
            recurse(child_node, X_child, y_child, warm_start=(learners, J))

    # Paso 4: comenzar desde la raíz
    recurse(root_node, X, y, warm_start=(learners_root, J_root))

    return root_node, node_models

# Predicting function for the custom tree with LogitBoost models
def predict_lmt_custom(X, root_node, node_models):
    """
    Predict class labels for a custom Logistic Model Tree.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Test samples.
    root_node : TreeNode
        Root of the custom decision tree.
    node_models : dict
        Maps node_id -> { 'learners': [...], 'J': int, ... }.

    Returns
    -------
    preds : ndarray of shape (n_samples,)
        Predicted class indices.
    """
    preds = np.empty(len(X), dtype=int)

    for i, x_i in enumerate(X):
        node = root_node
        while not node.is_leaf:
            if x_i[node.feature] <= node.tau:
                node = node.left
            else:
                node = node.right

        node_id = node.node_id
        learners = node_models[node_id]['learners']
        J = node_models[node_id]['J']
        p = logitboost.logitboost_predict(x_i.reshape(1, -1), learners, J)
        preds[i] = p[0]

    return preds



def predict_proba_lmt_custom(X, root_node, node_models):
    """
    Predict probability distributions for a custom LMT.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    root_node : TreeNode
        Root of the custom decision tree.
    node_models : dict mapping node_id -> { 'learners': [...], 'J': int, ... }

    Returns
    -------
    proba_matrix : ndarray of shape (n_samples, J_global)
        Predicted class probabilities for each sample.
    """
    J_global = max(m['J'] for m in node_models.values())
    proba_matrix = np.zeros((len(X), J_global), dtype=float)

    for i, x_i in enumerate(X):
        node = root_node
        while not node.is_leaf:
            if x_i[node.feature] <= node.tau:
                node = node.left
            else:
                node = node.right

        node_id = node.node_id
        learners = node_models[node_id]['learners']
        J = node_models[node_id]['J']
        p = logitboost.logitboost_predict_proba(x_i.reshape(1, -1), learners, J)
        proba_matrix[i, :J] = p[0]

    return proba_matrix


# --------------------------------------- #
# Visualization Functions for custom LMT
# --------------------------------------- #
def compute_tree_depth(node):
    if node.is_leaf:
        return 1
    return 1 + max(compute_tree_depth(node.left), compute_tree_depth(node.right))

def plot_custom_tree_with_models(
    node,
    node_models,
    X,
    title="Custom LMT with LogitBoost Models",
    model_threshold=1e-6,
    show_internal=False,
    depth=0,
    pos_x=0.5,
    pos_y=0.95,
    dx=0.2,
    dy=0.15,
    ax=None,
    parent_pos=None
):
    """
    Plots a custom decision tree (built from LeafNode / DecisionNode)
    and overlays the logitboost model formulas in each leaf (or optionally in all nodes).
    """
    if ax is None:
        depth_max = compute_tree_depth(node)
        fig_height = 1.5 + depth_max * dy * 5  # altura proporcional
        fig, ax = plt.subplots(figsize=(20, fig_height))
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)  # deja un pequeño margen para el título
        ax.set_title(title, fontsize=14, y=1.02)  # empuja título hacia arriba
        plot_custom_tree_with_models(node, node_models, X, title, model_threshold, show_internal, depth, pos_x, pos_y, dx, dy, ax)
        plt.tight_layout()
        plt.show()
        return

    # Draw edge from parent
    if parent_pos is not None:
        ax.add_line(mlines.Line2D([parent_pos[0], pos_x], [parent_pos[1], pos_y], color='black'))

    # Get linear model if available
    model_str = ""
    if node.node_id in node_models:
        mdl = node_models[node.node_id]
        intercepts, coefs = logitboost.extract_linear_models(mdl['learners'], mdl['J'], X.shape[1])
        lines = []
        for j in range(mdl['J']):
            b = intercepts[j]
            parts = [f"F{j}(x)={b:.2f}"]
            for k in range(X.shape[1]):
                a = coefs[j, k]
                if abs(a) > model_threshold:
                    parts.append(f"{'+' if a >= 0 else '-'}{abs(a):.2f}*x[{k}]")
            lines.append(" ".join(parts))
        model_str = "\n" + "\n".join(lines)

    # Node label
    if node.is_leaf:
        label = (
            f"Leaf\n"
            f"Predict: {node.prediction}\n"
            f"value = {node.value.tolist()}\n"
            f"n = {node.n_samples}"
        )
        if show_internal or True:
            label += model_str
        color = 'lightgreen'
    else:
        label = (
            f"[x[{node.feature}] ≤ {node.tau:.2f}]\n"
            f"depth = {depth} | n = {node.n_samples}\n"
            f"value = {node.value.tolist()}"
        )
        if show_internal:
            label += model_str
        color = 'lightblue'

    # Draw the node
    bbox = dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor=color)
    ax.text(pos_x, pos_y, label, ha='center', va='center', bbox=bbox, fontsize=10)

    # Recursive calls
    if not node.is_leaf:
        dx_child = dx * 0.8
        dy_child = dy
        plot_custom_tree_with_models(node.left, node_models, X, title, model_threshold, show_internal, depth + 1,
                                     pos_x - dx, pos_y - dy, dx_child, dy_child, ax, (pos_x, pos_y))
        plot_custom_tree_with_models(node.right, node_models, X, title, model_threshold, show_internal, depth + 1,
                                     pos_x + dx, pos_y - dy, dx_child, dy_child, ax, (pos_x, pos_y))


def plot_decision_regions_custom_tree(
    X,
    y,
    root_node,
    node_models,
    feature_pair=(0, 1),
    show_scatter=True,
    fill_value="mean",  # "mean" | "median" | float
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title="Decision regions (Custom LMT)"
):
    """
    Plots the 2D decision regions of a custom Logistic Model Tree using its predictions.

    Parameters
    ----------
    X : ndarray (n_samples, D)
        Full dataset.
    y : ndarray (n_samples,)
        True labels (0 … J-1), used for overlay and J=number of classes.
    root_node : custom tree root node
        The root of the DecisionNode / LeafNode structure.
    node_models : dict
        Mapping node_id -> {'learners', 'J', ...}.
    feature_pair : tuple(int i, int j)
        Features to plot on x/y axes.
    fill_value : str | float
        How to fill remaining dimensions.
    grid_steps : int
        Grid resolution.
    cmap : str or Colormap
        Matplotlib colormap.
    ax : matplotlib axis
        Optional axis for subplotting.
    title : str
    """
    i, j = feature_pair
    D = X.shape[1]
    classes = np.unique(y)
    J = len(classes)

    # 1) Grid in 2D
    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    n_grid = grid_pts.shape[0]

    # 2) Fill rest of features
    if fill_value == "mean":
        default = X.mean(axis=0)
    elif fill_value == "median":
        default = np.median(X, axis=0)
    else:
        default = np.full(D, float(fill_value))

    X_grid = np.tile(default, (n_grid, 1))
    X_grid[:, i] = grid_pts[:, 0]
    X_grid[:, j] = grid_pts[:, 1]

    # 3) Predict using your custom tree
    Z_flat = predict_lmt_custom(X_grid, root_node, node_models)
    Z = Z_flat.reshape(xx.shape)

    # 4) Plot regions
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    levels = np.arange(J + 1) - 0.5
    cf = ax.contourf(xx, yy, Z, levels=levels, cmap=cmap, alpha=0.3)

    # 5) Overlay scatter
    if show_scatter:
        norm = cf.norm
        cmap_used = cf.cmap
        for cls in classes:
            mask = (y == cls)
            color = cmap_used(norm(cls))
            sc = ax.scatter(
                X[mask, i], X[mask, j],
                c=[cls] * np.sum(mask),
                cmap=cmap_used,
                norm=norm,
                s=20,
                edgecolor='k',
                linewidth=0.3
            )
        handles, _ = sc.legend_elements()
        ax.legend(handles, [f"class {c}" for c in classes], loc="lower right")

    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_title(title)

    return ax, cf

def plot_probability_surface_custom_tree(
    root_node,
    node_models,
    X,
    feature_pair=(0,1),
    prob_class=1,
    fixed_vals=None,
    grid_steps=200,
    cmap='RdYlBu',
    ax=None,
    title='Predicted probability surface (Custom LMT)',
    show_scatter=True
):
    """
    Plots a 2D probability surface from your custom Logistic Model Tree (LMT),
    colouring each point by P(y = prob_class), and overlays the training points
    coloured by their own predicted probability.

    Parameters
    ----------
    root_node : root of the custom tree (from fit_logistic_model_tree_custom)
    node_models : dict mapping node_id to trained LogitBoost models
    X : array-like, shape (n_samples, n_features)
        Training data
    feature_pair : tuple of two ints (i, j)
        Indices of features to plot
    prob_class : int, default=1
        Class index to show probability surface for
    fixed_vals : array-like (n_features,), optional
        Default values for non-displayed dimensions
    grid_steps : int
        Grid resolution
    cmap : str or colormap
    ax : matplotlib Axes
    title : str
    show_scatter : bool
        Whether to overlay the training points
    """
    i, j = feature_pair

    # 1) Build 2D grid
    x_min, x_max = X[:, i].min(), X[:, i].max()
    y_min, y_max = X[:, j].min(), X[:, j].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps)
    )

    # 2) Lift grid to full D-dimensional space
    if fixed_vals is None:
        fixed_vals = X.mean(axis=0)
    base = np.tile(fixed_vals, (xx.size, 1))
    base[:, i] = xx.ravel()
    base[:, j] = yy.ravel()

    # 3) Predict probabilities using your tree
    Z_flat = predict_proba_lmt_custom(base, root_node, node_models)[:, prob_class]
    Z_flat = np.clip(Z_flat, 0, 1)
    Z = Z_flat.reshape(xx.shape)

    # 4) Plot surface
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    levels = np.linspace(0, 1, 51)
    contour = ax.contourf(
        xx, yy, Z,
        levels=levels,
        cmap=cmap,
        vmin=0, vmax=1,
        alpha=0.8
    )
    cbar = plt.colorbar(contour, ax=ax, label=f'P(y={prob_class})')

    # 5) Overlay training points colored by their predicted probability
    if show_scatter:
        pt_probs = predict_proba_lmt_custom(X, root_node, node_models)[:, prob_class]
        pt_probs = np.clip(pt_probs, 0, 1)
        ax.scatter(
            X[:, i], X[:, j],
            c=pt_probs,
            cmap=cmap,
            vmin=0, vmax=1,
            edgecolor='k',
            s=20,
            alpha=0.6
        )

    # 6) Labels
    ax.set_xlabel(f'Feature {i}')
    ax.set_ylabel(f'Feature {j}')
    ax.set_title(title)

    return ax, contour


# -------------------------- #
# Regression Pruning
# -------------------------- #
from sklearn.metrics import roc_auc_score
from collections import deque
from copy import deepcopy
import numpy as np

def get_sample_to_node_mask(root_node, X):
    """
    Given a tree built with DecisionNode and LeafNode objects,
    return a dictionary that maps each node_id to a boolean mask over the samples in X.
    """
    n_samples = X.shape[0]
    node_to_samples = {}

    for i in range(n_samples):
        x_i = X[i]
        node = root_node
        path = []

        while not node.is_leaf:
            path.append(node.node_id)
            f = node.feature
            tau = node.tau  # if this is a spline function, call tau(x)

            # If tau is a callable (e.g., spline), evaluate it
            if callable(tau):
                go_left = x_i[f] <= tau(x_i[f])
            else:
                go_left = x_i[f] <= tau

            node = node.left if go_left else node.right

        # Add the final leaf as well
        path.append(node.node_id)

        # Mark sample i as belonging to all nodes along its path
        for node_id in path:
            if node_id not in node_to_samples:
                node_to_samples[node_id] = np.zeros(n_samples, dtype=bool)
            node_to_samples[node_id][i] = True

    return node_to_samples


def regression_pruning_spline_bfs(
    X,
    y,
    root_node,
    node_models,
    threshold,
    multiclass=False,
    average='macro',
    verbose=False
):
    """
    Prunes a spline-based LMT tree using AUC performance at each node.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    root_node : Node
        Root of the tree structure (custom tree).
    node_models : dict
        node_id → {'learners': ..., 'J': ...}
    threshold : float
        Minimum AUC to keep a node.
    multiclass : bool
    average : str
        Average type for multiclass AUC.
    verbose : bool

    Returns
    -------
    pruned_root : Node
        Root of the pruned tree.
    pruned_node_models : dict
        node_id → model info, excluding pruned nodes.
    """

    pruned_root = deepcopy(root_node)
    pruned_node_models = deepcopy(node_models)
    sample_masks = get_sample_to_node_mask(pruned_root, X)

    # BFS traversal with parent tracking
    queue = deque([(pruned_root, None, None)])  # (node, parent, is_left)

    while queue:
        node, parent, is_left = queue.popleft()

        if node is None or node.is_leaf:
            continue

        node_id = node.node_id
        mask = sample_masks.get(node_id, np.zeros(len(y), dtype=bool))
        X_node = X[mask]
        y_node = y[mask]

        # AUC calculation
        if len(np.unique(y_node)) < 2:
            perf = 1.0
        else:
            model_info = pruned_node_models[node_id]
            learners = model_info['learners']
            J = model_info['J']

            p = logitboost.logitboost_predict_proba(X_node, learners, J)
            perf = roc_auc_score(y_node, p if multiclass else p[:, 1], average=average if multiclass else 'macro')

        if verbose:
            print(f"[Node {node_id}] AUC = {perf:.4f}")

        if perf > threshold:
            if verbose:
                print(f"→ Pruning node {node_id} (AUC={perf:.4f})")

            # Convert to real LeafNode
            new_leaf = LeafNode(y_node, node_id=node.node_id)

            # Replace in parent
            if parent is None:
                pruned_root = new_leaf
            else:
                if is_left:
                    parent.left = new_leaf
                else:
                    parent.right = new_leaf

            # Clean up children models
            if node.left:
                pruned_node_models.pop(node.left.node_id, None)
            if node.right:
                pruned_node_models.pop(node.right.node_id, None)

            # Remove model for this node if needed
            # (optional: keep it for leaf prediction)
        else:
            queue.append((node.left, node, True))
            queue.append((node.right, node, False))

    return pruned_root, pruned_node_models

def pipeline_spline_tree_lmt(
    X_train, y_train, X_test, y_test,
    features=[0, 1],
    max_depth=6,
    min_samples_leaf=10,
    purity_threshold=0.95,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0,
    pruning_threshold=0.8,
    multiclass=False,
    average='macro',
    verbose=False
):
    """
    Full pipeline to fit a custom Logistic Model Tree with LogitBoost and prune it.
    
    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    features : list of feature indices to use
    max_depth, min_samples_leaf, purity_threshold : tree parameters
    lb_* : LogitBoost parameters
    pruning_threshold : AUC threshold for pruning
    multiclass : whether the problem is multiclass
    average : averaging method for AUC
    verbose : whether to print progress
    """
    # Step 1: Fit the custom Logistic Model Tree
    tree, node_models = fit_logistic_model_tree_custom(
        X_train, y_train,
        features=features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        purity_threshold=purity_threshold,
        verbose=verbose,
        lb_n_estimators=lb_n_estimators,
        lb_eps=lb_eps,
        lb_cv_splits=lb_cv_splits,
        lb_random_state=lb_random_state
    )

    # Step 2: Prune the tree based on AUC performance
    pruned_tree, pruned_node_models = regression_pruning_spline_bfs(
        X_train, y_train, tree, node_models,
        threshold=pruning_threshold,
        multiclass=multiclass,
        average=average,
        verbose=verbose
    )

    # Step 3: Predict and evaluate on test set
    y_pred = predict_lmt_custom(X_test, pruned_tree, pruned_node_models)
    accuracy = np.mean(y_pred == y_test)

    return pruned_tree, pruned_node_models, accuracy