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

def compute_brier_score(model, X, y):
    """Computes Brier score given model and input"""
    prob = model.predict_proba(X)[:, 1]
    return np.mean((y - prob) ** 2)

# ------------------------
# Recursive Tree Builder
# ------------------------

def grow_spline_tree(X, y, features, depth=0,
                     max_depth=3, min_samples_leaf=10,
                     purity_threshold=0.95, 
                     metric='auc',
                     verbose=False, node_id_counter=[0]):
    """
    Recursive function to grow a spline tree.
    Params:
    X : array-like, shape (n_samples, n_features)
        The input features.
    y : array-like, shape (n_samples,)
        The target values.
    features : list
        List of feature indices to consider for splitting.
    depth : int, (default=0)
        The current depth of the tree.
    max_depth : int, optional (default=3)
        The maximum depth of the tree.
    min_samples_leaf : int, optional (default=10)
        The minimum number of samples required to be at a leaf node.
    purity_threshold : float, optional (default=0.95)
        The threshold for node purity.
    metric : str, optional (default='auc')
        The metric to optimize ('auc' or 'brier').
    verbose : bool, optional (default=False)
        Whether to print verbose output.
    node_id_counter : list, optional (default=[0])
        A mutable counter for assigning node IDs.

    Returns:
    LeafNode or DecisionNode
    """
    n = len(y)

    # Stop conditions
    if depth >= max_depth or n < 2 * min_samples_leaf or purity(y) >= purity_threshold:
        node_id = node_id_counter[0]
        node_id_counter[0] += 1
        return LeafNode(y, node_id)

    # Initialize best metric and split
    if metric == 'auc':
        best_metric = -np.inf

    if metric == 'brier':
        best_metric = np.inf

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
            if metric == 'auc':
                metric_L = compute_auc(model, X_model[mask_left], y[mask_left])
                metric_R = compute_auc(model, X_model[mask_right], y[mask_right])
                weighted_metric = (mask_left.sum() / n) * metric_L + (mask_right.sum() / n) * metric_R

                if verbose:
                    print(f"Feature {feature}, tau = {tau:.2f}, AUC_L = {metric_L:.3f}, AUC_R = {metric_R:.3f}, Weighted AUC = {weighted_metric:.3f}")

                # Update best split if this one is better
                if weighted_metric > best_metric:
                    best_metric = weighted_metric
                    best_split = {
                        'feature': feature,
                        'tau': tau,
                        'X_left': X[mask_left],
                        'y_left': y[mask_left],
                        'X_right': X[mask_right],
                        'y_right': y[mask_right]
                    }

            elif metric == 'brier':
                metric_L = compute_brier_score(model, X_model[mask_left], y[mask_left])
                metric_R = compute_brier_score(model, X_model[mask_right], y[mask_right])
                weighted_metric = (mask_left.sum() / n) * metric_L + (mask_right.sum() / n) * metric_R

                if verbose:
                    print(f"Feature {feature}, tau = {tau:.2f}, Brier_L = {metric_L:.3f}, Brier_R = {metric_R:.3f}, Weighted Brier = {weighted_metric:.3f}")
                
                # Update best split if this one is better (lower Brier is better)
                if weighted_metric < best_metric:
                    best_metric = weighted_metric
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
                                  depth + 1, max_depth, min_samples_leaf, purity_threshold, metric=metric, verbose=verbose, node_id_counter=node_id_counter)
    right_node = grow_spline_tree(best_split['X_right'], best_split['y_right'], features,
                                   depth + 1, max_depth, min_samples_leaf, purity_threshold, metric=metric, verbose=verbose, node_id_counter=node_id_counter)

    # Use the current node_id_counter to create a decision node
    node_id = node_id_counter[0]
    node_id_counter[0] += 1

    return DecisionNode(best_split['feature'], best_split['tau'], left_node, right_node, best_split['y_left'].tolist() + best_split['y_right'].tolist(), node_id)

# Prediction function for the tree
def predict_single(x, node):
    """
    Predicts class for a single observation x (as a pandas Series)
    using the decision tree rooted at `node`.
    Params:
    - x : pandas Series
        The input features for the observation to predict.
    - node : DecisionNode
        The root node of the decision tree.
    
    Returns:
    - prediction : int
        The predicted class for the observation.
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
    Params:
    - X : DataFrame
        The input features for the observations to predict.
    - root : DecisionNode
        The root node of the decision tree.

    Returns:
    - predictions : numpy array
        The predicted classes for the observations.
    """
    return np.array([predict_single(row, root) for row in X])

# Predict probabilities functions
def predict_proba_single(x, node):
    """
    Predicts class probabilities for a single observation x (as a pandas Series)
    using the decision tree rooted at `node`.
    Params:
    - x : pandas Series
        The input features for the observation to predict.
    - node : DecisionNode
        The root node of the decision tree.

    Returns:
    - prediction : int
        The predicted class probabilities for the observation.
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
    Params:
    - X : DataFrame
        The input features for the observations to predict.
    - root : DecisionNode
        The root node of the decision tree.

    Returns:
    - probabilities : numpy array
        The predicted class probabilities for the observations.
    """
    return np.array([predict_proba_single(row, root) for row in X])


# -------------------------- #
# Visualization Functions
# -------------------------- #
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines


def plot_custom_tree(node, depth=0, pos_x=0.5, pos_y=1.0, dx=0.2, dy=0.15, ax=None, parent_pos=None):
    """
    Plots a custom decision tree based on spline.
    Params:
    - node : DecisionNode
        The root node of the decision tree.
    - depth : int
        The current depth of the tree (used for positioning).
    - pos_x : float
        The x-coordinate for the current node's position.
    - pos_y : float
        The y-coordinate for the current node's position.
    - dx : float
        The horizontal distance between sibling nodes.
    - dy : float
        The vertical distance between levels.
    - ax : matplotlib.axes.Axes, optional
        The axes to plot on (if None, a new figure is created).
    - parent_pos : tuple, optional
        The position of the parent node (for drawing edges).

    Returns:
    - None
    """
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
    Params:
    - clf_tree : Trained custom tree
    - X : DataFrame
        The input features for the observations to predict.
    - feature_pair : tuple
        The pair of features to plot (e.g., (0, 1) for the first two features).
    - y : Series, optional
        The true labels for the observations (used for coloring points).
    - fixed_vals : array-like, optional
        Fixed values for the features not in `feature_pair`.
    - grid_steps : int, optional
        The number of steps in the grid for plotting.
    - cmap : str, optional
        The colormap to use for the plot.
    - ax : matplotlib.axes.Axes, optional
        The axes to plot on (if None, a new figure is created).
    - title : str, optional
        The title of the plot.
    - plot_splits : bool, optional
        Whether to plot the split lines of the tree.

    Returns:
    - ax : matplotlib.axes.Axes
        The axes with the plot.
    - contour : matplotlib.contour.QuadContourSet
        The contour set for the decision surface.
    """

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
            """
            Recursively adds split lines for the decision tree.
            """
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
    Params:
    - clf_tree : Trained custom tree based on linear splines.
    - X : DataFrame
        The input features for the observations to predict.
    - feature_pair : tuple
        The pair of features to plot (e.g., (0, 1) for the first two features).
    - prob_class : int
        The class label for which to plot the probability surface.
    - fixed_vals : array-like, optional
        Fixed values for the features not in `feature_pair`.
    - grid_steps : int, optional
        The number of steps in the grid for plotting.
    - cmap : str, optional
        The colormap to use for the plot.
    - ax : matplotlib.axes.Axes, optional
        The axes to plot on (if None, a new figure is created).
    - title : str, optional
        The title of the plot.
    - plot_splits : bool, optional
        Whether to plot the split lines of the tree.

    Returns:
    - ax : matplotlib.axes.Axes
        The axes with the plot.
    - contour : matplotlib.contour.QuadContourSet
        The contour set for the decision surface.
    """
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
    metric='auc',
    verbose=False,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0
):
    """
    Train a custom linear spline based tree with a LogitBoost model at each node.
    In the root node, a SimpleLogistic model is trained (LogitBoost + 5-fold cross-validation).
    Parameters:
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    features : list of feature indices
    max_depth : int
    min_samples_leaf : int
    purity_threshold : float
    verbose : bool
    lb_* : params for LogitBoost

    Returns:
    root_node : root of the tree (TreeNode)
    node_models : dict with node_id -> {learners, J, M_star, cv_errors}
    """

    node_models = {}
    node_counter = [0]  # mutable counter to assign node_id

    # Step 1: build the base tree (with node_id)
    root_node = grow_spline_tree(
        X, y,
        features=features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        purity_threshold=purity_threshold,
        metric=metric,
        verbose=verbose,
        node_id_counter=node_counter
    )

    # Step 2: Train root model with cross-validation (SimpleLogistic)
    learners_root, J_root, M_star, cv_errs = logitboost.simple_logistic_fit(
        X, y,
        n_estimators=lb_n_estimators,
        eps=lb_eps,
        cv_splits=lb_cv_splits,
        warm_start=None,
        random_state=lb_random_state
    )

    # Save root model
    node_models[root_node.node_id] = {
        'learners': learners_root,
        'J': J_root,
        'M_star': M_star,
        'cv_errors': cv_errs
    }

    # Step 3: Recursive function to train children
    def recurse(node, X_sub, y_sub, warm_start):
        """
        Recursively train child nodes.
        """
        if node.is_leaf:
            return

        # Split data according to the current node's split
        mask_left = X_sub[:, node.feature] <= node.tau
        mask_right = ~mask_left

        for child_node, child_mask in zip([node.left, node.right], [mask_left, mask_right]):
            X_child = X_sub[child_mask]
            y_child = y_sub[child_mask]

            # Train classic LogitBoost with M_star and warm_start
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

            # Recursion
            recurse(child_node, X_child, y_child, warm_start=(learners, J))

    # Step 4: Start from the root
    recurse(root_node, X, y, warm_start=(learners_root, J_root))

    return root_node, node_models

# Predicting function for the custom tree with LogitBoost models
def predict_lmt_custom(X, root_node, node_models):
    """
    Predict class labels for a custom Logistic Model Tree.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Test samples.
    root_node : TreeNode
        Root of the custom decision tree.
    node_models : dict
        Maps node_id -> { 'learners': [...], 'J': int, ... }.

    Returns:
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

    Parameters:
    X : ndarray of shape (n_samples, n_features)
    root_node : TreeNode
        Root of the custom decision tree.
    node_models : dict mapping node_id -> { 'learners': [...], 'J': int, ... }

    Returns:
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
    """
    Compute the depth of the tree.
    """
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
    Params:
    - node: The current node to plot.
    - node_models: The models associated with each node.
    - X: The feature matrix used for training.
    - title: The title of the plot.
    - model_threshold: The threshold for displaying model terms.
    - show_internal: Whether to show internal node models.
    - depth: The current depth in the tree.
    - pos_x: The x position for the current node.
    - pos_y: The y position for the current node.
    - dx: The x offset for child nodes.
    - dy: The y offset for child nodes.
    - ax: The matplotlib axis to plot on.
    - parent_pos: The position of the parent node.

    Returns:
    None
    """
    if ax is None:
        depth_max = compute_tree_depth(node)
        fig_height = 1.5 + depth_max * dy * 5  # proportional height
        fig, ax = plt.subplots(figsize=(20, fig_height))
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)  # leave a small margin for the title
        ax.set_title(title, fontsize=14, y=1.02)  # push title up
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

    Parameters:
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

    Returns:
    ax : matplotlib axis
        The axis with the plotted decision regions.
    cf : ContourSet
        The contour set created by the contourf plot.
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

    Parameters:
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

    Returns:
    ax : matplotlib axis
        The axis with the plotted probability surface.
    contour : ContourSet
        The contour set created by the contourf plot.
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
    Params:
    - root_node: The root node of the tree.
    - X: The feature matrix.

    Returns:
    - node_to_samples: A dictionary mapping node_id to boolean masks for samples in X.
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

    Parameters:
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

    Returns:
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


# ---------------------------------------------------------------- #
# NEW                                                              #
# Gain regression pruning and cross-validation for selecting delta #
# ---------------------------------------------------------------- #
from copy import deepcopy
from collections import deque
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import KFold

def regression_pruning_gain_local(
    X, y, root_node, node_models, delta,
    multiclass=False, average='macro', verbose=False
):
    """
    Prunes a tree using local AUC gain from splitting: compares parent vs. weighted AUC of children.
    If the weighted AUC of the children does not improve over the parent by at least delta, prune.

    Parameters:
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    root_node : Node (custom tree)
    node_models : dict, node_id → model info
    delta : float, minimum AUC gain required to keep a split
    multiclass : bool
    average : str
    verbose : bool

    Returns:
    pruned_root : Node
    pruned_node_models : dict, pruned node models
    """
    pruned_root = deepcopy(root_node)
    pruned_node_models = deepcopy(node_models)
    sample_masks = get_sample_to_node_mask(pruned_root, X)
    queue = deque([(pruned_root, None, None)])

    while queue:
        node, parent, is_left = queue.popleft()
        if node is None or node.is_leaf:
            continue

        node_id = node.node_id
        mask = sample_masks.get(node_id, np.zeros(len(y), dtype=bool))
        X_node, y_node = X[mask], y[mask]

        if len(np.unique(y_node)) < 2 or len(y_node) < 5:
            queue.append((node.left, node, True))
            queue.append((node.right, node, False))
            continue

        model_info = pruned_node_models[node_id]
        learners = model_info['learners']
        J = model_info['J']
        p_parent = logitboost.logitboost_predict_proba(X_node, learners, J)
        auc_parent = roc_auc_score(y_node, p_parent if multiclass else p_parent[:, 1],
                                   average=average if multiclass else 'macro')

        left_mask = sample_masks.get(node.left.node_id, np.zeros(len(y), dtype=bool)) if node.left else None
        right_mask = sample_masks.get(node.right.node_id, np.zeros(len(y), dtype=bool)) if node.right else None

        if left_mask is None or right_mask is None:
            continue

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        if len(np.unique(y_left)) < 2 or len(y_left) < 5 or len(np.unique(y_right)) < 2 or len(y_right) < 5:
            continue

        auc_left = roc_auc_score(
            y_left,
            logitboost.logitboost_predict_proba(X_left, pruned_node_models[node.left.node_id]['learners'], J)[:, 1]
        )
        auc_right = roc_auc_score(
            y_right,
            logitboost.logitboost_predict_proba(X_right, pruned_node_models[node.right.node_id]['learners'], J)[:, 1]
        )

        n_left, n_right = len(y_left), len(y_right)
        weighted_auc_children = (n_left * auc_left + n_right * auc_right) / (n_left + n_right)
        gain = weighted_auc_children - auc_parent

        if verbose:
            print(f"[Node {node_id}] AUC_parent={auc_parent:.4f}, Weighted_AUC_children={weighted_auc_children:.4f}, Gain={gain:.4f}")

        if gain < delta:
            if verbose:
                print(f"→ Pruning node {node_id} (Gain={gain:.4f} < delta={delta})")
            new_leaf = LeafNode(y_node, node_id=node_id)
            if parent is None:
                pruned_root = new_leaf
            else:
                if is_left:
                    parent.left = new_leaf
                else:
                    parent.right = new_leaf

            if node.left:
                pruned_node_models.pop(node.left.node_id, None)
            if node.right:
                pruned_node_models.pop(node.right.node_id, None)
        else:
            queue.append((node.left, node, True))
            queue.append((node.right, node, False))

    return pruned_root, pruned_node_models


def evaluate_tree_on_fold(X, y, root_node, node_models, metric='brier', true_probs=None):
    """
    Evaluate the performance of the tree on a given fold.
    Params:
    - X: Feature matrix
    - y: True labels
    - root_node: Root node of the tree
    - node_models: Dictionary of models for each node
    - metric: Evaluation metric
    - true_probs: True probabilities for calibration (optional)

    Returns:
    - float: The evaluation score based on the specified metric.
    """
    sample_masks = get_sample_to_node_mask(root_node, X)
    y_pred = np.zeros_like(y, dtype=float)

    for node_id, mask in sample_masks.items():
        if not np.any(mask):
            continue
        model_info = node_models.get(node_id)
        if model_info is None:
            continue
        learners = model_info['learners']
        J = model_info['J']
        probs = logitboost.logitboost_predict_proba(X[mask], learners, J)
        y_pred[mask] = probs[:, 1]

    if true_probs is not None:
        # Use real probabilities to compute "true" calibration metrics
        if metric == 'brier':
            return np.mean((y_pred - true_probs) ** 2)
        elif metric == 'logloss':
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(true_probs * np.log(y_pred) + (1 - true_probs) * np.log(1 - y_pred))
        elif metric == 'auc':
            return roc_auc_score(true_probs, y_pred)
        else:
            raise ValueError(f"Unsupported metric '{metric}' with true_probs.")
    else:
        if metric == 'brier':
            return brier_score_loss(y, y_pred)
        elif metric == 'log-loss':
            return log_loss(y, y_pred)
        else:
            raise ValueError("Unsupported metric")

def cv_delta_pruning(X, y, root_node, node_models, deltas, K=5, metric_eval='brier',
                     metric_prune='auc', method='local', multiclass=False, average='macro', verbose=False, true_probs=None):
    """
    Cross-validate the pruning process to find the best delta.
    Params:
    - X: Feature matrix
    - y: True labels
    - root_node: Root node of the tree
    - node_models: Dictionary of models for each node
    - deltas: List of delta values to test
    - K: Number of cross-validation folds
    - metric_eval: Evaluation metric for validation
    - metric_prune: Evaluation metric for pruning
    - method: Pruning method to use
    - multiclass: Whether the problem is multiclass
    - average: Averaging method for multiclass metrics
    - verbose: Whether to print progress
    - true_probs: True probabilities for calibration (optional)

    Returns:
    - float: The mean delta value across all folds.
    """
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    delta_per_fold = []

    for k, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_rest, y_rest = X[train_idx], y[train_idx]
        X_fold, y_fold = X[val_idx], y[val_idx]
        true_probs_fold = true_probs[val_idx] if true_probs is not None else None

        best_score = np.inf
        best_delta = None

        for delta in deltas:
            if method == 'original':
                pruned_tree, pruned_models = regression_pruning_spline_bfs(
                    X_rest, y_rest, root_node, node_models, delta,
                    multiclass=multiclass, average=average, verbose=verbose
                    )
            elif method == 'local':
                pruned_tree, pruned_models = regression_pruning_gain_local(
                    X_rest, y_rest, root_node, node_models, delta,
                    multiclass=multiclass, average=average, verbose=verbose
                )
                
            score = evaluate_tree_on_fold(X_fold, y_fold, pruned_tree, pruned_models, metric=metric_eval, true_probs=true_probs_fold)

            if verbose:
                print(f"[Fold {k}] Delta={delta:.4f}, {metric_eval}={score:.4f}")

            if score < best_score:
                best_score = score
                best_delta = delta

        delta_per_fold.append(best_delta)

    return np.mean(delta_per_fold)


# Full pipeline function
def pipeline_spline_tree_lmt(
    X_train, y_train, 
    X_val, y_val,
    X_test, y_test,
    features=[0, 1],
    max_depth=6,
    min_samples_leaf=10,
    purity_threshold=0.95,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0,
    verbose=False,
    metric_prune='auc', # metric for pruning
    method = 'local', # or 'original'
    metric_eval='log-loss', # metric for evaluation (or 'brier')
    K=5,
    true_probs=None,
    deltas=None
):
    """
    Full pipeline to fit a custom Logistic Model Tree with LogitBoost, prune it and get predictions.
    Params:
    - X_train, y_train : training data
    - X_test, y_test : test data
    - X_val, y_val : validation data
    - features : list of feature indices to use
    - max_depth, min_samples_leaf, purity_threshold : tree parameters
    - lb_* : LogitBoost parameters
    - verbose : whether to print progress
    - metric_prune : metric for pruning process, default AUC
    - method : pruning method to use, default 'local', another option is 'original'
    - metric_eval : metric for evaluation, default log-loss, another option is 'brier'
    - K : number of cross-validation folds, default 5
    - true_probs : true probabilities for calibration (optional)
    - deltas : list of delta values to test (optional)

    Returns:
    - pruned_tree : the pruned tree
    - pruned_node_models : the pruned node models
    - y_pred : predictions on the test set
    - y_pred_probs : predicted probabilities on the test set
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

    # Step 2: find the best delta for pruning with the cv routine
    if deltas is None:
        if method == 'local':
            deltas = np.linspace(0, 0.0030, 30)
        elif method == 'original':
            deltas = np.linspace(0.60, 0.90, 30)

    delta_star = cv_delta_pruning(
        X_val, y_val, tree, node_models,
        deltas=deltas, K=K,
        metric_eval=metric_eval,
        metric_prune=metric_prune,  # the metric used during pruning
        method=method,  # or 'original'
        verbose=verbose,
        true_probs=true_probs
    )

    # Step 3: Prune the tree using the best delta
    if method == 'local':
        pruned_tree, pruned_node_models = regression_pruning_gain_local(
            X_train, y_train, tree, node_models,
            delta=delta_star, verbose=True
        )
    elif method == 'original':
        pruned_tree, pruned_node_models = regression_pruning_spline_bfs(
            X_train, y_train, tree, node_models,
            threshold=delta_star, verbose=True
        )


    # Step 4: Predict on test set
    y_pred = predict_lmt_custom(X_test, pruned_tree, pruned_node_models)
    y_pred_probs = predict_proba_lmt_custom(X_test, pruned_tree, pruned_node_models)

    return pruned_tree, pruned_node_models, y_pred, y_pred_probs