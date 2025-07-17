from itertools import product
import numpy as np
from . import logitboost_j_implementation as logitboost

# ----------------------------------------- #
# Main class for composite tree classifier  #
# ----------------------------------------- #
# Versión actualizada del CompositeTreeClassifier para múltiples thresholds por feature (intervalos explícitos)

class CompositeTreeClassifier:
    def __init__(self, trees, region_conditions, feature_names=None):
        """
        Parameters:
        - trees: list of fitted DecisionTreeClassifier, one per region
        - region_conditions: list of conditions per region
          Each region is a list of (feature_index, lower_bound, upper_bound)
        """
        self.trees = trees
        self.region_conditions = region_conditions
        self.feature_names = feature_names
        
    def apply(self, X):
        """
        Devuelve el ID del nodo hoja para cada muestra en X.
        Usa el árbol base (self.tree) para aplicar el .apply() real.
        """
        return self.tree.apply(X)

    def _region_mask(self, X, region):
        mask = np.ones(len(X), dtype=bool)
        for f_idx, low, high in region:
            mask &= (X[:, f_idx] > low) & (X[:, f_idx] <= high)
        return mask

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=int)
        assigned = np.zeros(len(X), dtype=bool)

        for i, (tree, region) in enumerate(zip(self.trees, self.region_conditions)):
            if tree is None:
                continue

            mask = self._region_mask(X, region)
            if np.any(mask):
                predictions[mask] = tree.predict(X[mask])
                assigned[mask] = True

        predictions[~assigned] = 0
        return predictions

    def predict_proba(self, X):
        n_classes = self.trees[0].n_classes_
        probs = np.zeros((len(X), n_classes))
        assigned = np.zeros(len(X), dtype=bool)

        for i, (tree, region) in enumerate(zip(self.trees, self.region_conditions)):
            if tree is None:
                continue

            mask = self._region_mask(X, region)
            if np.any(mask):
                probs[mask] = tree.predict_proba(X[mask])
                assigned[mask] = True

        return probs


# ----------------------------------------- #
# Additional functions for tree building    #
# ----------------------------------------- #
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from itertools import product

from sklearn.metrics import mutual_info_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def find_best_cuts(X, y, max_cuts=2, criterion='entropy', n_bins=10):
    """
    Find the best cuts (thresholds) in different features that maximize the given criterion.
    Parameters:
    - X: np.array (n_samples, n_features)
    - y: np.array (n_samples,)
    - max_cuts: int, maximum number of features to cut
    - criterion: 'entropy' or 'gini'
    - n_bins: how many bins to try per feature (how many thresholds to discretize)

    Returns:
    - cut_dict: dictionary {feature_index: threshold}
    """
    n_features = X.shape[1]
    candidate_splits = []

    # Iterate over each feature to find the best cut
    for j in range(n_features):
        xj = X[:, j]
        thresholds = np.linspace(np.percentile(xj, 5), np.percentile(xj, 95), n_bins)
        best_gain = -np.inf
        best_t = None
        
        # Iterate over each threshold to find the best gain
        for t in thresholds:
            left = y[xj <= t]
            right = y[xj > t]
            if len(left) == 0 or len(right) == 0:
                continue

            p_left = len(left) / len(y)
            p_right = 1 - p_left

            if criterion == 'entropy':
                def entropy(v): return -np.sum([np.mean(v == c) * np.log2(np.mean(v == c)+1e-10) for c in np.unique(y)])
                gain = entropy(y) - (p_left * entropy(left) + p_right * entropy(right))
            elif criterion == 'gini':
                def gini(v): return 1 - np.sum([np.mean(v == c)**2 for c in np.unique(y)])
                gain = gini(y) - (p_left * gini(left) + p_right * gini(right))
            else:
                raise ValueError("criterion debe ser 'entropy' o 'gini'")
            
            if gain > best_gain:
                best_gain = gain
                best_t = t
        
        if best_t is not None:
            candidate_splits.append((best_gain, j, best_t))

    # Sort by gain and select the best max_cuts
    candidate_splits.sort(reverse=True, key=lambda x: x[0])
    selected = candidate_splits[:max_cuts]

    return {j: t for _, j, t in selected}

def find_best_cuts_multiple_per_feature(X, y, max_cuts=3, criterion='entropy', n_bins=10, top_k_per_feature=2):
    """
    Find the best cuts across all features, possibly multiple per feature.

    Parameters:
    - X: np.array (n_samples, n_features)
    - y: np.array (n_samples,)
    - max_cuts: int, maximum total number of cuts to return
    - criterion: 'entropy' or 'gini'
    - n_bins: number of candidate thresholds to try per feature
    - top_k_per_feature: number of best thresholds to keep per feature

    Returns:
    - cut_list: list of tuples (feature_index, threshold)
    """
    n_features = X.shape[1]
    candidate_splits = []

    for j in range(n_features):
        xj = X[:, j]
        thresholds = np.linspace(np.percentile(xj, 5), np.percentile(xj, 95), n_bins)
        gain_list = []

        for t in thresholds:
            left = y[xj <= t]
            right = y[xj > t]
            if len(left) == 0 or len(right) == 0:
                continue

            p_left = len(left) / len(y)
            p_right = 1 - p_left

            if criterion == 'entropy':
                def entropy(v): return -np.sum([np.mean(v == c) * np.log2(np.mean(v == c)+1e-10) for c in np.unique(y)])
                gain = entropy(y) - (p_left * entropy(left) + p_right * entropy(right))
            elif criterion == 'gini':
                def gini(v): return 1 - np.sum([np.mean(v == c)**2 for c in np.unique(y)])
                gain = gini(y) - (p_left * gini(left) + p_right * gini(right))
            else:
                raise ValueError("criterion must be 'entropy' or 'gini'")
            
            gain_list.append((gain, j, t))

        # Keep the top k thresholds per feature
        gain_list.sort(reverse=True, key=lambda x: x[0])
        candidate_splits.extend(gain_list[:top_k_per_feature])

    # Select the global top max_cuts thresholds
    candidate_splits.sort(reverse=True, key=lambda x: x[0])
    selected = candidate_splits[:max_cuts]

    # Group by feature index
    cut_dict = {}
    for _, j, t in selected:
        if j not in cut_dict:
            cut_dict[j] = []
        cut_dict[j].append(t)

    return cut_dict


def create_root_with_joint_cuts(X, y, cut_dict, size='regular', criterion='entropy', random_state=0):
    """
    Create a root node with multiple cuts (one for each feature) that generate disjoint regions,
    and train an independent decision tree in each region.

    Parameters:
    - X: np.array, shape (n_samples, n_features)
    - y: np.array, labels
    - cut_dict: dict, {feature_idx: threshold}
    - size: 'shallow', 'regular', 'overfit'
    - criterion: 'entropy' or 'gini'
    - random_state: int, seed for reproducibility

    Returns:
    - List of trained trees (one for each child region)
    - List of masks (conditions for each region)
    """
    presets = {
        'shallow':  {'max_depth': 1,  'min_samples_leaf': 20, 'min_samples_split': 40},  # total depth 2
        'regular':  {'max_depth': 3,  'min_samples_leaf': 5,  'min_samples_split': 15},
        'overfit':  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
    }

    params = presets.get(size)
    if params is None:
        raise ValueError("Invalid size parameter. Choose from 'shallow', 'regular', or 'overfit'.")

    # If max_depth is used, subtract 1 for the root node
    if params.get('max_depth') is not None:
        params['max_depth'] = max(1, params['max_depth'] - 1)

    n = len(X)
    m = len(cut_dict)
    conditions = list(product([True, False], repeat=m))  # all combinations of cuts (2^m)
    feature_indices = list(cut_dict.keys())
    thresholds = list(cut_dict.values())

    trees = []
    region_masks = []

    for cond in conditions:
        mask = np.ones(n, dtype=bool)
        for i in range(m):
            f_idx = feature_indices[i]
            t = thresholds[i]
            if cond[i]:
                mask &= X[:, f_idx] <= t
            else:
                mask &= X[:, f_idx] > t

        X_region = X[mask]
        y_region = y[mask]

        if len(y_region) == 0:
            trees.append(None)
            region_masks.append(mask)
            continue

        clf = DecisionTreeClassifier(criterion=criterion, random_state=random_state, **params)
        clf.fit(X_region, y_region)

        trees.append(clf)
        region_masks.append(mask)
    
    return trees, region_masks, conditions, feature_indices, thresholds


from itertools import product

def create_root_with_multiple_thresholds(X, y, cut_dict, size='regular', criterion='entropy', random_state=0):
    """
    Extends create_root_with_joint_cuts to support multiple thresholds per feature.
    
    Parameters:
    - cut_dict: {feature_index: [threshold1, threshold2, ...]}
    
    Returns:
    - trees: list of trained DecisionTreeClassifier per region
    - region_masks: list of boolean masks for each region
    - region_conditions: list of condition tuples like [(feature_idx, operator, threshold), ...]
    - feature_indices: list of sorted feature indices
    - all_thresholds: ordered list of thresholds aligned with feature_indices
    """
    from sklearn.tree import DecisionTreeClassifier

    # Get sorted list of features
    feature_indices = sorted(cut_dict.keys())
    thresholds_by_feature = [sorted(cut_dict[f]) for f in feature_indices]

    # Create all possible threshold bins for each feature
    # For example, if thresholds = [0.3, 0.7], this creates 3 regions:
    # (-inf, 0.3], (0.3, 0.7], (0.7, inf)
    def get_conditions(feature_idx, thresholds):
        edges = [-np.inf] + thresholds + [np.inf]
        return [(feature_idx, edges[i], edges[i+1]) for i in range(len(edges)-1)]

    # Build all region definitions as combinations of intervals
    all_region_defs = list(product(*[
        get_conditions(f_idx, cut_dict[f_idx])
        for f_idx in feature_indices
    ]))

    presets = {
        'shallow':  {'max_depth': 1,  'min_samples_leaf': 20, 'min_samples_split': 40},
        'regular':  {'max_depth': 3,  'min_samples_leaf': 5,  'min_samples_split': 15},
        'overfit':  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
    }

    params = presets.get(size)
    if params is None:
        raise ValueError("Invalid size parameter.")

    trees = []
    region_masks = []
    region_conditions = []

    for region in all_region_defs:
        mask = np.ones(len(X), dtype=bool)
        condition_list = []
        for (f_idx, low, high) in region:
            mask &= (X[:, f_idx] > low) & (X[:, f_idx] <= high)
            condition_list.append((f_idx, low, high))

        X_region = X[mask]
        y_region = y[mask]

        if len(y_region) == 0:
            trees.append(None)
            region_masks.append(mask)
            region_conditions.append(condition_list)
            continue

        clf = DecisionTreeClassifier(criterion=criterion, random_state=random_state, **params)
        clf.fit(X_region, y_region)

        trees.append(clf)
        region_masks.append(mask)
        region_conditions.append(condition_list)

    return trees, region_masks, region_conditions, feature_indices, thresholds_by_feature


# ------------------------------------------- #
# Visualization functions for composite trees #
# ------------------------------------------- #

def visualize_root_and_subtrees_grid(trees, conditions, feature_indices, thresholds, feature_names=None):
    """
    Draw the root node with labels and below a grid with the child trees visualized with plot_tree.
    """
    num_trees = len(trees)
    cols = min(2, num_trees)
    rows = (num_trees + cols - 1) // cols

    fig = plt.figure(figsize=(10 * cols, 3 + 6 * rows))
    ax_root = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)

    # Plot root node
    ax_root.axis('off')
    ax_root.text(0.5, 0.8, "Root Node", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black"), fontsize=14)

    # Horizontal positions for arrows
    x_positions = np.linspace(0.1, 0.9, num_trees)

    for i, (tree, cond) in enumerate(zip(trees, conditions)):
        if tree is None:
            continue

        cond_label = " ∧ ".join([
            f"{feature_names[f]} ≤ {thresholds[j]:.2f}" if c else f"{feature_names[f]} > {thresholds[j]:.2f}"
            for j, (f, c) in enumerate(zip(feature_indices, cond))
        ])

        # Annotate arrow from root to condition label
        ax_root.annotate("",
                         xy=(x_positions[i], 0.55), xycoords='axes fraction',
                         xytext=(0.5, 0.75), textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="->", lw=1.5))
        ax_root.text(x_positions[i], 0.45, cond_label,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black"))

        # Draw the tree in the corresponding subplot
        r, c = divmod(i, cols)
        ax_sub = plt.subplot2grid((rows + 1, cols), (r + 1, c))
        ax_sub.set_title(f"Tree #{i}", fontsize=12)
        plot_tree(tree, ax=ax_sub, filled=True, feature_names=feature_names, class_names=True)

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np

def visualize_root_and_subtrees_grid_with_intervals(trees, region_conditions, feature_names=None, title="Composite Tree Root Node with Subtrees"):
    """
    Draw the root node with region intervals and a grid of subtrees.
    
    Parameters
    ----------
    trees : list
        Trained decision trees, one per region.
    region_conditions : list
        Each region is defined by a list of tuples: (feature_index, lower_bound, upper_bound).
    feature_names : list or None
        Optional list of feature names to use for labels.
    """
    num_trees = len(trees)
    cols = min(2, num_trees)
    rows = (num_trees + cols - 1) // cols

    fig = plt.figure(figsize=(10 * cols, 3 + 6 * rows))
    ax_root = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)

    ax_root.axis('off')
    ax_root.text(0.5, 0.8, "Root Node", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black"), fontsize=14)

    x_positions = np.linspace(0.1, 0.9, num_trees)

    for i, (tree, conds) in enumerate(zip(trees, region_conditions)):
        if tree is None:
            continue

        cond_label = " ∧ ".join([
            f"{feature_names[f] if feature_names else f'X{f}'} ∈ ({low:.2f}, {high:.2f}]" 
            for (f, low, high) in conds
        ])

        ax_root.annotate("",
                         xy=(x_positions[i], 0.55), xycoords='axes fraction',
                         xytext=(0.5, 0.75), textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="->", lw=1.5))
        ax_root.text(x_positions[i], 0.45, cond_label,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black"))

        r, c = divmod(i, cols)
        ax_sub = plt.subplot2grid((rows + 1, cols), (r + 1, c))
        ax_sub.set_title(f"Tree #{i}", fontsize=12)
        plot_tree(tree, ax=ax_sub, filled=True, feature_names=feature_names, class_names=True)
        # Add the general title for the whole figure
        fig.suptitle(title, fontsize=16, y=0.95)

    plt.tight_layout()
    plt.show()



# ------------------------------------------- #
# Evaluation functions for composite trees    #
# ------------------------------------------- #

from sklearn.metrics import accuracy_score

def evaluate_forest(trees, X, y, cut_dict):
    """
    Evaluate the ensemble of trees (one per region) on a given set X, y.
    Returns the overall accuracy and the predictions.
    Parameters:
    - trees: list of DecisionTreeClassifier objects, one for each region
    - X: np.array, shape (n_samples, n_features)
    - y: np.array, labels
    - cut_dict: dict, mapping from feature indices to threshold values
    Returns:
    - acc: float, accuracy of the ensemble on the data
    - predictions: np.array, predicted labels for each sample in X
    - assigned: np.array, boolean mask indicating which samples were assigned a prediction
    """
    m = len(cut_dict)
    feature_indices = list(cut_dict.keys())
    thresholds = list(cut_dict.values())

    predictions = np.zeros(len(X), dtype=int)
    assigned = np.zeros(len(X), dtype=bool)

    conditions = list(product([True, False], repeat=m))

    for i, cond in enumerate(conditions):
        tree = trees[i]
        if tree is None:
            continue

        mask = np.ones(len(X), dtype=bool)
        for j in range(m):
            f_idx = feature_indices[j]
            t = thresholds[j]
            if cond[j]:
                mask &= X[:, f_idx] <= t
            else:
                mask &= X[:, f_idx] > t

        if np.any(mask):
            predictions[mask] = tree.predict(X[mask])
            assigned[mask] = True

    # Evaluate accuracy if there are assigned predictions
    acc = accuracy_score(y[assigned], predictions[assigned])
    return acc, predictions, assigned


from sklearn.metrics import accuracy_score

def evaluate_forest_with_intervals(trees, X, y, region_conditions):
    """
    Evaluate a composite forest using explicit region intervals.

    Parameters
    ----------
    trees : list
        List of DecisionTreeClassifier objects (or None) per region.
    X : np.array
        Input data of shape (n_samples, n_features).
    y : np.array
        True labels of shape (n_samples,).
    region_conditions : list
        List of region-specific conditions:
        Each item is a list of (feature_index, lower_bound, upper_bound) tuples.

    Returns
    -------
    acc : float
        Accuracy on the assigned samples.
    predictions : np.array
        Predicted class labels for each sample.
    assigned : np.array
        Boolean mask indicating which samples received a prediction.
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=int)
    assigned = np.zeros(n_samples, dtype=bool)

    for i, (tree, conds) in enumerate(zip(trees, region_conditions)):
        if tree is None:
            continue

        mask = np.ones(n_samples, dtype=bool)
        for f_idx, low, high in conds:
            mask &= (X[:, f_idx] > low) & (X[:, f_idx] <= high)

        if np.any(mask):
            predictions[mask] = tree.predict(X[mask])
            assigned[mask] = True

    acc = accuracy_score(y[assigned], predictions[assigned])
    return acc, predictions, assigned



# ------------------------------------------- #
# LMT functions for composite trees           #
# ------------------------------------------- #
from scipy import sparse

def fit_logistic_composite_tree_v2(
    X, y,
    cut_dict,
    size='regular',
    pruning=True,
    tree_random_state=0,
    lb_n_estimators=200,
    lb_eps=1e-5,
    lb_cv_splits=5,
    lb_random_state=0
):
    """
    Composite Tree + LogitBoost V2:
    - Use create_root_with_multiple_thresholds to create multiple subtrees by region.
    - Fit SimpleLogistic (LogitBoost with CV) at the root of each region to get M_star.
    - Fit LogitBoost with M_star rounds in all child nodes, warm-starting from parent.
    
    Returns
    -------
    trees          : list of fitted DecisionTreeClassifier, one per region.
    node_models    : dict with key = (region_idx, node_id) and value = {
                        'learners': list of boosting rounds,
                        'J': number of classes,
                        'M_star': int,
                        'cv_errors': np.array or None
                     }
    """
    # Step 1: Build composite tree
    trees, masks, region_conditions, feats, thresholds = create_root_with_multiple_thresholds(
        X, y, cut_dict=cut_dict, size=size
    )

    for i, tree in enumerate(trees):
        print(f"Tree {i} is None? {tree is None}")

    node_models = {}

    # Step 2: Loop over each region (subtree)
    for region_idx, (tree, mask) in enumerate(zip(trees, masks)):
        if tree is None:
            continue

        X_region, y_region = X[mask], y[mask]

        # Step 2.1: Get decision_path matrix for this region’s tree
        node_indicator = tree.decision_path(X_region)

        # Step 2.2: Fit root node (node 0) using SimpleLogistic (with CV)
        root_learners, J_root, M_star, cv_errs = logitboost.simple_logistic_fit(
            X_region, y_region,
            n_estimators=lb_n_estimators,
            eps=lb_eps,
            cv_splits=lb_cv_splits,
            warm_start=None,
            random_state=lb_random_state
        )

        # Step 2.3: Recursively go through the nodes
        def recurse(node_id, warm_start):
            # Get samples passing through this node
            if sparse.issparse(node_indicator):
                sample_mask = node_indicator[:, node_id].toarray().ravel().astype(bool)
            else:
                sample_mask = node_indicator[:, node_id].astype(bool)

            X_node = X_region[sample_mask]
            y_node = y_region[sample_mask]

            if node_id == 0:
                # Use root’s SimpleLogistic result
                learners_node, J_node, cv_err_node = root_learners, J_root, cv_errs
            else:
                # Fit LogitBoost for M_star with warm_start
                learners_node, J_node = logitboost.logitboost_fit(
                    X_node, y_node,
                    n_estimators=M_star,
                    eps=lb_eps,
                    warm_start=warm_start
                )
                cv_err_node = None

            # Store in node_models
            node_models[(region_idx, node_id)] = {
                'learners': learners_node,
                'J': J_node,
                'M_star': M_star,
                'cv_errors': cv_err_node
            }

            # Recurse to children
            left_id = tree.tree_.children_left[node_id]
            right_id = tree.tree_.children_right[node_id]

            if left_id != -1:
                recurse(left_id, warm_start=(learners_node, J_node))
            if right_id != -1:
                recurse(right_id, warm_start=(learners_node, J_node))

        # Start recursion at root node (node 0)
        recurse(0, warm_start=None)

    return trees, node_models, region_conditions, feats, thresholds


def predict_composite_lmt(X, composite_clf, node_models):
    """
    Predict class labels for a Composite Logistic Model Tree.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    composite_clf : CompositeTreeClassifier
        Fitted composite tree.
    node_models : dict
        Dictionary mapping (region_idx, node_id) to fitted LogitBoost models.
        
    Returns
    -------
    preds : np.ndarray of shape (n_samples,)
        Predicted class labels.
    """
    preds = np.empty(len(X), dtype=int)
    assigned = np.zeros(len(X), dtype=bool)

    for region_idx, tree in enumerate(composite_clf.trees):
        if tree is None:
            continue

        # Find samples assigned to this region
        region_mask = composite_clf._region_mask(X, composite_clf.region_conditions[region_idx])
        if not np.any(region_mask):
            continue

        # For these samples, get their leaf node IDs
        X_region = X[region_mask]
        leaf_ids = tree.apply(X_region)

        region_preds = np.empty(len(X_region), dtype=int)
        for i, (x_i, node_id) in enumerate(zip(X_region, leaf_ids)):
            learners = node_models[(region_idx, node_id)]['learners']
            J        = node_models[(region_idx, node_id)]['J']
            p = logitboost.logitboost_predict(x_i.reshape(1, -1), learners, J)
            region_preds[i] = p[0]

        preds[region_mask] = region_preds
        assigned[region_mask] = True

    preds[~assigned] = 0  # default prediction for unassigned
    return preds


def predict_proba_composite_lmt(X, composite_clf, node_models):
    """
    Predict class probabilities for a Composite Logistic Model Tree.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    composite_clf : CompositeTreeClassifier
        Fitted composite tree.
    node_models : dict
        Dictionary mapping (region_idx, node_id) to fitted LogitBoost models.
        
    Returns
    -------
    proba_matrix : np.ndarray of shape (n_samples, max_J)
        Predicted probabilities for each sample.
    """
    # Find max number of classes
    max_J = max(model['J'] for model in node_models.values())
    probas = np.zeros((len(X), max_J), dtype=float)
    assigned = np.zeros(len(X), dtype=bool)

    for region_idx, tree in enumerate(composite_clf.trees):
        if tree is None:
            continue

        region_mask = composite_clf._region_mask(X, composite_clf.region_conditions[region_idx])
        if not np.any(region_mask):
            continue

        X_region = X[region_mask]
        leaf_ids = tree.apply(X_region)

        region_probas = np.zeros((len(X_region), max_J), dtype=float)
        for i, (x_i, node_id) in enumerate(zip(X_region, leaf_ids)):
            model = node_models[(region_idx, node_id)]
            p = logitboost.logitboost_predict_proba(x_i.reshape(1, -1), model['learners'], model['J'])
            region_probas[i, :model['J']] = p[0]

        probas[region_mask] = region_probas
        assigned[region_mask] = True

    return probas


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import plot_tree
import numpy as np
import re

def plot_tree_with_linear_models_grid_improved(
    trees,
    region_conditions,
    node_models_list,
    X,
    feature_names=None,
    title="Composite Tree with LogitBoost Models",
    model_threshold=1e-6,
    show_internal=False  # <--- NEW toggle
):
    num_trees = len(trees)
    cols = min(2, num_trees)
    rows = (num_trees + cols - 1) // cols

    fig = plt.figure(figsize=(10 * cols, 3 + 6 * rows))
    ax_root = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)
    ax_root.axis('off')

    # Root label
    ax_root.text(0.5, 0.8, "Root Node", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black"), fontsize=14)

    x_positions = np.linspace(0.1, 0.9, num_trees)

    for i, (tree, conds) in enumerate(zip(trees, region_conditions)):
        if tree is None:
            continue

        cond_label = " ∧ ".join([
            f"{feature_names[f] if feature_names else f'X{f}'} ∈ ({low:.2f}, {high:.2f}]" 
            for (f, low, high) in conds
        ])

        ax_root.annotate("",
                         xy=(x_positions[i], 0.55), xycoords='axes fraction',
                         xytext=(0.5, 0.75), textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="->", lw=1.5))
        ax_root.text(x_positions[i], 0.45, cond_label,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black"))

        r, c = divmod(i, cols)
        ax_sub = plt.subplot2grid((rows + 1, cols), (r + 1, c))

        node_models = node_models_list[i]
        n_features = X.shape[1]
        feature_labels = feature_names if feature_names else [f"x[{k}]" for k in range(n_features)]

        intercepts_dict = {}
        coefs_dict = {}
        for nid, mdl in node_models.items():
            ints, cos = logitboost.extract_linear_models(mdl['learners'], mdl['J'], n_features)
            intercepts_dict[nid] = ints
            coefs_dict[nid] = cos

        plot_tree(
            tree,
            feature_names=feature_labels,
            filled=True,
            node_ids=True,
            ax=ax_sub,
            fontsize=8
        )

        # Find leaf ids
        tree_ = tree.tree_
        leaf_ids = [j for j in range(tree_.node_count) if tree_.children_left[j] == -1]

        for txt in ax_sub.texts:
            full = txt.get_text()
            first_line = full.split("\n", 1)[0]
            m = re.search(r'\b(\d+)\b', first_line)
            if not m:
                continue
            node_id = int(m.group(1))
            if not show_internal and node_id not in leaf_ids:
                continue

            ints = intercepts_dict.get(node_id)
            cos = coefs_dict.get(node_id)
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

        ax_sub.set_title(f"Tree #{i}", fontsize=12)

    fig.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()


from typing import List, Dict, Tuple
import numpy as np


def split_node_models_by_tree(node_models, n_trees):
    """
    Transforma un dict {(region_id, node_id): model, ...} a una lista de dicts por árbol.
    """
    models_list = [{} for _ in range(n_trees)]
    for (region_id, node_id), model in node_models.items():
        models_list[region_id][node_id] = model
    return models_list