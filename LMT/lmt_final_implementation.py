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
import logitboost_j_implementation as logitboost
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from  sklearn.model_selection import cross_val_score
from sklearn import tree


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

def predict_proba_lmt(X, clf_tree, node_models):
    """
    Return P(y=1) for each row of X by routing it to its leaf, then 
    calling logitboost_predict_proba on that leaf's model.
    """
    leaf_ids = clf_tree.apply(X)
    probs = []
    for x_i, leaf in zip(X, leaf_ids):
        learners = node_models[leaf]['learners']
        J        = node_models[leaf]['J']
        # returns array[[p(0), p(1)]]
        p01 = logitboost.logitboost_predict_proba(x_i.reshape(1, -1), learners, J)
        probs.append(p01[0, 1])
    return np.array(probs)

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
):
    """
    Draws a DecisionTree with plot_tree and then appends J linear models
    F_j(x) = intercepts[j] + sum_k coefs[j,k] * x[k] in each leaf and, optionally,
    in internal nodes as well.

    Parameters
    ----------
    clf_tree : DecisionTreeClassifier
    node_models : dict[node_id] -> {'learners': [...], 'J': int, ...}
    X : array-like, shape (n_samples, n_features)
    title : title of the graph
    class_names : list of str, optional
        Names of classes; if None, uses '0', '1', ...
    show_internal : bool, default False
        If True, show linear models in all nodes; otherwise only in leaves.
    model_threshold: float
        Gives the threshold of printing the coefficients of the model, if coef>model_threshold it is going to be printed.
    """
    n_features = X.shape[1]
    feature_labels = [f"x[{k}]" for k in range(n_features)]

    # precompute intercepts & coefs for each node
    intercepts_dict = {}
    coefs_dict      = {}
    for nid, mdl in node_models.items():
        ints, cos = logitboost.extract_linear_models(mdl['learners'], mdl['J'], n_features)
        intercepts_dict[nid] = ints
        coefs_dict[nid]      = cos

    # draw base tree with node IDs
    fig, ax = plt.subplots(figsize=(16, 8))
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
    tree_    = clf_tree.tree_
    leaf_ids = [i for i in range(tree_.node_count)
                if tree_.children_left[i] == -1]

    # append formulas in each text box per node
    for txt in ax.texts:
        full = txt.get_text()
        first_line = full.split("\n", 1)[0]
        m = re.search(r'\b(\d+)\b', first_line)
        if not m:
            continue
        node_id = int(m.group(1))
        # skip internals if not requested
        if not show_internal and node_id not in leaf_ids:
            continue

        ints = intercepts_dict[node_id]
        cos  = coefs_dict[node_id]
        J    = len(ints)
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

    plt.title(title)
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
