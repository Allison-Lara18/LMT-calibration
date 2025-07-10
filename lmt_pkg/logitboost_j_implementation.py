"""
LogitBoost algorithm implementation in Python.
Version for J classes.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

"""
Multiclass LogitBoost with support for J classes and SimpleLogistic routine, i.e., LogitBoost + 5 fold cross-validation.
* One-attribute weighted LinearRegression as the weak learner
* Fits and predicts with scikit-learn only for the Linear Regression piece

How to use:
learners, J = logitboost_fit(X_train, y_train,
                                     n_estimators=500,
                                     eps=1e-5,
                                     warm_start=None)

y_hat   = logitboost_predict(X_test,  learners, J)
proba   = logitboost_predict_proba(X_test, learners, J)

"""
# ------------------------------------------------------------------ #
# Auxiliary functions   
# ------------------------------------------------------------------ #

# def _softmax(F):
#     return np.exp(F) / np.sum(np.exp(F), axis=1, keepdims=True)
def _softmax(F, axis=1):
    """
    Numerically-stable soft-max.
    """
    F = F - np.max(F, axis=axis, keepdims=True)
    np.exp(F, out=F)
    F /= np.sum(F, axis=axis, keepdims=True)
    return F

def _best_feature_lr(X, z, w):
    """
    Function to return the single feature whose weighted simple (linear) regression gives the smallest 
    weighted sum of square error w·(z − f)^2
    
    Parameters:
    X : array (n_samples, n_features)
    z : array (n_samples,) -- working response
    w : array (n_samples,) -- weights for each sample

    Returns:
    feat_idx : int -- index of the best feature
    intercept : float -- intercept of the best linear regressor
    slope : float -- slope of the best linear regressor
    fitted_values : array (n_samples,) -- predicted values from the best linear regressor
    """
    # Data initialization
    best_err = np.inf
    best_idx = None
    best_b0  = best_b1 = 0.0
    best_fit = None

    # Iterate over each feature
    for k in range(X.shape[1]):
        # Fit a linear regression model to the k-th feature
        # using the working response z and weights w
        lr = LinearRegression()
        lr.fit(X[:, k].reshape(-1, 1), z, sample_weight=w)

        # Predict the fitted values
        f_hat = lr.predict(X[:, k].reshape(-1, 1))
        # Calculate the weighted sum of squared errors
        err   = np.sum(w * (z - f_hat) ** 2)

        # Update the best feature if the current error is lower
        # than the best error found so far
        if err < best_err:
            best_err = err
            best_idx = k
            best_b0  = lr.intercept_
            best_b1  = lr.coef_[0]
            best_fit = f_hat

    return best_idx, best_b0, best_b1, best_fit


# ------------------------------------------------------------------ #
# training                                                           #
# ------------------------------------------------------------------ #
def logitboost_fit(X, y, n_estimators=500, eps=1e-5, warm_start=None):
    """
    Function to fit multiclass LogitBoost.
    Parameters:
    X : array (n_samples, n_features)
    y : array (n_samples,) -- int labels in {0, 1, ..., J-1}
    n_estimators : int, number of boosting rounds (M)
    eps : float, lower/upper bound for p so that p∈[eps, 1-eps]
    warm_start : None or list of learners, if provided, will continue training from the last state

    Returns:
    learners : list of length n_estimators where each element is a list of J tuples (feat_idx, b0, b1)
                where feat_idx is the index of the feature used, b0 is the intercept, and b1 is the slope of the linear regressor
    J : int, number of classes 
    """
    # Data validation and conversion
    X = np.asarray(X, float)
    y = np.asarray(y,  int).ravel()
    n_samples, n_features = X.shape
    
    if warm_start is not None:
        # If warm_start is provided, we assume it contains the learners from a previous fit
        learners, J = warm_start
        # compute existing predictions
        F = _accumulate_F(X, learners, J)
        p = np.clip(_softmax(F), eps, 1-eps)

    else:
        J = int(y.max() + 1)
        # Step 1 : Start with weights w_ij = 1/N, F(x) = 0, p_ij = 1/J
        w = np.full(n_samples, 1.0 / n_samples)
        F = np.zeros((n_samples, J))
        p = np.full_like(F, 1.0 / J)
    
    learners = []

    # Step 2 : Iterative boosting
    for m in range(n_estimators):
        # Initialize lists for this round
        round_learners = []        # [(idx,b0,b1)  for j in 0..J-1]
        fits           = []        # list of arrays shape (n_samples,)

        # Step 2.a : Iterate over classes
        for j in range(J):
            # Step 2.a.i : Compute working response and weights for class j
            # y == j  (boolean mask for class j)
            w = np.clip(p[:, j] * (1.0 - p[:, j]), eps, None)  # weights w_ij with numerical safety
            z = ((y == j).astype(float) - p[:, j]) / w             # z_ij

            # Step 2.a.ii : Fit the function f_mj(x) by a weighted least-squares regression of z_ij to x_i with weights w_ij.
            idx, b0, b1, f_hat = _best_feature_lr(X, z, w)

            round_learners.append((idx, b0, b1))
            fits.append(f_hat)

        # Step 2.b : Update of f_mj and F_j(x)
        fits   = np.vstack(fits)  # Outputs of the weak learners, shape J × n_samples
        mean_f = fits.mean(axis=0, keepdims=True) # mean over all classes, shape 1 × n_samples
        adj_f  = (J - 1) / J * (fits - mean_f) # adjust the scores
        F += adj_f.T # F_j(x) = F_j(x) + f_mj(x)

        # Step 2.c : Update p_ij = σ(F_j(x))
        p  = _softmax(F)
        p  = np.clip(p, eps, 1.0 - eps) # numerical safety

        # Store the learners for this round
        learners.append(round_learners)

    return learners, J


# ------------------------------------------------------------------ #
# prediction                                                         #
# ------------------------------------------------------------------ #
def _accumulate_F(X, learners, J):
    """
    Function to compute additive scores  F_j(x)  for every sample.
    Parameters:
    X : array (n_samples, n_features)
    learners : list of length n_estimators
    J : int, number of classes

    Returns:
    F : array (n_samples, J) -- additive scores for each class
    """
    # Data validation and conversion
    X = np.asarray(X, float)
    F = np.zeros((X.shape[0], J))

    # Iterate over learners and accumulate scores
    for round_learners in learners:
        # Each round_learners is a list of tuples (idx, b0, b1) for each class
        fits = []
        # For each learner, compute the fitted values
        for (idx, b0, b1) in round_learners:
            fits.append(b0 + b1 * X[:, idx])
        # Stack the fitted values and compute the mean
        fits   = np.vstack(fits)
        mean_f = fits.mean(axis=0, keepdims=True)
        # Adjust the scores to ensure they sum to zero across classes
        adj_f  = (J - 1) / J * (fits - mean_f)
        # Updtate the additive scores F_j(x) = F_j(x) + f_mj(x)
        F     += adj_f.T
    return F


def logitboost_predict_proba(X, learners, J):
    return _softmax(_accumulate_F(X, learners, J))


def logitboost_predict(X, learners, J):
    # Step 3: Output the classifier argmax_j F_j(x)
    return _accumulate_F(X, learners, J).argmax(axis=1)


def extract_linear_models(learners, J, n_features):
    """
    Extract the final additive linear model coefficients (a_j, b_j) 
    for each class j from the LogitBoost learners.

    Parameters
    ----------
    learners : list of length M, where each element is a list of J tuples (idx, b0, b1)
               representing the weak learner for each class at that boosting round.
    J        : int, number of classes
    n_features : int, total number of features in the original X

    Returns
    -------
    intercepts : array, shape (J,)
        The summed intercept for each class.
    coefs      : array, shape (J, n_features)
        The summed slope coefficients for each class over all rounds.
    """
    intercepts = np.zeros(J)
    coefs = np.zeros((J, n_features))

    for round_learners in learners:
        # Gather raw b0, b1 across classes for this round
        b0s = np.array([b0 for (_, b0, _) in round_learners])
        b1s = np.array([b1 for (_, _, b1) in round_learners])
        # Compute means for sum-to-zero correction
        mean_b0 = b0s.mean()
        mean_b1 = b1s.mean()
        # Apply the LogitBoost adjustment and accumulate
        for j, (idx, b0, b1) in enumerate(round_learners):
            adj_b0 = (J - 1) / J * (b0 - mean_b0)
            adj_b1 = (J - 1) / J * (b1 - mean_b1)
            intercepts[j] += adj_b0
            coefs[j, idx] += adj_b1

    return intercepts, coefs



# ------------------------------------------------------------------ #
# SimpleLogistic routine (LogitBoost + 5-fold CV every node)         #
# ------------------------------------------------------------------ #
# Binary version
# def simple_logistic_fit(X, y,
#                         n_estimators=200,
#                         eps=1e-5,
#                         cv_splits=5,
#                         warm_start=None,
#                         random_state=0):
#     """
#     Implements SimpleLogistic:
#       1) 5-fold CV to find the best M* iterations (minimizing error)
#       2) Final fit on all data for M* rounds, optionally warm-start.

#     Returns
#     -------
#     final_learners : list of M* boosting rounds
#     J              : number of classes
#     M_star         : selected iteration count
#     cv_errs_mean   : array of length n_estimators of CV errors
#     """
#     X = np.asarray(X)
#     y = np.asarray(y).astype(int).ravel()

#     # get starting learners & J from warm_start or scratch
#     if warm_start is not None:
#         init_learners, J = warm_start
#     else:
#         # one-step fit to discover J
#         _, J = logitboost_fit(X, y, n_estimators=1, eps=eps)

#     # CV setup
#     skf = StratifiedKFold(n_splits=cv_splits,
#                           shuffle=True,
#                           random_state=random_state)
#     cv_errors = np.zeros((cv_splits, n_estimators))

#     # 1) loop folds
#     for fold_idx, (tr, val) in enumerate(skf.split(X, y)):
#         X_tr, y_tr = X[tr], y[tr]
#         X_val, y_val = X[val], y[val]

#         # full-fit on this training fold
#         learners_fold, _ = logitboost_fit(X_tr, y_tr,
#                                           n_estimators=n_estimators,
#                                           eps=eps,
#                                           warm_start=None)
#         # record error at each m
#         for m in range(1, len(learners_fold)+1):
#             preds = logitboost_predict(X_val, learners_fold[:m], J)
#             cv_errors[fold_idx, m-1] = np.mean(preds != y_val)

#     # average & select M*
#     cv_errs_mean = cv_errors.mean(axis=0)
#     M_star = int(np.argmin(cv_errs_mean)) + 1

#     # 2) final fit on all data for M_star
#     if warm_start is not None:
#         init_learners, _ = warm_start
#         if len(init_learners) >= M_star:
#             final_learners = init_learners[:M_star]
#         else:
#             to_add = M_star - len(init_learners)
#             new_ls, _ = logitboost_fit(X, y,
#                                        n_estimators=to_add,
#                                        eps=eps,
#                                        warm_start=warm_start)
#             final_learners = init_learners + new_ls
#     else:
#         final_learners, _ = logitboost_fit(X, y,
#                                            n_estimators=M_star,
#                                            eps=eps,
#                                            warm_start=None)

#     return final_learners, J, M_star, cv_errs_mean


# Multiclass version
import numpy as np
from sklearn.model_selection import StratifiedKFold

def simple_logistic_fit(
    X, y,
    n_estimators=200,
    eps=1e-5,
    cv_splits=5,
    warm_start=None,
    random_state=0
):
    """
    SimpleLogistic with proper multiclass support via local J.

    1) 5-fold CV to pick the best M* (min error), using the fold’s own J_fold.
    2) Final fit on all data for M* rounds, optionally warm-start with matching J_node.

    Returns
    -------
    final_learners : list of M* boosting rounds
    J_node         : number of classes in y
    M_star         : selected iteration count
    cv_errs_mean   : array of length n_estimators of CV errors
    """
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int).ravel()

    # --- 0) determine the local number of classes at this node
    classes = np.unique(y)
    J_node  = classes.size
    # (if your labels are not 0..J_node-1 you may need an inverse mapping here)

    # --- 1) get initial learners & J from warm_start or fresh
    if warm_start is not None:
        init_learners, init_J = warm_start
        # init_J should match J_node; else drop warm_start or remap
        if init_J != J_node:
            init_learners = None
            warm_start = None
    else:
        init_learners = None

    # prepare CV
    skf = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )
    cv_errors = np.zeros((cv_splits, n_estimators))

    # --- 2) cross-validation to find M_star
    for fold_idx, (tr, val) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_val, y_val = X[val], y[val]

        # fresh fit on this fold (no warm_start here)
        learners_fold, J_fold = logitboost_fit(
            X_tr, y_tr,
            n_estimators=n_estimators,
            eps=eps,
            warm_start=None
        )

        # record the misclassification error at each m, using the fold’s J_fold
        for m in range(1, len(learners_fold)+1):
            preds = logitboost_predict(
                X_val,
                learners_fold[:m],
                J_fold
            )
            cv_errors[fold_idx, m-1] = np.mean(preds != y_val)

    cv_errs_mean = cv_errors.mean(axis=0)
    M_star = int(np.argmin(cv_errs_mean)) + 1

    # --- 3) final fit on ALL data for M_star
    if warm_start is not None and len(init_learners) >= M_star:
        final_learners = init_learners[:M_star]
    else:
        # if warm_start wasn’t usable or too short, do a fresh fit for M_star
        final_learners, _ = logitboost_fit(
            X, y,
            n_estimators=M_star,
            eps=eps,
            warm_start=None
        )

    return final_learners, J_node, M_star, cv_errs_mean


# ----------------------------------------------- #
# Visualization functions                         #
# ----------------------------------------------- #
# Visualize the class predictions
def plot_decision_boundary(
        X, y, learners, J,
        title="Decision boundary",
        fill_value="mean",       # cómo rellenar las dimensiones extra: "mean" | "median" | float
        h=0.01
    ):
    """
    X : array (n_samples, D)         Datos completos.
    y : array (n_samples,)           Etiquetas de 0 … J-1.
    J : int                          Número de clases.
    learners : objeto(s) del modelo  Pasan a logitboost_predict.
    title : str                      Título de la figura.
    fill_value : str | float         Cómo rellenar dims 3…D para la rejilla.
    h : float                        Resolución de la rejilla.
    """

    # --- 1. Preparar rejilla en las primeras dos dimensiones -----------------
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # --- 2. Construir matriz completa para el predictor -----------------------
    n_grid = xx.size
    X_grid = np.zeros((n_grid, 2))
    X_grid[:, 0] = xx.ravel()
    X_grid[:, 1] = yy.ravel()

    if X.shape[1] > 2:
        X_grid = np.hstack((X_grid, (np.sum(X_grid**2, axis=1).reshape(-1, 1))))

    # --- 3. Predicción --------------------------------------------------------
    Z = logitboost_predict(X_grid, learners, J=J)   # (n_grid,) con etiquetas 0…J-1
    Z = Z.reshape(xx.shape)

    # --- 4. Dibujar -----------------------------------------------------------
    # 4.1 Puntos de entrenamiento
    plt.figure(figsize=(8, 6))
    J = len(np.unique(y))

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,
                s=30, edgecolors='k')

    # 4.2 Frontera de decisión
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def plot_probability_region_logit(
    X, y, learners, J,
    feature_pair=(0,1),
    prob_class=1,
    fill_value="mean",   # "mean" | "median" | any float
    h=0.01,
    cmap="RdYlBu",
    ax=None,
    title="Probability region"
):
    """
    Plots the 2D probability region of a LogitBoost model over features i vs j.

    Parameters
    ----------
    X : array (n_samples, D)
    y : array (n_samples,)                      # only used for setting axis limits
    learners : list of boosting rounds
    J : int                                     # number of classes
    feature_pair : tuple(int i, int j)
    prob_class : int                            # which column of predict_proba to show
    fill_value : "mean" | "median" | float      # how to fill the other dimensions
    h : float                                   # grid step size
    cmap : str or Colormap
    ax : matplotlib Axes, optional
    title : str
    """
    i, j = feature_pair
    D = X.shape[1]

    # 1) grid in feature-i vs feature-j space
    x0_min, x0_max = X[:, i].min() - 1, X[:, i].max() + 1
    x1_min, x1_max = X[:, j].min() - 1, X[:, j].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x0_min, x0_max, h),
        np.arange(x1_min, x1_max, h)
    )
    n_grid = xx.size

    # 2) build full-dim grid, filling other dims per `fill_value`
    if fill_value == "mean":
        default = X.mean(axis=0)
    elif fill_value == "median":
        default = np.median(X, axis=0)
    else:
        default = np.full(D, float(fill_value))

    X_grid = np.tile(default, (n_grid, 1))
    X_grid[:, i] = xx.ravel()
    X_grid[:, j] = yy.ravel()

    # 3) predict probabilities on the grid
    P = logitboost_predict_proba(X_grid, learners, J)[:, prob_class]
    P = np.clip(P, 0.0, 1.0)       # guard numerical overshoot
    Z = P.reshape(xx.shape)

    # 4) contourf with explicit levels 0→1
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    levels = np.linspace(0.0, 1.0, 51)
    contour = ax.contourf(
        xx, yy, Z,
        levels=levels,
        cmap=cmap,
        alpha=0.6
    )
    contour.set_clim(0, 1)         # enforce color limits
    cbar = plt.colorbar(contour, ax=ax, label=f"P(y={prob_class})")

    # 5) overlay training points, coloured by their own probability
    P_pts = logitboost_predict_proba(X, learners, J)[:, prob_class]
    P_pts = np.clip(P_pts, 0.0, 1.0)
    ax.scatter(
        X[:, i], X[:, j],
        c=P_pts,
        cmap=cmap,
        vmin=0, vmax=1,
        edgecolor='k',
        s=30
    )

    # 6) labels & title
    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_title(title)
    ax.set_xlim(x0_min, x0_max)
    ax.set_ylim(x1_min, x1_max)
    return ax, contour