# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

"""
Logistic Model Trees (LMT) implementation in Python.
This code implements a logistic model tree (LMT) algorithm, which is a decision tree that uses logistic regression as the model at each leaf node. Following the
paper structure from 2005 by Landwehr.

How to use:
tree, meta = lmt_fit(train_df, target='target')
proba = lmt_predict_proba(test_df, tree, meta)
pred  = proba.argmax(axis=1)

Notes:
train_df should be a pandas DataFrame with the target variable as one of the columns.
test_df should be a pandas DataFrame with the same features as train_df, excluding the target variable.
"""

# ---------------------------------
# Section 1 : utilities
# ---------------------------------

# def _softmax(F):
#     """
#     Function to compute the softmax of an array F.
#     Parameters:
#     F : array-like, input array for which the softmax is to be computed

#     Returns:
#     Softmax of the input array F.
#     """
#     return np.exp(F) / np.sum(np.exp(F), axis=1, keepdims=True)

def _softmax(F, axis=1):
    """
    Numerically-stable soft-max.
    """
    F = F - np.max(F, axis=axis, keepdims=True)
    np.exp(F, out=F)
    F /= np.sum(F, axis=axis, keepdims=True)
    return F


def _entropy(y):
    """
    Function to compute the multiclass entropy in bits. E(I) = -∑p(I)*log2p(I)
    Parameters:
    y : array-like, input array of class labels

    Returns:
    Entropy of the class labels y in bits.
    """
    _, cnts = np.unique(y, return_counts=True)
    p = cnts / cnts.sum()
    return -np.sum(p * np.log2(p + 1e-12))

# ----------------------------------------------------------------------
# Section 2 : LogitBoost with early stopping and 5-fold cross-validation
# ----------------------------------------------------------------------
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

# -----------
# TRAINING 
# -----------
def _logitboost_fit(X, y, n_estimators=200, eps=1e-5, init_F=None):
    """
    Function to fit multiclass LogitBoost with a parameter to obtain weights and probabilities as an input.
    Parameters:
    X : array (n_samples, n_features)
    y : array (n_samples,) -- int labels in {0, 1, ..., J-1}
    n_estimators : int, number of boosting rounds (M)
    eps : float, lower/upper bound for p so that p∈[eps, 1-eps]
    init_F : array (n_samples, J) -- initial scores / probabilities, if None then F is initialized to zeros, is F_{ij}

    Returns:
    learners : list of length n_estimators where each element is a list of J tuples (feat_idx, b0, b1)
                where feat_idx is the index of the feature used, b0 is the intercept, and b1 is the slope of the linear regressor
    J : int, number of classes
    F : array (n_samples, J) -- final scores / probabilities after fitting
    """
    # Data initialization and conversion
    X = np.asarray(X, float)
    y = np.asarray(y,  int).ravel()
    n_samples, n_features = X.shape
    J = int(y.max()) + 1

    # — initial scores / probas —
    if init_F is None:
        # Step 1 : Start with weights w_ij = 1/N, F(x) = 0, p_ij = 1/J
        w = np.full(n_samples, 1.0 / n_samples)
        F = np.zeros((n_samples, J))
        p = np.full_like(F, 1.0/J)
    else:
        # If you have initial scores, use them as a warm-start
        F = init_F.copy()
        p = _softmax(F)

    learners = []

    # Step 2 : Iterative boosting
    for m in range(n_estimators):
        # Initialize lists for this round
        round_learners, fits = [], []

        # Step 2.a : Iterate over classes
        for j in range(J):
            # Step 2.a.i ; Compute working response and weights for class j
            # y == j is the binary response for class j (boolean mask for class j)
            w = np.clip(p[:, j] * (1.0 - p[:, j]), eps, None) # weights w_ij with numerical safety
            z = ((y == j).astype(float) - p[:, j]) / w  # z_ij

            # Step 2.a.ii : Fit the function f_mj(x) by a weighted least-squares regression of z_ij to x_i with weights w_ij.
            idx, b0, b1, f_hat = _best_feature_lr(X, z, w)

            round_learners.append((idx, b0, b1))
            fits.append(f_hat)

        # Step 2.b : Update of f_mj and F_j(x)
        fits = np.vstack(fits)  # Outputs of the weak learners, shape J × n_samples
        mean_f = fits.mean(axis=0, keepdims=True)  # mean over all classes, shape 1 × n_samples
        adj_f  = (J - 1) / J * (fits - mean_f) # adjust the scores
        F += adj_f.T # F_j(x) = F_j(x) + f_mj(x)

        # Step 2.c : Update of p_ij = softmax(F_j(x))
        p  = _softmax(F)
        p  = np.clip(p, eps, 1.0 - eps) # numerical safety

        # Store the learners for this round
        learners.append(round_learners)

    return learners, J, F # F is returned for warm-starting

# ------------------------------------------------------------
def _choose_cv_folds(y, k_desired=5):
    """
    Function to choose the number of stratified folds for cross-validation.
    Parameters:
    y : array-like, input array of class labels
    k_desired : int, desired number of folds for cross-validation

    Returns:
    k  : int, number of folds for cross-validation
    or  None  if CV should be skipped.
    
    Note : This function ensures that there are at least 2 samples per class in each fold.
    If there are not enough samples to stratify, it returns None to skip cross-validation.  
    """
    y = np.asarray(y).ravel()
    m = np.bincount(y, minlength=y.max()+1).min()
    if m < 2:                       # cannot stratify
        return None                 # -> skip CV
    return max(2, min(k_desired, m))

def logitboost_fit_cv(X, y, max_estimators=200, k=5, eps=1e-5, patience=5):
    """
    Function to fit multiclass LogitBoost with early stopping and 5-fold cross-validation in the *simple-logistic* style fit.
    Parameters:
    X : array (n_samples, n_features)
    y : array (n_samples,) -- int labels in {0, 1, ..., J-1}
    max_estimators : int, maximum number of boosting rounds (M)
    k : int, number of folds for cross-validation
    eps : float, lower/upper bound for p so that p∈[eps, 1-eps]
    patience : int, number of rounds without improvement before early stopping

    Returns:
    learners : list of length n_estimators where each element is a list of J tuples (feat_idx, b0, b1)
                where feat_idx is the index of the feature used, b0 is the intercept, and b1 is the slope of the linear regressor
    J : int, number of classes
    """
    # Data initialization and conversion
    X = np.asarray(X, float)
    y = np.asarray(y,  int).ravel()
    J = int(y.max()) + 1

    k = _choose_cv_folds(y)
    if k is None:                            # not enough data to CV
        learners, J_out, _ = _logitboost_fit(
                                X, y,
                                n_estimators=min(10, max_estimators),
                                eps=eps)
        return learners, J_out               # early exit


    # Initialize the cross-validation and variables for early stopping
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=13)
    # Initialize variables for early stopping
    # best_m is the number of estimators, best_score is the best score found so far, no_imp is the number of rounds without improvement
    best_m, best_score, no_imp = 1, np.inf, 0
    scores_path = [] # for post-hoc inspection

    # Cross-validation loop for early stopping
    # Iterate over the number of estimators
    for m in range(1, max_estimators+1):
        fold_scores = []
        # For each fold, fit the model and compute the score
        # Note: we use the simple-logistic style fit, i.e., we do not use the warm-starting
        for train_idx, val_idx in cv.split(X, y):
            # Fit the model on the training set
            learners, _, _ = _logitboost_fit(X[train_idx], y[train_idx], n_estimators=m, eps=eps)
            # Predict on the validation set
            val_pred = logitboost_predict(X[val_idx], learners, J)
            # Store the mean classification error score
            fold_scores.append((y[val_idx] != val_pred).mean())

        # Compute the average score for this fold
        avg = np.mean(fold_scores)
        # Store the score for this round
        scores_path.append(avg)

        # Check for improvement
        if avg + 1e-8 < best_score: # improvement
            # Update the best model and score
            best_m, best_score, no_imp = m, avg, 0
        else:
            # No improvement, increment the no_imp counter
            no_imp += 1
            # Check for early stopping
            if no_imp >= patience:  # early-stop
                break

    # With the best number of estimators, fit the final model
    learners, _, _ = _logitboost_fit(X, y, n_estimators=best_m, eps=eps)
    return learners, J


# -------------
# PREDICTION 
# -------------
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

    # Iterate over learners and accumulate the scores
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

# ----------------------------------------------------------------------
# Section 3 : Data preprocessing
# ----------------------------------------------------------------------
def preprocess_dataframe(df, numeric_strategy='mean', cat_strategy='most_frequent'):
    """
    Function to preprocess a pandas DataFrame by imputing missing values and one-hot encoding categorical variables.
    Parameters:
    df : pandas DataFrame, input DataFrame to preprocess
    numeric_strategy : str, strategy for imputing numeric columns (default: 'mean')
    cat_strategy : str, strategy for imputing categorical columns (default: 'most_frequent')

    Returns:
    X : numpy array, preprocessed feature matrix
    ohe : OneHotEncoder, fitted OneHotEncoder for categorical variables
    imp_num : SimpleImputer, fitted SimpleImputer for numeric variables
    imp_cat : SimpleImputer, fitted SimpleImputer for categorical variables
    names : list, names of the features after preprocessing
    """
    # Ensure the input is a pandas DataFrame and separate numeric and categorical columns
    df_num = df.select_dtypes(include=[np.number])
    df_cat = df.select_dtypes(exclude=[np.number])

    # ---- numeric part ------------------------------------------------------
    imp_num = SimpleImputer(strategy=numeric_strategy)
    X_num   = imp_num.fit_transform(df_num)

    # ---- categorical part --------------------------------------------------
    if df_cat.shape[1] > 0:
        imp_cat = SimpleImputer(strategy=cat_strategy)
        X_cat   = imp_cat.fit_transform(df_cat)

        enc     = OneHotEncoder(sparse=False, handle_unknown='ignore')
        X_cat_ohe = enc.fit_transform(X_cat)

        cat_names = list(enc.get_feature_names_out(df_cat.columns))
    else:
        # nothing categorical
        imp_cat   = None
        enc       = None
        X_cat_ohe = np.empty((len(df), 0))
        cat_names = []

    # ---- concatenate & return ---------------------------------------------
    X = np.hstack([X_num, X_cat_ohe]).astype(float)
    feat_names = list(df_num.columns) + cat_names
    return X, enc, imp_num, imp_cat, feat_names

# ----------------------------------------------------------------------
# Section 4 : Splitting using C4.5-like information gain
# ----------------------------------------------------------------------
def _info_gain_numeric(X_col, y):
    """
    Function to compute the best threshold and information gain for a numeric attribute. 
    Information gain is defined as the difference between the entropy of the parent node and the weighted average entropy of the child nodes. IG = entropy(parent) - (p_left * entropy(left) + p_right * entropy(right))
    Parameters:
    X_col : array-like, numeric attribute values for the samples
    y : array-like, class labels for the samples

    Returns:
    best_gain : float, the best information gain found
    best_thr : float, the threshold that gives the best information gain
    """
    # Ensure the input is a numpy array and sort the values
    X_col = np.asarray(X_col, float)
    sort_idx  = np.argsort(X_col)
    X_sorted  = X_col[sort_idx]
    y_sorted  = y[sort_idx]

    # Initialize variables for the best gain and threshold
    best_gain, best_thr = -np.inf, None
    # Compute the entropy of the parent node
    # parent_H = -∑p(I)*log2p(I) where p(I) is the probability of class I
    parent_H = _entropy(y)

    # candidate thresholds = mid-points where class label actually changes
    # Iterate over the sorted values to find the best threshold
    for i in range(1, len(y_sorted)):
        # Skip if the current and previous class labels are the same
        if y_sorted[i-1] == y_sorted[i]:
            continue
        # Compute the threshold as the average of the current and previous values
        thr = (X_sorted[i-1] + X_sorted[i]) / 2.0
        # Create masks for left and right splits based on the threshold
        left_mask  = X_col <= thr
        right_mask = ~left_mask
        # If either split has less than 2 samples, skip this threshold
        if left_mask.sum() < 2 or right_mask.sum() < 2:
            continue
        # Compute the entropy for the left and right splits
        H_left  = _entropy(y[left_mask])
        H_right = _entropy(y[right_mask])
        # Compute the information gain
        gain = parent_H - (left_mask.mean() * H_left + right_mask.mean() * H_right)
        # If the gain is better than the best found so far, update the best gain and threshold
        if gain > best_gain:
            best_gain, best_thr = gain, thr
    return best_gain, best_thr


def _best_split(X, y, min_info_gain=1e-6):
    """
    Function to find the best attribute and threshold for splitting the dataset based on information gain.
    Parameters:
    X : array-like, shape (n_samples, n_features), feature matrix
    y : array-like, shape (n_samples,), class labels
    min_info_gain : float, minimum information gain to consider a split (default: 1e-6)

    Returns:
    best_attr : int, index of the best attribute for splitting
    best_thr : float, threshold for the best attribute
    best_gain : float, information gain for the best split
    """
    # Initialize variables to track the best attribute, threshold, and gain
    best_attr, best_thr, best_gain = None, None, 0
    # Iterate over each feature in the dataset
    for k in range(X.shape[1]):
        # Compute the information gain for the k-th feature
        col = X[:, k]
        gain, thr = _info_gain_numeric(col, y)
        # If the gain is better than the best found so far and meets the minimum threshold, update the best values
        if gain > best_gain and gain >= min_info_gain:
            best_attr, best_thr, best_gain = k, thr, gain
    return best_attr, best_thr, best_gain

# ----------------------------------------------------------------------
# Section 5 : Tree growth
# ----------------------------------------------------------------------

# helper – pad / map local → global
def _pad_to_global(p_local, J_global):
    """
    Expand a local probability vector (length J_node ≤ J_global)
    to length J_global by zero-padding the *missing* classes.
    """
    p_full = np.zeros(J_global, dtype=float)
    p_full[: len(p_local)] = p_local          # assumes labels 0..k map 1-to-1
    return p_full


def _build_node(X, y, parent_model, J_global, max_iter=200, min_split_size=15, min_lr_size=5, min_info_gain=1e-6):
    """
    Function to recursively build a decision tree node for logistic model trees (LMT).
    Parameters:
    X : array-like, shape (n_samples, n_features), feature matrix
    y : array-like, shape (n_samples,), class labels
    parent_model : tuple (learners, J) or None, model from the parent node for warm-starting
    J_global : int, number of classes in the global context
    max_iter : int, maximum number of boosting rounds for logistic regression (default: 200)
    min_split_size : int, minimum number of samples required to consider splitting (default: 15)
    min_lr_size : int, minimum number of samples required to fit a logistic regression model (default: 5)
    min_info_gain : float, minimum information gain to consider a split (default: 1e-6)

    Returns:
    node : dict, a dictionary representing the tree node with the following keys:
          - 'is_leaf': bool, True if the node is a leaf
          - 'model': tuple (learners, J) or None, model for the node
          - 'split_attr': int or None, index of the attribute used for splitting
          - 'split_thr': float or None, threshold value for the split
          - 'left': dict, left child node
          - 'right': dict, right child node
    """
    n = len(y)
    # Decide whether to fit a logistic model here
    model = None
    if n >= min_lr_size:
        model = logitboost_fit_cv(X, y, max_estimators=max_iter)
    else:
        model = parent_model # inherit

    # Splitting conditions
    # If there are not enough samples to split, return a leaf node
    if n < min_split_size:
        return {'is_leaf':True,  'model':model, 'split_attr':None, 'split_thr':None}
    
    # Find the best attribute and threshold for splitting
    attr, thr, gain = _best_split(X, y, min_info_gain)

    # If no viable split is found, return a leaf node
    if attr is None:
        return {'is_leaf':True,  'model':model, 'split_attr':None,'split_thr':None}
    
    # Create masks for left and right splits based on the threshold
    left_idx  = X[:, attr] <= thr
    right_idx = ~left_idx
    # If either split has less than 2 samples, return a leaf node
    if left_idx.sum() < 2 or right_idx.sum() < 2:
        return {'is_leaf':True,  'model':model, 'split_attr':None,'split_thr':None}

    # Recursively build the left and right child nodes
    # Note: we pass the parent model for warm-starting
    left_child  = _build_node(X[left_idx],  y[left_idx], parent_model=model, J_global=J_global, max_iter=max_iter, min_split_size=min_split_size, min_lr_size=min_lr_size, min_info_gain=min_info_gain)
    right_child = _build_node(X[right_idx], y[right_idx], parent_model=model, J_global=J_global, max_iter=max_iter, min_split_size=min_split_size, min_lr_size=min_lr_size, min_info_gain=min_info_gain)
    # Return the node with the model, split attribute, threshold, and child nodes
    return {'is_leaf':False, 'model':model, 'split_attr':attr, 'split_thr':thr, 'left':left_child, 'right':right_child}


def lmt_fit(df, target, max_iter=200, min_split_size=15, min_lr_size=5, min_info_gain=1e-6):
    """
    Function to fit a logistic model tree (LMT) to a pandas DataFrame.
    Parameters:
    df : pandas DataFrame, input DataFrame containing features and target variable
    target : str, name of the target variable column in the DataFrame
    max_iter : int, maximum number of boosting rounds for logistic regression (default: 200)
    min_split_size : int, minimum number of samples required to consider splitting (default: 15)
    min_lr_size : int, minimum number of samples required to fit a logistic regression model (default: 5)
    min_info_gain : float, minimum information gain to consider a split (default: 1e-6)

    Returns:
    tree : dict, the root node of the fitted logistic model tree
    meta : dict, metadata containing the encoder, imputers, feature names, and target name
    """
    # Drop the target column from the DataFrame and preprocess the features
    X_raw  = df.drop(columns=[target])
    y      = df[target].values
    X, enc, imp_num, imp_cat, feat_names = preprocess_dataframe(X_raw)

    J_global      = int(y.max()) + 1          # assumes labels are 0..J-1
    label_set_all = np.arange(J_global)
    
    # Build the logistic model tree
    # Note: we pass None as the parent_model for the root node
    tree   = _build_node(X, y, parent_model=None, J_global=J_global, max_iter=max_iter, min_split_size=min_split_size, min_lr_size=min_lr_size, min_info_gain=min_info_gain)
    # Create metadata for the fitted model
    meta = dict(encoder=enc, imp_num=imp_num, imp_cat=imp_cat, feat_names=feat_names, target_name=target, J=J_global, labels=label_set_all)
    
    return tree, meta

# ----------------------------------------------------------------------
# Section 6 : Prediction
# ----------------------------------------------------------------------
def _predict_single_proba(x, node, J_global):
    """
    Function to predict the class probabilities for a single sample x using the logistic model tree node.
    Parameters:
    x : array-like, shape (n_features,), input sample for which to predict probabilities
    node : dict, the tree node containing the model and split information

    Returns:
    proba : array, predicted class probabilities for the sample x
    """
    # Traverse the tree until a leaf node is reached
    while not node['is_leaf']:
        if x[node['split_attr']] <= node['split_thr']:
            node = node['left']
        else:
            node = node['right']
    # At the leaf node, use the model to predict probabilities
    # Note: node['model'] is a tuple (learners, J)
    learners, J_local = node['model']

    p_local  = logitboost_predict_proba(x.reshape(1,-1), learners, J_local)[0]
    # Use the logitboost_predict_proba function to get the probabilities for the sample x
    return _pad_to_global(p_local, J_global)


def lmt_predict_proba(df, tree, meta):
    """
    Function to predict class probabilities for a pandas DataFrame using a fitted logistic model tree.
    Parameters:
    df : pandas DataFrame, input DataFrame containing features for prediction
    tree : dict, the fitted logistic model tree
    meta : dict, metadata containing the encoder, imputers, feature names, and target name

    Returns:
    proba : numpy array, predicted class probabilities for each sample in the DataFrame
    """
    X_raw = df.copy()
    # Impute missing values and encode categorical variables
    X_num = meta['imp_num'].transform(X_raw.select_dtypes(include=[np.number]))
    if meta['encoder'] is not None and X_raw.select_dtypes(exclude=[np.number]).shape[1] > 0:
        X_cat = meta['imp_cat'].transform(X_raw.select_dtypes(exclude=[np.number]))
        X_cat = meta['encoder'].transform(X_cat)
        X     = np.hstack([X_num, X_cat]).astype(float)
    else:
        X     = X_num.astype(float)
    
    J_global = meta['J']
    # Predict probabilities for each sample in the DataFrame
    return np.vstack([_predict_single_proba(x, tree, J_global) for x in X])


def lmt_predict(df, tree, meta):
    """
    Function to predict class labels for a pandas DataFrame using a fitted logistic model tree. 
    Parameters:
    df : pandas DataFrame, input DataFrame containing features for prediction
    tree : dict, the fitted logistic model tree
    meta : dict, metadata containing the encoder, imputers, feature names, and target name

    Returns:
    labels : numpy array, predicted class labels for each sample in the DataFrame
    """
    return lmt_predict_proba(df, tree, meta).argmax(axis=1)

#PENDING
# ----------------------------------------------------------------------
# Section 7 : Cost complexity pruning based on CART-style pruning
# ----------------------------------------------------------------------
