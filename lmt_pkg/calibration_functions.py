from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.gridspec import GridSpec
from . import logitboost_j_implementation as logitboost

# ------------------------------------------------ #
# Function to compare the different depth trees    #
# regarding the calibration curves and hists       #
# ------------------------------------------------ #
def plot_combined_calibration_and_hists(
    y_test, X_test,
    y_test_ext, X_test_ext,
    clf_orig_list, clf_ext_list,
    model_labels=["Shallow", "Regular", "Overfit", "Pruned"],
    n_bins=10
):
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 1.5, 1.5])
    
    # Axes
    ax_curve = fig.add_subplot(gs[0, :])
    hist_axes = [fig.add_subplot(gs[i, j]) for i in [1, 2] for j in [0, 1]]

    colors = plt.get_cmap("tab10")

    # Plot all calibration curves in the same axis
    for idx, (clf_orig, clf_ext) in enumerate(zip(clf_orig_list, clf_ext_list)):
        proba_orig = clf_orig.predict_proba(X_test)[:, 1]
        proba_ext = clf_ext.predict_proba(X_test_ext)[:, 1]
        CalibrationDisplay.from_predictions(
            y_test, proba_orig, n_bins=n_bins,
            name=f"{model_labels[idx]} - Original",
            ax=ax_curve, color=colors(2*idx)
        )
        CalibrationDisplay.from_predictions(
            y_test_ext, proba_ext, n_bins=n_bins,
            name=f"{model_labels[idx]} - Extended",
            ax=ax_curve, color=colors(2*idx+1)
        )

    ax_curve.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax_curve.set_title("Calibration Plots – All Trees")
    ax_curve.set_xlabel("Mean predicted probability (Positive class: 1)")
    ax_curve.set_ylabel("Fraction of positives (Positive class: 1)")
    ax_curve.legend(loc='best')
    ax_curve.grid(True)

    # Plot individual histograms
    for idx, (clf_orig, clf_ext) in enumerate(zip(clf_orig_list, clf_ext_list)):
        ax = hist_axes[idx]
        proba_orig = clf_orig.predict_proba(X_test)[:, 1]
        proba_ext = clf_ext.predict_proba(X_test_ext)[:, 1]

        ax.hist(proba_orig, bins=n_bins, range=(0, 1), alpha=0.7, color=colors(2*idx), label='Original')
        ax.hist(proba_ext, bins=n_bins, range=(0, 1), alpha=0.7, color=colors(2*idx+1), label='Extended')
        ax.set_title(f"{model_labels[idx]} Tree")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout()
    plt.show()


#import lmt_final_implementation as lmt
from . import lmt_final_implementation as lmt

def plot_lmt_combined_calibration_and_hists(
    X_test, y_test,
    X_test_ext, y_test_ext,
    clfs, nodes,
    clfs_ext, nodes_ext,
    model_labels=["Shallow", "Regular", "Overfit", "Pruned"],
    n_bins=10
):
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3, 1.5, 1.5])
    
    # Calibration curve plot (top row, spans both columns)
    ax_curve = fig.add_subplot(gs[0, :])
    hist_axes = [fig.add_subplot(gs[i, j]) for i in [1, 2] for j in [0, 1]]

    colors = plt.get_cmap("tab10")

    # Plot all calibration curves in a single axis
    for idx, (clf, node_model, clf_ext, node_model_ext) in enumerate(zip(clfs, nodes, clfs_ext, nodes_ext)):
        proba_orig = lmt.predict_proba_lmt(X_test, clf, node_model)
        proba_ext  = lmt.predict_proba_lmt(X_test_ext, clf_ext, node_model_ext)

        CalibrationDisplay.from_predictions(
            y_test, proba_orig, n_bins=n_bins,
            name=f"{model_labels[idx]} - Original",
            ax=ax_curve, color=colors(2 * idx)
        )
        CalibrationDisplay.from_predictions(
            y_test_ext, proba_ext, n_bins=n_bins,
            name=f"{model_labels[idx]} - Extended",
            ax=ax_curve, color=colors(2 * idx + 1)
        )

    ax_curve.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax_curve.set_title("Calibration Plots – All Trees")
    ax_curve.set_xlabel("Mean predicted probability (Positive class: 1)")
    ax_curve.set_ylabel("Fraction of positives (Positive class: 1)")
    ax_curve.legend(loc='best')
    ax_curve.grid(True)

    # Plot individual histograms
    for idx, (clf, node_model, clf_ext, node_model_ext) in enumerate(zip(clfs, nodes, clfs_ext, nodes_ext)):
        ax = hist_axes[idx]
        proba_orig = lmt.predict_proba_lmt(X_test, clf, node_model)
        proba_ext  = lmt.predict_proba_lmt(X_test_ext, clf_ext, node_model_ext)

        ax.hist(proba_orig, bins=n_bins, range=(0, 1), alpha=0.7, color=colors(2 * idx), label="Original")
        ax.hist(proba_ext, bins=n_bins, range=(0, 1), alpha=0.7, color=colors(2 * idx + 1), label="Extended")
        ax.set_title(f"{model_labels[idx]} Tree")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout()
    plt.show()




# ------------------------------------------------------ #
# Function to make the scatter plot based on quantiles   #
# ------------------------------------------------------ #
import matplotlib as mpl
def plot_scatter_by_quantile(
    X,
    y_proba,
    n_quantiles=5,
    title="Scatter by Quantile",
    ax=None,
    feature_1=0,
    feature_2=1,
    cmap='Accent',
    norm=None,
    q_edges=None
):
    """
    X: array of shape (n_samples, n_features)
    y_proba: array of predicted probabilities (n_samples,)
    ax: Matplotlib Axes (if None, plt.gca() is used)
    norm: Optional Normalize instance (default: Normalize(0, n_quantiles - 1))
    q_edges: Optional precomputed quantile edges to ensure consistency across plots
    """
    if ax is None:
        ax = plt.gca()

    # 1) Use provided or compute quantile edges
    if q_edges is None:
        q_edges = np.quantile(y_proba, np.linspace(0, 1, n_quantiles + 1))

    # 2) Digitize probabilities into quantile bins
    quantile_ids = np.digitize(y_proba, bins=q_edges[1:-1], right=True)
    quantile_ids = np.clip(quantile_ids, 0, n_quantiles - 1)

    # 3) Normalize colors
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=n_quantiles - 1)

    # 4) Get discrete colormap
    cmap = mpl.cm.get_cmap(cmap, n_quantiles)

    # 5) Plot scatter
    sc = ax.scatter(
        X[:, feature_1], X[:, feature_2],
        c=quantile_ids,
        cmap=cmap,
        norm=norm,
        edgecolor='k',
        alpha=0.8,
        s=20
    )

    ax.set_title(title)
    ax.set_xlabel(f"Feature {feature_1}")
    ax.set_ylabel(f"Feature {feature_2}")
    ax.grid(True)

    return sc


# ----------------------------------------- #
# Predicted vs True probabilty plot         #
# ----------------------------------------- #
def plot_pred_vs_true(
    models: dict,
    p_true: np.ndarray,
    X_test: np.ndarray,
    model_type: str = 'tree',
    orig_cols: int = 2,
    figsize: tuple = (12, 5)
):
    """
    models: 
      - if model_type=='tree': { name: clf, ... }
      - if model_type=='lmt' : { name: (clf, nodes), ... }
      - if model_type=='logitboost': { name: (learners, J), ... }
    p_true: array of true probabilities, shape (n_samples,)
    X_test: feature matrix, shape (n_samples, n_features)
    model_type: 'tree', 'lmt' or 'logitboost'
    orig_cols: how many cols from X_test to use for models tagged as '(orig)'
    """
    import matplotlib.cm as cm
    colors = cm.get_cmap("tab10")
    markers = ['o', '^', 's', 'v', '<', '>', 'X', 'D']

    # Create shared subplots for all models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for idx, ((name, val), marker) in enumerate(zip(models.items(), markers)):
        # Get X subset based on model name
        X_in = X_test[:, :orig_cols] if '(orig)' in name else X_test

        # Predict based on model type
        if model_type == 'tree':
            clf = val
            y_pred = clf.predict_proba(X_in)[:, 1]
        elif model_type == 'lmt':
            clf, nodes = val
            y_pred = lmt.predict_proba_lmt(X_in, clf, nodes)
        elif model_type == 'logitboost':
            learners, J = val
            y_pred = logitboost.logitboost_predict_proba(X_in, learners, J)[:, 1]
        else:
            raise ValueError("model_type must be 'tree', 'lmt', or 'logitboost'")

        # Scatter plot
        ax1.scatter(
            p_true, y_pred,
            alpha=0.5, s=20,
            marker=marker,
            label=name,
            color=colors(idx % 10)
        )

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("True probability $p_i$")
    ax1.set_ylabel("Predicted probability $\hat p_i$")
    ax1.set_title("Predicted vs True Probabilities")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize='small')

    # Histogram of true probabilities
    ax2.hist(p_true, bins=20, color=colors(2), alpha=0.7)
    ax2.set_title("Histogram of True Probabilities")
    ax2.set_xlabel("True probability $p_i$")
    ax2.set_ylabel("Count")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
