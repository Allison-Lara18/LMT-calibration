from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib.gridspec import GridSpec

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


import lmt_final_implementation as lmt

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
def plot_scatter_by_quantile(X, y_proba, n_quantiles=5, title="Scatter by Quantile", ax=None, feature_1=0, feature_2=1, cmap='Accent', norm=None):
    """
    X: array of shape (n_samples, n_features)
    y_proba: array of shape (n_samples,) with predicted probabilities
    ax:    a matplotlib Axes (if None, plt.gca() is used)
    cmap_name: name of a matplotlib colormap
    norm:  a matplotlib Normalize instance; if None, we create Normalize(0, n_quantiles-1)
    """
    if ax is None:
        ax = plt.gca()

    # 1) compute the quantile bins and assign each point an integer ID [0..n_quantiles-1]
    q_edges = np.quantile(y_proba, np.linspace(0, 1, n_quantiles+1))
    # digitize into bins [q_edges[0],q_edges[1]), [q_edges[1],q_edges[2]), …, [q_edges[-2],q_edges[-1]]
    quantile_ids = np.digitize(y_proba, bins=q_edges[1:-1], right=True)
    quantile_ids = np.clip(quantile_ids, 0, n_quantiles-1)

    # 2) ensure a global Normalize for 0..n_quantiles-1
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=n_quantiles-1)

    # 3) get a discrete colormap with n_quantiles colors
    cmap = mpl.cm.get_cmap(cmap, n_quantiles)

    # 4) scatter
    sc = ax.scatter(
        X[:, feature_1], X[:, feature_2],
        c=quantile_ids,
        cmap=cmap,
        norm=norm,
        edgecolor='k',
        alpha=0.8,
        s=40
    )

    ax.set_title(title)
    ax.set_xlabel(f"Feature {feature_1}")
    ax.set_ylabel(f"Feature {feature_2}")
    ax.grid(True)

    # return the PathCollection so we can use it for a shared colorbar
    return sc


# ----------------------------------------- #
# Predicted vs True probabilty plot         #
# ----------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np

def plot_pred_vs_true(
    models: dict,
    p_true: np.ndarray,
    X_test: np.ndarray,
    model_type: str = 'tree',
    orig_cols: int = 2,
    figsize: tuple = (10, 8)
):
    """
    models: 
      - if model_type=='tree': { name: clf, ... }
      - if model_type=='lmt' : { name: (clf, nodes), ... }
    p_true: array of true probabilities, shape (n_samples,)
    X_test: feature matrix, shape (n_samples, n_features)
    model_type: 'tree' or 'lmt'
    orig_cols: how many cols de X_test usar para los modelos “orig”
    """
    # marcadores diferentes para cada serie
    markers = ['o','s','^','v','<','>','X','D']
    
    plt.figure(figsize=figsize)
    
    for (name, val), marker in zip(models.items(), markers):
        # 1) descompón val en clf/nodes
        if model_type == 'tree':
            clf = val
            nodes = None
        elif model_type == 'lmt':
            clf, nodes = val
        else:
            raise ValueError("model_type must be 'tree' or 'lmt'")
        
        # 2) decide X_in según 'orig' o 'ext'
        if '(orig)' in name:
            X_in = X_test[:, :orig_cols]
        else:
            X_in = X_test
        
        # 3) obtén y_pred
        if model_type == 'tree':
            y_pred = clf.predict_proba(X_in)[:, 1]
        else:  # 'lmt'
            # asume que tienes importado tu módulo lmt
            y_pred = lmt.predict_proba_lmt(X_in, clf, nodes)
        
        # 4) scatter
        plt.scatter(
            p_true, y_pred,
            alpha=0.6,
            s=50,
            marker=marker,
            label=name
        )
    
    # 5) línea y=x
    lims = [0.0, 1.0]
    plt.plot(lims, lims, 'k--', label='Perfect calibration')
    
    # 6) detalles
    plt.xlabel("True probability")
    plt.ylabel("Predicted probability")
    plt.title("Predicted vs True probabilities")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()
