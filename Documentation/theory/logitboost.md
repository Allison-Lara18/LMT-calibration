
# LogitBoost

**Boosting** builds a strong classifier by adding up many weak learners trained sequentially on reweighted data. At each round, examples that are currently hard to predict get more influence, and the new weak learner is fitted to correct the current mistakes. **LogitBoost** is the version that performs Newton steps on the (multiclass) logistic loss; each weak learner is a **one-attribute weighted least-squares regression**.

In practice (as in the attached code), every boosting round chooses **one single feature** for each class and fits a **weighted simple linear regression** on that feature. The ensemble prediction is the **sum** of these per-round functions.

---

## Model and objective

Let $J$ be the number of classes. LogitBoost maintains real-valued **additive scores**

$$
F_j(x) \;=\; \sum_{m=1}^M f_{mj}(x), \qquad j=0,\dots,J-1,
$$

constrained so that $\sum_j F_j(x)=0$ (enforced by a centering step each round). Class probabilities come from the softmax

$$
p_j(x) \;=\; \frac{e^{F_j(x)}}{\sum_{k=0}^{J-1} e^{F_k(x)}}.
$$

The loss is the multinomial negative log-likelihood. A **Newton step in function space** leads to per-class working responses and weights:

$$
z_{ij} \;=\; \frac{y_{ij} - p_j(x_i)}{p_j(x_i)\,[1-p_j(x_i)]}, 
\qquad
w_{ij} \;=\; p_j(x_i)\,[1-p_j(x_i)],
$$

where $y_{ij}=1$ if $y_i=j$ and $0$ otherwise.

---

## Weak learner (what gets fitted each round)

For each class $j$ and boosting round $m$:

* We fit a **single-feature weighted linear model** $f_{mj}(x)=b_{0j}+b_{1j}\,x_{k_j}$ by minimizing

  $$
  \sum_{i=1}^{N} w_{ij}\,\bigl(z_{ij}-(b_{0j}+b_{1j}x_{ik})\bigr)^2
  $$
  **over all features $k$** and pick the feature $k_j$ that achieves the **smallest weighted squared error**.  
  (In code: `_best_feature_lr(X, z, w)` loops over features, uses `LinearRegression(..., sample_weight=w)`, and returns the best $(k, b_0, b_1, \hat f)`.)

* After fitting every class, stack the $J$ fitted vectors $\hat{f}_{mj}\in\mathbb{R}^N$, compute their mean across classes, and **center**:
  $$
  \tilde{f}_{mj}\;=\;\frac{J-1}{J}\,\Bigl(\hat{f}_{mj}-\frac{1}{J}\sum_{k=0}^{J-1}\hat{f}_{mk}\Bigr).
  $$
  This guarantees $\sum_j \tilde{f}_{mj}=0$ for each sample.

* Update scores and probabilities:

  $$
  F_j \leftarrow F_j + \tilde{f}_{mj},
  \qquad
  p \leftarrow \operatorname{softmax}(F).
  $$
  (In code: numerically stable softmax; probabilities are clipped to $[\varepsilon,\,1-\varepsilon]$ for safety.)

**Complexity per round:** $O\big(J \cdot D \cdot N\big)$ for $J$ classes, $D$ features, $N$ samples, since for each class we fit $D$ one-dimensional weighted regressions and pick the best.

---

## Training API (multiclass)

### `learners, J = logitboost_fit(X, y, n_estimators=..., eps=..., warm_start=None)`

* **Inputs:**
  `X`: $(N,D)$, `y`: integer labels $\{0,\dots,J-1\}$.
  `n_estimators = M`: number of boosting rounds.
  `eps`: probability clipping (numeric stability).
  `warm_start`: optionally continue from an existing `(learners, J)`.

* **Initialization:** If `warm_start` is absent: $F=0$, $p_j=1/J$. If present: reconstruct $F$ from the existing learners.

* **Output:**
  `learners`: a list of length $M$; each entry is a list of $J$ tuples `(feat_idx, b0, b1)` (one feature and linear params **per class** for that round).
  `J`: inferred number of classes.

### Prediction

* `logitboost_predict_proba(X, learners, J)` returns $(N,J)$ probabilities via softmax of accumulated scores.
* `logitboost_predict(X, learners, J)` returns `argmax_j F_j(x)`.

---

## Binary case ($J=2$)

When $J=2$ the softmax reduces to the logistic sigmoid. Writing the two scores as $F_1=-F_0$,

$$
p(y=1\mid x) \;=\; \frac{e^{F_1(x)}}{e^{F_0(x)}+e^{F_1(x)}} 
= \frac{e^{F_1(x)}}{e^{-F_1(x)}+e^{F_1(x)}} 
= \sigma \bigl(2F_1(x)\bigr),
$$

so the model is equivalent to a logistic regression in the **additive score** $F_1$. Working responses and weights specialize to the binomial form above.

---

## SimpleLogistic (model selection via 5-fold CV)

`simple_logistic_fit(X, y, n_estimators=200, cv_splits=5, eps=1e-5, warm_start=None, random_state=0)`

1. **Cross-validation:** For each fold, train LogitBoost for `n_estimators` rounds (fresh fit), and record the **misclassification error** on the validation split **at every round** $m=1,\ldots,M$.
2. **Select $M^*$:** Average validation errors across folds; pick $M^*$ with the smallest mean error.
3. **Final fit:** Train on **all** data for $M^*$ rounds. If a `warm_start` has at least $M^*$ rounds, truncate; otherwise continue training to reach $M^*$.

**Returns:** `final_learners`, `J_node` (classes present in `y`), `M_star`, and the mean CV error curve `cv_errs_mean`.

---

## Practical notes

* **Numerics:** Probabilities are clipped to $[\varepsilon,1-\varepsilon]$; softmax is implemented in a stable way by subtracting the row-wise max.
* **Feature selection, naturally:** Only features chosen by some round enter the model; features can be **reused** across rounds to refine the boundary.
* **Early stopping:** Despite handling $M \gg D$ just fine, choose $M$ by validation (SimpleLogistic) to avoid over-smoothing probabilities or needless computation.

---

## Summary

LogitBoost builds the additive score functions $F_j$ by repeated **one-attribute weighted least-squares** fits that implement a **Newton step** on the logistic loss. The attached implementation:

* fits one linear weak learner **per class and round**, choosing the best feature by weighted SSE,
* recenters updates to satisfy $\sum_j F_j=0$,
* predicts with softmax (or `argmax` for hard labels),
* and includes a SimpleLogistic routine (5-fold CV) to pick the optimal number of boosting rounds.
