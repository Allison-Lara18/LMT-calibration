# Linear–Spline Tree (Binary Classification)

## 1) Goal

Learn a decision tree for $y\in\{0,1\}$ where every internal split:

* uses **one** feature $X_j$ and **one** knot $\tau$;
* evaluates a **linear spline** basis $\{1,\, x_{i,j},\, (x_{i,j}-\tau)_+\}$;
* fits a **logistic GLM** at the node to score the split;
* grows until structural stopping rules trigger, then applies **post-pruning** (based on node-level GLM metrics such as AUC).

---

## 2) Setup and notation

Data $D=\{(x_i,y_i)\}_{i=1}^n$, with $x_i=(x_{i,1},\dots,x_{i,d})\in\mathbb{R}^d$, $y_i\in\{0,1\}$.

For $a\in\mathbb{R}$, define the truncated linear (ReLU) term $(a)_+=\max\{0,a\}$.

Logistic link $\sigma(z)=\dfrac{1}{1+e^{-z}}$.

For a node containing index set $\mathcal{I}\subseteq\{1,\dots,n\}$, let $n=|\mathcal{I}|$, class proportion $\bar{y}=\dfrac{1}{n}\sum_{i\in\mathcal{I}} y_i$, and purity $\text{purity}=\max\{\bar{y},\,1-\bar{y}\}$.

---

## 3) Spline split model (per feature $j$ and knot $\tau$)

Define the node-local linear spline score

$$
h_{j,\tau}(x_{i,j})=\beta_0+\beta_1\,x_{i,j}+\beta_2\,(x_{i,j}-\tau)_+.
$$

Equivalently, piecewise-linear with a slope change at $\tau$:

$$
h_{j,\tau}(x)=
\begin{cases}
\beta_0+\beta_1 x, & x\le \tau,\\
\beta_0+(\beta_1+\beta_2)\,x-\beta_2\,\tau, & x>\tau.
\end{cases}
$$

Class probability:

$$
p_i=\Pr(y_i=1\mid x_{i,j})=\sigma\!\left(h_{j,\tau}(x_{i,j})\right).
$$

Parameters $(\beta_0,\beta_1,\beta_2)$ are estimated by **logistic regression** on the node’s data using only the design columns $[1,\,x_{i,j},\,(x_{i,j}-\tau)_+]$.

---

## 4) Candidate splits and partition

For each feature $j$ and candidate knot $\tau$ (e.g., percentiles of $\{\,x_{i,j}\,\}_{i\in\mathcal{I}}$; typically 10%–90% or unique midpoints):

* define the **hard split**:
  * Left child $\mathcal{I}_L=\{\, i\in\mathcal{I} : x_{i,j}\le \tau \,\}$,
  * Right child $\mathcal{I}_R=\{\, i\in\mathcal{I} : x_{i,j}> \tau \,\}$.
* enforce **minimum child size**: $|\mathcal{I}_L|\ge \texttt{min\_samples}$ and $|\mathcal{I}_R|\ge \texttt{min\_samples}$; otherwise the split is invalid.

---

## 5) Split scoring (AUC or Brier)

Fit the GLM once for $(j,\tau)$ on the **parent** node’s data; compute its predicted probabilities $\{\,p_i\,\}_{i\in\mathcal{I}}$.  
Evaluate a metric **within each child**:

* **AUC** on $\mathcal{I}_L$ and $\mathcal{I}_R$ (requires at least one positive and one negative in the child; otherwise the child’s AUC is undefined and the split is skipped), or
* **Brier score** $\mathrm{Brier}=\dfrac{1}{|\mathcal{I}_\bullet|}\sum_{i\in\mathcal{I}_\bullet}(p_i-y_i)^2$ for $\bullet\in\{L,R\}$.

Combine by size-weighted aggregation:

$$
\mathrm{Score}(j,\tau)=
\begin{cases}
\displaystyle \frac{n_L}{n}\,\operatorname{AUC}_L+\frac{n_R}{n}\,\operatorname{AUC}_R, & \text{maximize},\\[1.0em]
\displaystyle \frac{n_L}{n}\,\mathrm{Brier}_L+\frac{n_R}{n}\,\mathrm{Brier}_R, & \text{minimize}.
\end{cases}
$$

Select $(j^\star,\tau^\star)$ giving the **best** score (highest AUC or lowest Brier).

---

## 6) Stopping rules (create a leaf)

At a node with data $\mathcal{I}$, stop and return a leaf if any holds:

1. depth $\ge$ `max_depth`;
2. $n < 2 \times \texttt{min\_samples}$ (cannot form two valid children);
3. $\text{purity} \ge$ `purity_threshold` (e.g., 0.95).

The leaf prediction is the node’s class probability $\hat{p}=\bar{y}$ (or a calibrated GLM estimate if you keep the last fit).

---

## 7) Post-pruning (after full growth)

Grow the maximal tree under the stopping rules above, then prune bottom-up:

* For each internal node, evaluate a validation metric (e.g., **AUC**) for the subtree vs. the node collapsed to a leaf (using node-level GLM or empirical $\bar y$).
* Replace a subtree by a leaf when it **improves** the chosen criterion (or under a cost–complexity trade-off). Select the final subtree by validation AUC (or Brier) across candidates.

---

## 8) Algorithm

1. **Node Check** ($\mathcal{I}$, depth):

   * if a stopping rule holds ⇒ return **Leaf** with $\hat p=\bar y$.
2. **Search Splits**:

   * For each $j\in\{1,\dots,d\}$:

     * Generate candidate $\tau$ values.
     * For each $\tau$:

       * Form design $[1, x_{i,j}, (x_{i,j}-\tau)_+]$ for $i\in\mathcal{I}$; fit logistic regression; get $p_i$.
       * Form children $\mathcal{I}_L,\mathcal{I}_R$; if either too small, continue.
       * Compute child metrics (AUC or Brier); aggregate to $\text{Score}(j,\tau)$.
   * Choose best valid $(j^\star,\tau^\star)$ (if none, return **Leaf**).
3. **Recurse**:

   * Create **InternalNode** with feature $j^\star$, threshold $\tau^\star$.
   * Left  $\leftarrow$ **NodeCheck**($\mathcal{I}_L$, depth+1).
   * Right $\leftarrow$ **NodeCheck**($\mathcal{I}_R$, depth+1).
4. **Prune** the grown tree using validation AUC (or Brier).

---

## 9) Practical notes

* **Metric direction:** When mixing metrics, treat all as “maximize” by negating those to minimize (e.g., use -Brier).
* **AUC edge cases:** Skip a split if a child has all-positive or all-negative labels (AUC undefined).
* **Candidate knots:** Percentiles reduce computational load and guard against unstable knots at extremes; unique midpoints offer finer search if $n$ is moderate.

