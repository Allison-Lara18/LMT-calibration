# Regression Pruning for Logistic Model Trees

## Setup and notation

* $T$ is a **binary** tree. Each internal node $v$ has children $L(v), R(v)$; leaves have none.
* Class set $\mathcal{Y}=\{1,\dots,C\}$ with $C\ge 2$.
* Each node $v$ carries its **own** LogitBoost model $M_v$, fitted during LMT induction on the node’s training reach set, and **frozen** during pruning.

  * Let $F_v(x)\in\mathbb{R}^C$ be the additive score vector from LogitBoost and $p_v(x)=\text{softmax}\!\big(F_v(x)\big)\in\Delta^{C-1}$.
* For labeled data $S=\{(x_i,y_i)\}_{i=1}^n$, the **reach set** of $v$ is

    $$
    S_v=\{\, i :\; x_i \text{ routes to } v \,\},\qquad n_v:=|S_v|.
    $$

  Collect the node probabilities as $P_{v,S_v}=\big[p_v(x_i)\big]_{i\in S_v}\in\mathbb{R}^{n_v\times C}$.

### Node AUC (explicit binary vs. multiclass rule)

$$
\mathrm{AUC}(M_v; S_v)=
\begin{cases}
\text{ROC-AUC}_{\text{binary}}\bigl(y_{S_v},\, s_{S_v}\bigr), & \text{if } C=2 \text{ and both classes present},\\[4pt]
\text{ROC-AUC}_{\text{macro OvR}}\bigl(y_{S_v},\, P_{v,S_v}\bigr), & \text{if } C>2 \text{ and at least two classes present},\\[4pt]
1, & \text{if fewer than two classes present (random baseline).}
\end{cases}
$$

* **Binary case ($C=2$)**: use the **standard** ROC AUC on a **1D score** $s_i := p_v\!\big(x_i,\, \text{positive}\big)$.
* **Multiclass case ($C>2$)**: use **macro-averaged** one-vs-rest ROC AUC (scikit-learn: `multi_class="ovr", average="macro"`).

Let $\delta$ be the pruning threshold.

---

## 1) Node-Quality (Original) Regression Pruning

**Idea.** Treat each node independently: if its own LogitBoost model already clears the discrimination bar $\delta$ on its traffic, collapse its descendants.

**Rule.** For any **non-leaf** node $v$,

$$
\mathrm{AUC}(M_v;S_v)>\delta\ \Longrightarrow\ \text{prune the entire subtree at }v\ (\text{make }v\text{ a leaf holding }M_v).
$$

**Algorithm (BFS).**

1. **Deep-copy** $T$ and $\{M_v\}$ (non-destructive).
2. **Reach sets.** Compute $S_v$ for all nodes (eagerly or lazily).
3. **Traverse (BFS).** For each non-leaf $v$:

   * Compute $A_P:=\mathrm{AUC}(M_v;S_v)$ using the piecewise definition above.
   * If $A_P>\delta$: replace the subtree by a leaf at $v$ (retain $M_v$; drop all descendants).
   * Else: enqueue $L(v)$, $R(v)$ and continue.
4. **Output.** Return the pruned tree and the remaining node models.

**Key property.** Decisions are **local**: pruning depends only on the node’s own LogitBoost model quality on $S_v$.

---

## 2) Local-Gain Regression Pruning (Binary)


**Idea.** Keep a split only if the children’s node models collectively improve AUC over the parent by at least $\delta$.

**AUC at a node.**

* **Binary target ($C=2$)**: standard ROC AUC on a **1D score** (e.g., $p(\text{positive})$).
* **Multiclass target ($C>2$)**: macro-averaged OvR ROC AUC with scikit-learn (`multi_class="ovr", average="macro"`).
  *(In the current implementation, AUC is only computed when the subset has $\ge$ 2 classes and $\ge$ 5 samples; otherwise, the code skips the node via `continue`—it does **not** substitute 0.5.)*

**Local gain at node $v$ with children $L,R$.**

$$
\begin{aligned}
A_P &:= \mathrm{AUC}(M_v;S_v),\\
A_L &:= \mathrm{AUC}(M_L;S_L),\quad A_R := \mathrm{AUC}(M_R;S_R),\\
A_C &:= \frac{n_L A_L + n_R A_R}{n_L+n_R},\qquad
\mathrm{Gain}(v) := A_C - A_P.
\end{aligned}
$$

**Rule.** If $\mathrm{Gain}(v)<\delta$, prune at $v$ (replace the subtree by a leaf that keeps $M_v$).

**Algorithm (BFS)**

1. **Deep-copy** $T$ and $\{M_v\}$; **compute** all reach sets $S_u$.
2. Traverse $T$ in **BFS**. For each **non-leaf** node $v$:

   1. **Parent insufficiency — *defer but descend*.**
      If $|\{y_i:i\in S_v\}|<2$ **or** $n_v<5$:

      * **Enqueue** $(L, v, \text{left})$ and $(R, v, \text{right})$.
      * **`continue`** (no parent AUC computed; no gain test at $v$).
   2. **Compute parent AUC** $A_P$ with the binary/multiclass rule above.
   3. **Child masks missing — *skip and stop here*.**
      If `left_mask is None` **or** `right_mask is None`:

      * **`continue`** (no child AUCs; **children are not enqueued here**, so traversal **does not** go deeper under $v$ in this pass).
   4. **Build child subsets** $(X_L,y_L)$, $(X_R,y_R)$.
   5. **Child insufficiency — *skip and stop here*.**
      If $|\{y\in y_L\}|<2$ **or** $n_L<5$ **or** $|\{y\in y_R\}|<2$ **or** $n_R<5$:

      * **`continue`** (no child AUCs; no gain test; **no enqueue** → the subtree under $v$ remains unchanged).
   6. **Child AUCs and gain.**
      Compute $A_L,A_R$, then $A_C$ and $\mathrm{Gain}(v)$.
   7. **Decision.**

      * If $\mathrm{Gain}(v)<\delta$: **prune** at $v$ (make $v$ a leaf with $M_v$).
      * Else: **enqueue** $L$ and $R$ to continue BFS.

**Important implications of the current behavior (no fallback):**

* **Parent insufficient:** decision at $v$ is **deferred**, but its children **are** explored.
* **Missing child mask:** $v$ is **skipped**, and traversal does **not** descend further below $v$ in this pass.
* **Child insufficient:** $v$ is **skipped**, and traversal does **not** descend further below $v$.
* **No automatic AUC=0.5** is ever used; these cases are simply **not evaluated** (they’re bypassed via `continue`).


---

## 3) Cross-Validation to choose $\delta$

**Goal.** Pick $\delta$ that yields the best **generalization** of the pruned tree, evaluated by a calibration-aware loss on held-out data.

**Inputs.**

* Candidate thresholds $\Delta=\{\delta^{(1)},\dots,\delta^{(m)}\}$, folds $K$, pruning method (Node-Quality or Local-Gain), evaluation metric $\mathcal{L}\in\{\text{log-loss},\ \text{Brier}\}$.

**Procedure.**

1. **K-fold split.** For fold $k$: training/pruning set $S_{\text{train}}^{(k)}$ (union of $K-1$ folds); validation set $S_{\text{val}}^{(k)}$.
2. **Training.** Fit the LMT on $S_{\text{train}}^{(k)}$, producing LogitBoost models $M_v$ at all nodes.
3. **Pruning grid.** For each $\delta\in\Delta$:

   * Prune the trained tree **on $S_{\text{train}}^{(k)}$** using the chosen rule and the **piecewise Node AUC** above.
   * Evaluate on $S_{\text{val}}^{(k)}$: get probabilities $\hat{p}_i$ from the pruned tree; compute $\mathcal{L}^{(k)}(\delta)$ (log-loss or Brier).
4. **Fold best.** $\displaystyle \delta_k \in \underset{\delta\in\Delta}{\arg\min}\ \mathcal{L}^{(k)}(\delta)$.
5. **Final selection.** Mean of fold-bests (as specified):

   $$
   \delta^*=\frac{1}{K}\sum_{k=1}^K \delta_k
   $$


---

## Practical notes and edge cases

* **Explicit AUC policy.** Binary targets → **standard** ROC AUC on 1D scores; multiclass targets → **macro OvR** ROC AUC. If a subset has $<2$ classes, use $1$.
* **Reasonable $\delta$ ranges.** Node-Quality: $[0.5,1)$. Local-Gain: small $\delta\ge 0$ (e.g., $0$–$0.02$).
* **Strict comparisons.** Keep “$>\delta$” (prune) for Node-Quality and “$<\delta$” (prune) for Local-Gain for deterministic ties.
* **No refitting.** Pruning **does not** re-train LogitBoost models; it only removes subtrees.
* **Efficiency.** Precompute reach sets; each AUC is $O(n_v)$–$O(n_v\log n_v)$ depending on implementation.

---

## Compact pseudocode

**Node-Quality pruning**

```
PRUNE_NODE_QUALITY(T, {M_v}, S, δ):
  T' ← deepcopy(T); M' ← deepcopy({M_v})
  compute reach sets {S_v} on T'
  Q ← [root(T')]
  while Q not empty:
    v ← pop_front(Q)
    if v is leaf: continue
    A ← NODE_AUC(M'_v, S_v)   # binary→standard AUC; multiclass→macro OvR
    if A > δ:
      prune_subtree_at(v)     # keep M'_v, drop descendants’ models
    else:
      push(Q, left(v)); push(Q, right(v))
  return (T', M')
```

**Local-Gain pruning**

```
PRUNE_LOCAL_GAIN(T, {M_v}, S, δ, n_min):
  T' ← deepcopy(T); M' ← deepcopy({M_v})
  compute reach sets {S_v} on T'
  Q ← [root(T')]
  while Q not empty:
    v ← pop_front(Q)
    if v is leaf: continue
    A_P ← NODE_AUC(M'_v, S_v)
    L, R ← left(v), right(v)

    # Child sufficiency checks
    A_L ← A_R ← 0.5
    if n_L ≥ n_min and |unique(y[S_L])| ≥ 2: A_L ← NODE_AUC(M'_L, S_L)
    if n_R ≥ n_min and |unique(y[S_R])| ≥ 2: A_R ← NODE_AUC(M'_R, S_R)

    A_C ← (n_L*A_L + n_R*A_R)/(n_L + n_R)
    if (A_C - A_P) < δ:
      prune_subtree_at(v)
    else:
      push(Q, L); push(Q, R)
  return (T', M')
```

