# Calibration
Binary-classifier outputs are often interpreted as probabilities. In sensitive domains (credit-risk, healthcare, insurance) we need those probabilities to be calibrated, i.e., the predicted 0-to-1 score should match the observed event frequency.  
Classic evaluation metrics (accuracy, AUC, F1) ignore that requirement because a model can rank well and still output over or under confident scores.

---

### 1  What is calibration?
A classifier is *perfectly calibrated* if  
  $\Pr\big(D=1\mid \hat{s}(x)=p\big)=p\quad\forall\,p\in[0,1]$  
  where $D$ is the binary outcome and $\hat{s}(x)$ is the model raw score for instance $x$.

This means that, whenever your model says "probability = p", roughly 30% of those cases should be positives in reality. Because $\hat{s}(x)=p$ is a measure-zero event, calibration is studied in bins (quantile groups) or with smoothing (local regressions).

---

### 2  How mis-calibration creeps in
* **Optimising the wrong objective:** Training for accuracy, log-loss, or AUC cares about ordering of scores, not the numeric scale, so the model may end up over or under confident.
* **Class imbalance:** With very few positives the learner plays it safe and parks most scores near 0 (or near 1 if positives dominate).
* **Regularisation / early stopping:** Penalties that shrink coefficients (or unfinished training that leaves weights small) compress logits, whereas lack of regularisation can inflate them, yielding systematic bias.
* **Dataset shift:** Prevalence or feature distribution changes between training and production.
* **Post-processing thresholds:** Manually tweaking the decision cut-off to hit a precision/recall target distorts the mapping between raw score and true event rate, so probabilities away from that threshold become inaccurate.

---

### 3  Metrics

#### Quantile-based measures

* **Calibration curve**

  * Basically, it shows if the model's “p = 0.30” really mean a 30 % chance.
  * How to build it:
    1. Split the scores into $B$ equal-size quantile bins $\mathcal B_1,\dots,\mathcal B_B$.
    2. For each bin $b$

       * Confidence  
         $\text{conf}(b)=\frac{1}{n_b}\sum_{i\in\mathcal B_b}\hat s(x_i)$
       * Accuracy  
         $\text{acc}(b)=\frac{1}{n_b}\sum_{i\in\mathcal B_b}\mathbf 1_{d_i=1}$  
    3. Plot the points $(\text{conf}(b),\text{acc}(b))$.
  * Read-out:
    * On the 45° line ⇒ good.
    * Above the line ⇒ model too low (under-confident).
    * Below the line ⇒ model too high (over-confident).

* **Expected Calibration Error (ECE)**

  * Formula  
    $\text{ECE}= \sum_{b=1}^{B}\frac{n_b}{n}\, \bigl|\text{acc}(b)-\text{conf}(b)\bigr|$  

    $n_b$=points in bin, $n$=total points.
  * Meaning: Average absolute gap between what the model predicts and what actually happens.
  * Caveat: The value changes if you change the number of bins $B$.


#### Local-regression measures

* **Smoothed calibration curve**

  * Instead of bins, fit a smooth line through the points $(\hat s(x),D)$ with LOESS or another local regression. No stair-step noise, user controls smoothness with the bandwidth.
  * How to build it:
    1. Pick a bandwidth (how wide the local neighbourhood is).
    2. Compute the smoothed value $\hat g(p)$ for many $p$ in $[0,1]$.
    3. Plot $(p,\hat g(p))$.

* **Local Calibration Score (LCS)**

  * Formula  
    $\text{LCS}= \sum_{i=1}^{N} w_i\bigl[\hat g(l_i)-l_i\bigr]^2$  

    $l_i$=grid points, $\hat g(l_i)$=smoothed curve, $w_i$=how common scores are near $l_i$.
  * Meaning: Squared distance between the smooth curve and the perfect 45° line, weighted by how often the model predicts each score. No bins, gives both a curve and a single number.


| Family               | Visual tool                | Numeric score | Parameter to tune                |
| -------------------- | -------------------------- | ------------- | -------------------------------- |
| **Quantile-based**   | Calibration curve (bins)   | ECE           | Number of bins $B$               |
| **Local-regression** | Smoothed calibration curve | LCS           | Bandwidth / k-nearest-neighbours |

Both families compare "model says" vs. "reality", but they differ in **how** they pool the data along the score axis.  
It is ideally to track one discrimination metric (e.g., AUC) plus one calibration metric.

#### Another metrics to be taken into account
  * **Brier Score**
    * Formula  
      $\text{BS}(f) = \mathbb(E)[||f(X) - Y'||_2^2]$  
      * $Y'$ is one-hot encoded $Y$.
    * Meaning: The estimator of the BS is equivalent to the mean squared error (MSE), illustrating that it does not purely capture model calibration. Can be interpreted as a comprehensive measure of model performance, simultaneously capturing model fit and calibration. If it is equals to 0, $f$ is perfectly calibrated according to this metric.

  * **Kolmogorov-Smirnov calibration error (KS)**
  * **Maximum mean calibration error (MMCE)**
  * **Kernel calibration error (KCE)**

---

### 4  Visualisations

1. **Reliability diagram (binned curve)**
   *X-axis* = average predicted probability per bin.
   *Y-axis* = actual fraction of positives.
   The 45° line is perfection.

2. **Smoothed calibration curve**
   Replace bins with a LOESS or local-polynomial fit → cleaner shape, less stair-step noise.

3. **Score histogram under the curve**
   Shows where the model actually predicts. A huge gap at p=0.9 is irrelevant if you never predict 0.9.

---

### 5  Re-calibration methods (post-hoc fixes)

| Method                               | Formula (binary classification)                | Tends to work when…                                       |
| ------------------------------------ | ---------------------------------------------- | --------------------------------------------------------- |
| **Platt scaling**                    | $\sigma(a\hat s + b)$                          | You have hundreds-thousands of points; need monotone fix. |
| **Isotonic regression**              | Monotone step function                         | Data-rich (>10 k) and you want maximum flexibility.       |
| **Beta calibration**                 | $\text{BetaCDF}(\hat s; a,b)$                  | Slightly more flexible than Platt, still monotone.        |
| **Local regression**                 | Smooth non-parametric curve                    | Need gentle, possibly non-monotone adjustment.            |

**Workflow:**
1. Split a calibration set not used for training.
2. Fit the mapping on that slice.
3. Apply the mapping to future predictions.

---

### 6  Multiclass classifiers

*Definition* generalises:  
$\Pr(Y=c \mid \hat{\mathbf p}(x)=\mathbf p) = p_c \quad \text{for every class } c.$  

**Practical tools**

* **One-vs-rest ECE** – compute ECE per class and average.
* **Softmax-ECE** – bin by the top predicted class & its probability.
* **Dirichlet / Matrix / Vector scaling** – parametric maps extending Platt/Temperature to the whole probability vector.
* **Reliability diagrams per class** – easiest to read.

---

### 7  Limits to keep in mind

* **Data hunger:** Reliable estimates need hundreds of positives per bin. In small datasets confidence intervals are wide, it is recommended to plot them.
* **Tails are untestable:** If the model never predicts > 0.97, you cannot say if that region is calibrated.
* **Changing prevalence:** A perfectly calibrated model at a 10 % base rate is not calibrated when base rate jumps to 30 % unless covariate shift assumptions hold.
* **Fairness trade-offs:** You can’t usually make a model simultaneously calibrated within each subgroup and equal in other fairness metrics.
* **Calibration ≠ usefulness:** A model that always predicts the base rate (e.g., 0.22) is perfectly calibrated but useless for ranking. Keep sharpness/discrimination in sight.

---

**Are my numeric scores believable probabilities?**.
