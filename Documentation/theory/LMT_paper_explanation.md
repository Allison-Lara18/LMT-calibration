# Logistic Model Trees
**Landwehr, N., Hall, M., & Frank, E. (2005).**

The paper proposes **Logistic Model Trees (LMT)** – a hybrid that merges the low-bias flexibility of decision-tree structures with the probabilistic outputs of **multinomial logistic regression**.  Instead of linear-regression models at the leaves (as in Quinlan's M5), LMT places a logistic regression model in each leaf, giving class probabilities and a single interpretable tree instead of one tree per class.

---

### 1  Introduction

* **Motivation:**  Decision trees fit non-linear boundaries but can overfit (high variance, low bias); logistic regression is stable but biased toward linear separations. No single method wins everywhere.
* **Idea:**  Let the learner start as a simple logistic model when data are scarce/noisy and grow tree structure only where the data justify it.
* **Promise:**  The authors preview that LMT beats C4.5, CART, plain logistic regression, functional trees, model trees, NBTree, LOTUS, and is competitive with boosted trees, across 36 UCI sets .

---

### 2  Tree Induction and Logistic Regression

#### 2.1 Tree induction

Explains classic top-down splitting, where this subdivision is made recursively splitting the instance space, stopping when the regions of the subdivision are reasonably ‘pure’ or meet the minimum size criteria. A tree can be seen as a set of rules that say how to classify instances, but they tend to overfit (high variance, low bias)

#### 2.2 Classification via regression

Shows the “one-vs-all indicator” method, that is given by the following structure:  
The idea is to recode a *J*-class problem into *J* separate binary questions: for each class you create a binary indicator column that is 1 when the instance belongs to that class and 0 otherwise.  You then fit an independent regression model to each column.  At prediction time every model produces a numerical score, and you simply pick the class whose model gives the largest score.  
Although this works mechanically, it has two big flaws.  First, the scores are real numbers, so they can’t be read as probabilities. Second, with more than two classes the trick can fail because two of the regressors may overlap in a way that "masks" the third class.


#### 2.3 Logistic regression

The paper switches to a model that directly predicts posterior probabilities.  For a J-class problem it defines J – 1 log-odds against a chosen base class, giving probabilities that are guaranteed to add to one and stay between zero and one.

Because maximum-likelihood estimates of the $β$-coefficients have no algebraic solution and it is necessary to use a numerical method, the authors adopt **LogitBoost**. This  is a boosting algorithm designed to fit logistic-regression models by adding one weak learner at a time instead of solving all coefficients in one shot. It minimises the negative log-likelihood.


*2.3.1 Attribute selection.*  
LogitBoost can fit a whole logistic model in one go, including every predictor, but the authors deliberately slow the process down so that it adds only one attribute at a time. In each boosting round they fit a simple least-squares regression that involves just a single candidate attribute, pick the one that minimises squared error, and append that term to the model. Because any full (multiple) logistic regression can be written as a sum of these simple terms, running LogitBoost to full convergence would eventually rebuild the same complete model; the trick is to stop early—with the number of rounds chosen by 5-fold cross-validation—so only the attributes that genuinely improve validation accuracy ever make it in. This is called **SimpleLogistic**.

*2.3.2 Nominal attributes & missing values.*  
Nominals are binarised (one-hot encoding). Missing values are imputed once with mean for numerical values and mode for categorical values before any tree building.

---

### 3  Related Tree-Based Learners

A concise explaination of all the algorithms that LMT is going to be compared with:

* **Model trees (M5'):** Grows a regression tree, then replaces each node with a linear regression model and optionally “smooths” predictions along the root-to-leaf path; it delivers piece-wise linear fits that often beat constant-leaf regression trees.
* **Stepwise Model Tree Induction (SMOTI):** Incremental simple-regression leaves. Interleaves simple one-attribute regressions with splits, incrementally assembling the final multi-linear model as you move down the path—capturing global effects while keeping every fit cheap and reducing sharp jumps between neighbouring leaves .
* **Lotus:** Builds binary logistic trees, chooses split variables via a modified χ² test to avoid bias, fits (simple or multiple) logistic models at nodes, and prunes with CART-style cost-complexity based on deviance rather than mis-classification error
* **Functional trees (LTree):** Can insert constructed attributes with linear discriminants or logistic outputs, both as oblique split tests and as leaf predictors, then prunes each node among “keep subtree / constant leaf / functional leaf” options with C4.5’s error estimate lmt
* **NBTree:** Decides at every node whether to split or to stop and train a local Naive Bayes; the choice is made by cross-validated accuracy, yielding trees that often outperform either standalone Naive Bayes or plain decision trees.
* **Boosting trees:** AdaBoost.M1 repeatedly re-weights data and grows many weak C4.5 trees, then combines them by a weighted vote; the committee is highly accurate but costs more training time and loses the single-tree interpretability advantage.

---

### 4  Logistic Model Trees

#### 4.1 The model

In an ordinary decision tree each inner node asks a question about one attribute and sends the instance down a branch, ergo, each leaf predicts a class. LMT keeps this tree skeleton but replaces every leaf’s single class label with a **local multinomial logistic-regression model**. Formally, the tree's splits carve the instance space into disjoint regions $S_t$; inside a region the leaf model $f_t(x)$ computes class probabilities with a linear log-odds formula that may use only a subset of the original attributes, chosen by the attribute-selection routine described earlier.

Because both extremes are included, LMT can adapt its complexity:

* If the best bias-variance trade-off is a single global model, the tree is pruned back to the root and you just get standard logistic regression.
* If the data demand more nuance, the algorithm keeps some splits, fitting progressively refined logistic models on the smaller subsets beneath them.
* A plain decision tree is the opposite limiting case where every leaf’s attribute set is empty (i.e., the leaf just holds a majority class).


#### 4.2 Building LMTs

Key algorithm steps:

1. **Root model.**  Fit a SimpleLogistic model with LogitBoost + 5-fold CV.
2. **Split test.**  Evaluate candidate splits with the regular C4.5 information-gain measure (they tried LogitBoost-response-based splits; gains were negligible).
3. **Child models.**  Resume LogitBoost in each child, starting from the parent’s coefficients and CV-choosing extra iterations. This incremental refinement reuses global effects and only adds local terms.
4. **Stopping.**  Don’t split if fewer than 15 instances or no gain, or if too few cases to cross-validate.
5. **Pruning.**  Use CART's cost-complexity pruning with 10-fold CV to trade training error against (tree size × penalty) .
6. **Missing/nominal handling.**  Same global imputation; nominal-to-binary (one-hot encoding) done locally at each node.

#### 4.3 Complexity & Speed-up heuristics

Full LogitBoost + nested CV is slow.  Two heuristics help:

* **One-time CV:** determine the best #iterations at the root and reuse it below.
* **Early-stop LogitBoost:** if validation error hasn’t improved for 50 rounds, quit.
  These give big speed-ups without hurting accuracy .

---

### 5  Experiments

* **Datasets.**  36 diverse UCI sets (57–20 000 rows) with stratified 10×10-CV and a corrected resampled $t$-test .
* **Competitors.**  C4.5, CART, Simple & full logistic, NBTree, Lotus, LTree, M5’(classification), AdaBoost(10/100) + C4.5, etc.
* **Key findings.**

  * LMT **never loses significantly** to any base learner; often wins (e.g., 16 datasets vs C4.5) .
  * Trees are *far* smaller than C4.5 or CART – sometimes pruned to a single root model.
  * Outperforms or matches NBTree, LTree and Lotus on most sets .
  * Competitive with boosted trees: wins on some, loses on others, but with a simpler single-tree explanation.
  * Variable-selection SimpleLogistic beats full logistic in accuracy on many sets.

---

### 6  Conclusions & Future Work

* **Strengths.**  Accurate, compact, interpretable single model; adapts between pure logistic and deep tree.
* **Limitations.**  Computationally heavy; relies on simple global imputation; heuristic stopping rules.
* **Future work.**  Faster logistic fitting, smarter missing-value treatment, more formal parameter-selection methods .

---

## Summary

| Concept                          | Why it matters                                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **LogitBoost-based leaf models** | Gives calibrated probabilities and built-in feature selection.                                            |
| **Incremental refinement**       | Reuses higher-level coefficients, so leaves learn mainly *local* corrections.                             |
| **CART cost-complexity pruning** | Lets the tree collapse to a single logistic model when that’s best.                                       |
| **Empirical win profile**        | Beats single trees and logistic regression; trades roughly evenly with boosted trees while being simpler. |
| **Speed vs. accuracy trade-off** | Nested CV and many LogitBoost rounds cost time; heuristics mitigate this without hurting performance.     |