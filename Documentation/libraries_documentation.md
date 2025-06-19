# LMT original algorithm
The idea is to start with one simple logistic model, let LogitBoost add terms bit-by-bit, pause to see whether a C4.5 split would help, copy the model down each new branch, repeat the boost-or-split cycle, and finally prune the over-grown tree. In other words, at every node it first improves the local logistic model, then asks whether a split would explain the data better, and finally trims the finished tree so it is not too big.  
**Step by step**
  1. Fit a boosted logistic model on all rows with the LogitBoost procedure, adding one weak learner at a time. A five-fold cross-validation decides when to stop adding learners so the model does not overfit.
  2. Test every possible split, for each candidate variable (and each cut-point, if the variable is numeric) temporarily divide the data into two parts. In each part, warm-start the logistic model with the parent's coefficients, run more LogitBoost iterations, and measure how much the total log-likelihood improves. This gain is the information gain for that split.
  3. Keep the split with the largest gain, if the best gain is larger than a small threshold (the stopping rule), make the split and create two child nodes. Otherwise, stop and keep the current node as a leaf.
  4. Repeat inside every child node. Go back to Step 1 with the subset of rows in that child. The parent's boosted model serves as a starting point, so fewer boosting iterations are usually needed lower in the tree.
  5. Prune the completed tree. After no more useful splits can be found, perform cost-complexity pruning (borrowed from the CART method).
  6. Return the final tree where each terminal node holds its own boosted logistic regression. Predictions for new data follow the path down the tree and use the probabilities from that leaf.

---

# R implementations
## `glmtree` CRAN package
Logistic regression tree by **Stochastic-Expectation-Maximization**, i.e., it is not the classic algorithm proposed by Landwehr et al. Is an entire package that builds logistic regression trees with a stochastic-EM search strategy inspired by Ehrhardt (2019). It imports `partykit` only for representing and plotting the trees.  
The idea is to shuffle the data into several provisional groups, fit a logistic regression in each group, reshuffle based on the fits, reshape the tree if that helps, score the result, and repeat many times, finally keeping the best-scoring tree.  
**Step by step**
  1. Choose a number *K* of initial groups (default 10) and randomly label every row with one of these groups.
  2. Fit a logistic regression inside every group.
  3. Re-assign each row to the group where its data are now most likely under the current fits (this is the expectation step of the expectation–maximization idea).
  4. With the new labels, re-draw the actual tree shape using information-gain style splits, so the tree matches the labels better.
  5. Compute a score such as the Akaike information criterion, the Bayesian information criterion, or the Gini statistic. If this score beats all previous scores, remember the tree.
  6. Repeat steps 2-5 for the required number of iterations (default 200).
  7. Return the stored best tree together with its score.

Key takeaway: This method is *stochastic and exploratory*: it tries many different trees, keeps track of the best one found, and you control how long it searches. Wanders through many possible trees and keeps the one that scores best on a chosen information or accuracy measure. Good when pure predictive power is the main goal and you can afford a longer search.



## `glmtree` partykit function
Is a single function inside the partykit toolkit that builds generalised linear-model trees with the model-based recursive partitioning (MOB) algorithm.  
The idea is to start with the whole data set, look for the first place where the current logistic regression stops fitting well, cut there, and keep doing that inside every new piece.  
**Step by step**
   1. Fit one logistic regression to all rows.
   2. For every candidate splitting variable, ask: *Do the regression coefficients change when I look at different values of this variable?*  This is checked with a statistical test; the smaller the probability value, the stronger the change.
   3. Pick the variable that shows the strongest change, and find the cut-point that improves the log-likelihood the most.
   4. Slice the data at that point, giving two child nodes.
   5. Inside each child, repeat steps 1-4 until 
      * The test no longer finds a meaningful change.
      * A minimum leaf size is reached.
      * The maximum depth is reached.
   6. Return a tree object where each end-node stores its own logistic model, and the usual `predict()` and `plot()` methods work.

Key takeaway: This method is *deterministic and sequential*: it tests one variable at a time, always keeps the best confirmed split, and stops when no further significant change can be detected. Builds the tree in a single pass, always choosing the most statistically evident split at every step. Good for interpretable splits and quick results.


---

# Python implementations
## `lrtree` library
Implements the `glmtree` algorithm from CRAN (Adrien Ehrhardt, 2024). Leaves are classical logistic regressions andthe tree itself is found with a stochastic EM search, not LogitBoost, but it behaves like an LMT in practice.

## `linear-tree` library
Generic model-tree framework where you plug any linear model into the leaves. And using `LogisticRegression()` as the base estimator it can be obtained a LMT-like classifier. The idea is to turn a normal decision tree into one whose leaves hold their own logistic regressions. And due to everything is written in Python, cross-validate and pipeline it exactly like any other scikit-learn model.  
**Step by step**
  1. Pick a base learner, i.e., any scikit-learn logistic model, in this case `LogisticRegression()`, as base_estimator.
  2. Grow the tree (greedy CART-style) For each candidate split the library:
     * Fits one logistic model to the would-be left child and one to the right child;
     * Evaluates how much the split lowers the overall log-loss (or Gini) compared with keeping the parent node;
     * Keeps the split with the best gain. This repeats depth-first until a stopping rule such as min_samples_leaf, max_depth or min_impurity_decrease is met. 
  3. Predict: New rows follow the decision rules; the leaf's logistic model outputs calibrated probabilities with predict_proba.
  4. Interpret: Use export_text or plot_tree for the rules, and leaf.estimator_.coef_ to see each local logistic equation.

## `python-weka-wrapper` library
Is the canonical Logistic-Model Tree written by the article authors (Landwehr et al) in Java (LogitBoost + C4.5 + pruning) straight from Python by spinning up a Java Virtual Machine in the background.

---

# Original implementation
Available in Java programming language.