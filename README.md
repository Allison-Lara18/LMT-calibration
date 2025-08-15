# LMT-calibration

Repository for my MITACS summer internship at Université du Québec à Montréal (UQAM), under project **45286 – Combining Decision Trees and Regression Models**.

**Professor**: Dr. Arthur Charpentier  
**Supervisors**: Ewen Gallic and Agathe Fernandes Machado
**Intern**: Allison Lara Nieva

---

## Project Overview

Twenty years ago, Landwehr, Hall & Frank (2005) proposed combining classification trees and regression models on categorical variables (particularly logistic regression) into **Logistic Model Trees (LMTs)** — trees whose leaves contain linear regression functions.

This project revisits that idea from a **calibration** perspective, exploring whether these trees can be improved and combined in ensemble approaches to enhance predictive performance.  

We implement the algorithms (in Python) and evaluate them across various synthetic and real datasets, with a focus on:
- **Reproducing** the original LMT algorithm from Landwehr et al. (2005)
- **Exploring alternative tree-building strategies**
- **Experimenting with regression pruning methods**
- **Assessing calibration quality** (Brier Score, ECE, reliability curves)
- **Developing improved methodologies** (our final MVP experiment)

---

## Repository Structure

Below is a high-level overview of each folder in the repository:

- **`Alternative pruning/`**  
  First tests and comparisons of *regression pruning* across multiple datasets and algorithms.  
  Contains **3 notebooks**: two testing datasets, and one organizing visualizations.

- **`Alternative tree (linear spline)/`**  
  Construction of a **linear spline-based tree** and testing on an absolute value dataset generated with our Data Generation Process (DGP).  
  Includes additional regression pruning tests and variations.  
  Contains **3 notebooks**:  
    1. Tree construction process  
    2. Testing more metrics during tree construction and pruning (**final winning methodology**)  
    3. Analysis of how standard performance metrics behave with different regression pruning thresholds.

- **`Alternative tree (root multi-cuts)/`**  
  Experiment with poor performance results, aiming for a tree with **multiple cuts only at the root**, then growing traditional subtrees for each root child.  
  Contains **2 notebooks**: one for tree construction, and one for visualizations.

- **`Calibration testing/`**  
  Analysis with synthetic datasets generated via DGP to examine **calibration** in the original LMT implementation.  
  Contains **3 notebooks**: two tests and one visualization notebook.

- **`Construction and tests/`**  
  Development of the **original LMT implementation** based on Landwehr et al. (2005) and tests on various datasets.  
  Contains **7 notebooks**: two for construction, the rest for dataset testing.

- **`Documentation/`**  
  Markdown documents with detailed descriptions of each folder and notebook, as well as theoretical background.

- **`Papers/`**  
  Reference papers used in the project.

- **`lmt_pkg/`**  
  Python package containing **all implementations** mentioned above in `.py` files for easy reuse.  
  Includes functions for:
    - Tree construction (original and experimental variants)  
    - Regression pruning methods  
    - Visualization utilities  

---

## Getting Started

### Requirements
- Python 3.10+

### Installation
```bash
git clone https://github.com/Allison-Lara18/LMT-calibration.git
cd LMT-calibration
```

---

## Final MVP Experiment
Our final and best-performing methodology is documented in detail inside the Alternative tree (linear spline) folder, in the notebook testing extended metrics during construction and regression pruning. This serves as the Minimum Viable Product (MVP) for the project.

---

## References
 - Landwehr, N., Hall, M., & Frank, E. (2005). Logistic model trees. *Machine Learning, 59*(1–2), 161–205. [https://doi.org/10.1007/s10994-005-0466-3](https://doi.org/10.1007/s10994-005-0466-3)

 - Gruber, S. G., & Buettner, F. (2022). Better uncertainty calibration via proper scores for classification and beyond. In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 627–640). Curran Associates, Inc. 

 - Machado, A. F., Charpentier, A., & Fernandes, F. (2024). From uncertainty to precision: Enhancing binary classifier performance through calibration.

 - Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: A statistical view of boosting. *The Annals of Statistics, 28*(2), 337–407. [https://doi.org/10.1214/aos/1016218223](https://doi.org/10.1214/aos/1016218223)

 - Breiman, L., Friedman, J. H., Olshen, R., & Stone, C. J. (1984). *Classification and regression trees*. Wadsworth & Brooks/Cole Advanced Books & Software.
