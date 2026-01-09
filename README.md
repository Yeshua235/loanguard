
# LoanGuard — Credit Default Risk Modeling

> **Tagline**: Guarding lenders against risky loans with cost‑aware, class‑imbalance‑robust modeling.


LoanGuard is a decision‑aware machine learning pipeline that predicts default probability and converts it into business‑aligned approvals using a cost matrix and threshold optimization. The project tackles severe class imbalance, reports PR‑AUC, recall@fixed precision, expected cost, and calibrated confusion matrices, and includes notebooks for EDA, feature engineering, training, and evaluation.



## Overview

**Goal**
Predict the probability that a loan applicant will **default**, then convert probabilities into **decisions** using a **cost matrix** and **threshold optimization** that reflects business risk tolerance.

**Why it matters**
In consumer lending, defaults are **rare** and the cost of mistakes is **asymmetric**. A model that is merely “accurate” can be **unprofitable** if it ignores the cost of false negatives (missed risky loans). This project demonstrates **decision‑aware modeling** using robust pipelines, interpretable baselines, and **precision‑recall** driven evaluation.



## Dataset

**Source**

*   **Kaggle** → *Home Credit Default Risk*
    Only **`application_train.csv`** is used (~158MB).

    Kaggle competition page: <https://www.kaggle.com/c/home-credit-default-risk>

**Target**: `TARGET` (1 = default, 0 = non‑default)
**Features**: Mixed numerical & categorical columns (demographics, employment, income, credit amounts, flags).
**Imbalance**: Defaults ≈ **8%**, so accuracy alone is misleading.



## Project Structure

    loanguard/
    ├─ README.md ✅
    ├─ LICENSE ✅
    ├─ data/
    │  └─ application_train.csv ✅
    ├─ utility/
    │  └─ utils.py ✅
    ├─ notebooks/
    │  ├─ eda.ipynb ✅
    │  ├─ feature_engineering.ipynb ✅
    │  ├─ model_selection.ipynb ✅
    │  └─ loanguard.ipynb
    ├─ reports/
    │  ├─eda/
    │  │  └─target_classes.dist.png ✅
    │  ├─feature_engineering/
    │  │  ├─distance-preservation.png ✅
    │  │  ├─Frobenius-vs-dimensionality.png ✅
    │  │  ├─MSE-vs-dimensionality.png ✅
    │  │  ├─pca-cumulative-explained-variance.png ✅
    │  │  ├─R2-vs-dimensionality.png ✅
    │  │  ├─reconstruction-error-r2.png ✅
    │  │  └─Spectral-vs-dimensionality.png ✅
    │  ├─model_selection/
    │  │  ├─learning_curves/
    │  │  │  ├─dummy_classifier.png ✅
    │  │  │  ├─hist_gradient_boosting_classifier.png ✅
    │  │  │  ├─logistic_regression.png ✅
    │  │  │  ├─random_forest.png ✅
    │  │  │  └─stochastic_gradient_descent.png ✅
    │  │  ├─full-cv-f1-scores.png ✅
    │  │  ├─full-cv-pr-auc.png ✅
    │  │  ├─reduced-cv-f1-scores.png ✅
    │  │  ├─reduced-cv-pr-auc.png ✅
    │  │  └─selected_algorithm_pr_curve.png ✅
    │  ├─ pr_curves.png
    │  ├─ confusion_matrix.png
    │  └─ feature_importance.png
    ├─ models/
    │  ├─
    │  └─ loanguard.pkl
    ├─ environment.yml ✅
    ├─ requirements.txt ✅
    └─ .gitignore ✅



## Reproducibility

- Set `random_state=37` where applicable.
- Save artifacts under `models/` and `reports/`.


## Results

- PR AUC: ---
- Recall@Precision= --- : ---
- Optimal threshold: ---
- Expected cost per 1,000 applications: --- to ---

See `reports/` for exact figures and plots.



## Metrics

*   **Precision‑Recall Curve**
*   **PR AUC**
*   **Recall\@Fixed Precision**
*   **Expected Cost**
*   **Confusion Matrix at Optimal Threshold**



## Getting Started

### Prerequisites

- Python 3.11+
- conda (preferred) or pip
- Required Python packages (see `environment.yml` or `requirements.txt`)

### Download dataset

1. Ensure the Kaggle CLI is configured (Windows):
   - Place your `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.
   - Make sure the file is readable by your user.
   - Install the CLI: `pip install kaggle`

2. Download and extract:

```bash
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data
unzip data/application_train.csv.zip -d data
```

Windows PowerShell alternative to unzip:

```bash
powershell -Command "Expand-Archive -Path data/application_train.csv.zip -DestinationPath data -Force"
```


### Installation & Usage

#### Using Conda

1. **Clone the repository:**
	```sh
	git clone https://github.com/Yeshua235/loanguard.git
	cd loanguard
	```

2. **Install dependencies:**
	```sh
	conda env create -f environment.yml
	```

3. **Verify the dataset is present:**
   - Ensure `data/application_train.csv` exists.

4. **Run the notebook:**
   - Open `notebooks/loanguard.ipynb` in VS Code or Jupyter and run all cells.
   - Artifacts (e.g., `loanguard.pkl`, PR curves) will be saved under `reports/` and `models/` directory.


#### Using pip

If you prefer to use pip instead of conda, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Yeshua235/loanguard.git
    cd loanguard
    ```

2. **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    .\venv\Scripts\activate
    ```
    *(On macOS/Linux, use `source venv/bin/activate`)*

3. **Install dependencies with pip:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Verify the dataset is present:**
   - Ensure `data/application_train.csv` exists.

5. **Run the notebook:**
   - Open `notebooks/loanguard.ipynb` in VS Code or Jupyter and run all cells.
   - Artifacts (e.g., `loanguard.pkl`, PR curves) will be saved under `reports/` and `models/` directory.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



## Acknowledgments

- **Dataset**: *Home Credit Default Risk* (`application_train.csv`) — Kaggle competition.
- **Libraries**: scikit‑learn, pandas, numpy, matplotlib/seaborn.
