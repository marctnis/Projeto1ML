# Bank Marketing Campaign: Conversion Prediction & EDA

## 📌 Project Overview
An exploratory data analysis (EDA) and data preparation project to predict term deposit subscriptions using the UCI Bank Marketing dataset. The goal is to clean, analyze, and prepare the data to build robust Machine Learning classification models.

## Dataset
* **Source:** [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Target Variable:** `y` (Has the client subscribed a term deposit? 'yes' or 'no')

## Technologies & Libraries
* **Language:** Python
* **Environment:** Jupyter Notebook / Google Colab
* **Libraries:** `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn`, `ucimlrepo`

## Key Steps & Insights
1. **Data Collection:** Direct integration with the UCI API using `ucimlrepo`.
2. **Exploratory Data Analysis (EDA):**
   * Identification of missing values (e.g., high null rate in the `poutcome` column) and duplicated records.
   * Target class imbalance analysis (approx. 88.3% 'no' vs. 11.7% 'yes').
   * Outlier detection using boxplots (focusing on `balance` and `duration`).
3. **Correlation & Feature Engineering:**
   * Spearman correlation heatmap to identify relationships between variables.
   * Investigated the relationship between `pdays` and `previous`.
   * Replaced non-logical placeholder values (e.g., `-1` in `pdays`) with `NaN` to prevent model bias.
4. **Data Visualization:** Density plots (KDE) to evaluate numerical distributions grouped by the target variable.

## How to Run
1. Clone this repository or download the `.ipynb` file.
2. Open the notebook in **Google Colab** or a local **Jupyter** environment.
3. Ensure you have the required libraries installed (`pip install ucimlrepo pandas seaborn scikit-learn imbalanced-learn`).
4. Run the cells sequentially.

## Next Steps
* Apply data transformations (e.g., Yeo-Johnson / Signed Log) to handle outliers.
* Implement class balancing techniques (such as SMOTE).
* Train and evaluate classification models (Random Forest, XGBoost, etc.).
