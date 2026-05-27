# CTR Prediction and Probability Calibration for Advertising Auction

<img width="550" height="318" alt="image" src="https://github.com/user-attachments/assets/6a61b8c0-ef68-4f86-ae2a-5e3fe428a612" />


## Project Description

This project focuses on building a binary classification model to predict the Click-Through Rate (CTR) for advertising banners within a Real-Time Bidding (RTB) advertising auction for **Advandex**. The key objective is not only accurate CTR prediction but also the **calibration** of predicted probabilities to align as closely as possible with actual click frequencies. This is critically important because the auction winner is determined by the formula `Bid × Predicted CTR`.

## Data

*   **Source:** `ad_click_dataset` — an analytical data mart of advertising banner impressions.
*   **Target Variable:** `click` (1 — click, 0 — no click).
*   **Characteristics:** Class imbalance is present. Data is anonymized, with categorical values represented by hashes (e.g., `f028772b`).

## Evaluation Metrics

The following metrics are used to evaluate model quality:

*   **PR-AUC:** The primary metric, accounting for class imbalance and focusing on clicks.
*   **Log Loss:** Evaluates the quality of predicted probabilities, which is crucial for bid calculation.
*   **Brier Score:** Measures the accuracy and calibration of predictions, with direct financial implications.

## Project Structure

The project is organized as a Jupyter Notebook, which includes the following main stages:

1.  **Environment Setup and Data Loading:**
    *   Installation and import of necessary libraries.
    *   Configuration of display parameters for plots and dataframes.
    *   Setting the `RANDOM_SEED` constant for reproducibility.
    *   Loading data from a CSV file and initial inspection.

2.  **Exploratory Data Analysis (EDA):**
    *   Basic dataset information (number of observations, features, data types).
    *   Analysis of target variable distribution and class imbalance.
    *   Feature analysis: determining usefulness, types (categorical/numerical), initial selection.
    *   Missing values analysis and selection of imputation strategy.
    *   Categorical feature analysis and their cardinality.
    *   Outlier and numerical feature distribution analysis.
    *   Correlation analysis of features with the target variable and among themselves.
    *   EDA conclusions and definition of preprocessing steps.

3.  **Data Preprocessing:**
    *   Application of selected strategies for handling missing values.
    *   Encoding categorical features (One-Hot Encoding, Target Encoding).
    *   Scaling numerical features.

4.  **Feature Selection:**
    *   Using feature selection methods to improve model performance and reduce dimensionality.

5.  **Model Training:**
    *   Training various binary classification models (e.g., Logistic Regression, LinearSVC, LightGBM).
    *   Hyperparameter tuning using cross-validation.

6.  **Probability Calibration:**
    *   Applying calibration methods (e.g., `CalibratedClassifierCV`) to ensure predicted probabilities align with actual ones.

7.  **Model Evaluation:**
    *   Evaluating model performance using PR-AUC, Log Loss, and Brier Score.
    *   Comparison of calibrated and uncalibrated models.

## Technologies and Libraries Used

*   **Data Processing:** `pandas`, `numpy`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Machine Learning:** `scikit-learn` (including `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `TargetEncoder`, `LogisticRegression`, `LinearSVC`, `CalibratedClassifierCV`, `GridSearchCV`, `cross_val_score`, `average_precision_score`, `log_loss`, `brier_score_loss`, etc.), `lightgbm`
*   **Correlation Analysis:** `phik`
*   **Model Saving:** `joblib`

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone <repository_URL>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/macOS
    # .\venv\Scripts\activate  # For Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Note: the `requirements.txt` file will be created based on the imports in the notebook)
4.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open the `cleaned_notebook_en.ipynb` file in your browser.
