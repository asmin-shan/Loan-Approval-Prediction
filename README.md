# Loan-Approval-Prediction-ML

---

##  Project Overview

This project focuses on building and evaluating machine learning models to analyze financial risk and predict loan outcomes. Using the **Financial Risk for Loan Approval dataset** from **Kaggle**, the workflow addresses two critical financial modeling objectives:

1.  **Risk Score Regression:** Predicting a continuous **RiskScore** associated with an individual's likelihood of loan default or financial instability.
2.  **Binary Classification:** Determining the final loan outcome, predicting whether an applicant is likely to be **Approved (1)** or **Denied (0)**.

The final model selected for classification, a tuned **GradientBoostingClassifier**, demonstrated the highest overall performance in predicting loan approval.

##  Dataset

The data is sourced from the **Kaggle: Financial Risk for Loan Approval** dataset.

* **Source:** [Kaggle: Financial Risk for Loan Approval](https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv)
* **Key Target Variables:**
    * `RiskScore` (Continuous for Regression)
    * `LoanApproved` (Binary for Classification)

The dataset includes various features such as `AnnualIncome`, `CreditScore`, `LoanAmount`, `DebtToIncomeRatio`, and `PaymentHistory`.

##  Technical Stack

The project is built using Python and the standard data science ecosystem:

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Modeling:** `scikit-learn` (specifically **`GradientBoostingClassifier`** and **`GradientBoostingRegressor`**)
* **Visualization:** `matplotlib.pyplot`, `seaborn`
* **Notebook:** Jupyter Notebook (`loanproject.ipynb`)

##  Key Features and Methodology

* **Data Preprocessing:** Handled missing values, engineered a key feature (`BankBalance`), and addressed **multicollinearity** by excluding highly correlated features like `TotalDebtToIncomeRatio`.
* **Feature Scaling:** Applied **`StandardScaler`** to normalize numerical features for optimal model performance.
* **Risk Score Model:** Utilized the **`GradientBoostingRegressor`** for predicting the continuous `RiskScore`.
* **Approval Classification:** Employed a final, highly-tuned **`GradientBoostingClassifier` (learning_rate=0.45)** as the best model for predicting `LoanApproved`.
* **Model Persistence:** Both the regression and classification models are saved as pickle files for easy deployment and future use.

##  Repository Structure
