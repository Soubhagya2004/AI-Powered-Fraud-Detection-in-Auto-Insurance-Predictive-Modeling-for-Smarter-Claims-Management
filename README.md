# AI-Powered-Fraud-Detection-in-Auto-Insurance-Predictive-Modeling-for-Smarter-Claims-Management
# Auto Insurance Fraud Detection

## Project Overview

This project analyzes an auto insurance claims dataset to identify fraudulent claims. The primary goal is to build a machine learning model that can accurately predict whether a claim is fraudulent based on its features. The analysis involves data cleaning, exploratory data analysis, model building, and feature importance evaluation.

---

## Dataset

The dataset used is `Auto_Insurance_Fraud_Claims_File01.csv`, which contains information on 40,000 insurance claims with 53 different features, including policy details, insured person's information, accident details, and claim amounts.

---

## Methodology

The project follows these key steps:

1.  **Data Cleaning and Preprocessing:**
    * Missing values in categorical columns were filled with the most frequent value.
    * Missing values in numerical columns were filled with the mean.
    * Categorical features were converted to numerical format using `LabelEncoder`.
    * All features were scaled using `StandardScaler` for model compatibility.

2.  **Initial Modeling & Overfitting:**
    * Several classification models were initially trained, including Logistic Regression, Decision Tree, Random Forest, KNN, and XGBoost.
    * Models like Random Forest and Decision Tree achieved a perfect accuracy of 100%, which indicated a high degree of **overfitting**. This means the models were too closely fitted to the training data and would not generalize well to new data.

3.  **Refined Analysis and Feature Importance:**
    * To address overfitting, a simpler `DecisionTreeClassifier` with a `max_depth` of 5 was trained.
    * The feature importances from this model were extracted to identify the most influential predictors of fraud.

---

## Key Findings

The refined analysis provided a more realistic model and highlighted the most critical features for fraud detection.

* **Model Performance:** The new Decision Tree model achieved an **accuracy of 91%**. It showed high recall (95%) for genuine claims and good recall (80%) for fraudulent claims.

* **Top 15 Most Important Features:** The analysis identified the following features as the most significant predictors of fraud:
    1.  `Vehicle_Claim`
    2.  `Total_Claim`
    3.  `Property_Claim`
    4.  `Accident_Severity`
    5.  `Injury_Claim`
    6.  `Age_Insured`
    7.  `Accident_Hour`
    8.  `Policy_Premium`
    9.  `Capital_Loss`
    10. `Policy_Ded`
    11. `Capital_Gains`
    12. `Customer_Life_Value1`
    13. `Auto_Year`
    14. `Witnesses`
    15. `Num_of_Vehicles_Involved`

![Top 15 Features](https://i.imgur.com/your-image-url.png)
*You can replace the link above with a screenshot of the feature importance plot from your notebook.*

---

## Files in this Repository

* `app.ipynb`: The main Jupyter Notebook containing all the code for the analysis.
* `Auto_Insurance_Fraud_Claims_File01.csv`: The original dataset used for this project.
* `important.csv`: A new, filtered CSV file containing only the top 15 most predictive features along with the fraud indicator. This dataset can be used for building more efficient models.

---

## How to Use

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  Install the required libraries:
    ```bash
    pip install pandas scikit-learn matplotlib seaborn xgboost
    ```
3.  Open and run the `app.ipynb` notebook in a Jupyter environment.

---

## Dependencies

* pandas
* scikit-learn
* matplotlib
* seaborn
* xgboost
