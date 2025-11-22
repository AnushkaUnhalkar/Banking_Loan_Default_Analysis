# Banking Loan Default Analysis Pipeline

## Overview
This project provides a deterministic, step-by-step pipeline for performing banking loan default analysis.
It includes modules for data loading, cleaning, feature engineering, exploratory data analysis, SQL-style summary generation, machine learning model training, and Power BI export preparation.

The entire workflow is executed automatically using a Python pipeline runner script.

---

## 📌 Pipeline Steps

### 1. Data Loading & Cleaning (`1_data_loading_cleaning.py`)
- Loads the raw CSV dataset.
- Handles missing values, inconsistent formats, and basic preprocessing.
- Saves the cleaned dataset for the next stage.

### 2. Feature Engineering (`2_feature_engineering.py`)
- Creates additional meaningful features.
- Encodes categorical variables.
- Outputs a fully processed dataset.

### 3. Exploratory Data Analysis (`3_eda_analysis.py`)
- Generates visualizations (distribution plots, boxplots, correlations).
- Saves graphs to the `outputs/plots/` directory.

### 4. SQL Outputs Generator (`4_sql_outputs_generator.py`)
- Computes SQL-like analytical summaries.
- Outputs aggregate CSV files for reporting and BI tools.

### 5. Model Training (`5_model_training.py`)
- Trains a machine learning classifier on processed data.
- Logs accuracy, AUC, and other model metrics.
- Saves the trained model and evaluation files.

### 6. Power BI Export Preparation (`6_powerbi_export_preparation.py`)
- Produces BI-friendly datasets.
- Generates purpose-level summaries and risk categorizations.

---

## ▶ Running the Entire Pipeline

Run the controller script:

```
python run_pipeline.py
```

This script:
- Executes each Python file sequentially  
- Aborts if any step fails  
- Ensures consistent, reproducible processing  

---

## Folder Structure

```
project/
│
├── 1_data_loading_cleaning.py
├── 2_feature_engineering.py
├── 3_eda_analysis.py
├── 4_sql_outputs_generator.py
├── 5_model_training.py
├── 6_powerbi_export_preparation.py
├── run_pipeline.py
│
├── data/
│   ├── credit_test.csv
│   ├── credit_train.csv
│
└── outputs/
    ├── cleaned/
    ├── engineered/
    ├── plots/
    ├── sql_summaries/
    ├── model/
    └── powerbi/
```

---

## ✔ Requirements

Install required libraries:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## 📊 Final Output

After the complete pipeline run:
- Cleaned data files  
- Engineered features  
- SQL-style summary CSVs  
- ML model + metrics  
- Power BI dataset exports  

All outputs are stored in the `outputs/` directory.

---

## 📝 Author
**Anushka Unhalkar**

This project is part of a Banking Loan Default Analysis automation workflow.
