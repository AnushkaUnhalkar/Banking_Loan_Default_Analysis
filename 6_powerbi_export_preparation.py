# Banking Loan Default Analysis - Deterministic Branches version

from pathlib import Path
import pandas as pd
import joblib

# Base directory and output directory paths
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'outputs'

def main():
    # Load engineered training dataset
    p = OUT_DIR / 'train_features.csv'
    if not p.exists():
        raise FileNotFoundError('Run 2_feature_engineering.py first')
    df = pd.read_csv(p)

    # Path to trained logistic regression model
    model_path = OUT_DIR/'logreg_default_model.joblib'

    # If model exists, load it and generate prediction probabilities
    if model_path.exists():
        try:
            clf = joblib.load(model_path)  # Load saved model

            # List of features used during model training
            candidate_features = [
                'current_loan_amount','term','credit_score','annual_income','years_in_current_job_num',
                'monthly_debt','dti','years_of_credit_history','months_since_last_delinquent',
                'number_of_open_accounts','number_of_credit_problems','current_credit_balance',
                'maximum_open_credit','bankruptcies','tax_liens','home_ownership','purpose','risk_segment'
            ]

            # Select only available features from dataset
            features = [f for f in candidate_features if f in df.columns]
            X = df[features].copy()

            # Predict default probabilities and store in dataframe
            try:
                df['default_probability'] = clf.predict_proba(X)[:,1]
            except Exception as e:
                print('Could not compute probabilities:', e)

        except Exception as e:
            print('Could not load model:', e)

    # Export full dataset for Power BI
    df.to_csv(OUT_DIR/'powerbi_customer_dataset.csv', index=False)

    # Risk segment summary (count + default rate)
    if 'risk_segment' in df.columns:
        df.groupby('risk_segment').agg(
            count=('loan_id','count'),
            default_rate=('default_flag','mean')
        ).reset_index().to_csv(OUT_DIR/'powerbi_risk_summary.csv', index=False)

    # Purpose-wise summary for Power BI
    if 'purpose' in df.columns:
        df.groupby('purpose').agg(
            total_loans=('loan_id','count'),
            avg_amount=('current_loan_amount','mean'),
            default_rate=('default_flag','mean')
        ).reset_index().to_csv(OUT_DIR/'powerbi_purpose_summary.csv', index=False)

    print('Power BI datasets saved')

if __name__ == '__main__':
    main()
