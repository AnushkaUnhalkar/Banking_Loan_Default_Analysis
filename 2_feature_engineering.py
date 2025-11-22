# Banking Loan Default Analysis - Deterministic Branches version

from pathlib import Path
import pandas as pd
import numpy as np

# Base directory of the script
BASE_DIR = Path(__file__).resolve().parent

# Output directory containing cleaned datasets
OUT_DIR = BASE_DIR / 'outputs'

def parse_term(x):
    """
    Converts loan 'term' values into numeric months.
    - Extracts digits from strings like '36 months'
    - Handles NaN and non-string inputs
    """
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, str):
            # Extract only digit characters
            digits = ''.join(ch for ch in x if ch.isdigit())
            return float(digits) if digits else np.nan
        return float(x)
    except:
        return np.nan

def main():
    # Load cleaned training data generated in the previous script
    p = OUT_DIR / 'clean_train.csv'
    if not p.exists():
        raise FileNotFoundError('Run 1_data_loading_cleaning.py first')
    df = pd.read_csv(p)

    # Convert loan term into numeric months
    if 'term' in df.columns:
        df['term_months'] = df['term'].apply(parse_term)

    # Compute repayment_rate using monthly_debt × term_months / current_loan_amount
    if {'monthly_debt','term_months','current_loan_amount'}.issubset(df.columns):
        df['estimated_total_payment'] = df['monthly_debt'] * df['term_months']
        df['repayment_rate'] = df['estimated_total_payment'] / df['current_loan_amount']

        # Replace infinite repayment values with NaN (e.g., division by zero)
        df['repayment_rate'].replace([np.inf,-np.inf], np.nan, inplace=True)

    # Convert loan_status into a binary default flag
    if 'loan_status' in df.columns:
        df['default_flag'] = df['loan_status'].astype(str).str.lower().apply(
            lambda s: 1 if any(k in s for k in ['charged','default','collection','late','written_off','bad']) else 0
        )
    else:
        # If no status column, default all rows to non-default
        df['default_flag'] = 0

    # Rule-based risk segmentation
    def risk_category(r):
        """
        Categorizes rows into Low, Medium, or High risk using:
        - Credit score
        - DTI (debt-to-income)
        - Repayment rate
        - Default history
        """
        score = r.get('credit_score', np.nan)
        dti = r.get('dti', np.nan)
        repay = r.get('repayment_rate', np.nan)

        # High risk conditions
        if r.get('default_flag', 0) == 1: 
            return 'High'
        if pd.notna(score) and score < 580:
            return 'High'
        if pd.notna(dti) and dti > 40:
            return 'High'

        # Medium risk conditions
        if (pd.notna(score) and score < 700) or (pd.notna(repay) and repay < 0.5):
            return 'Medium'

        # Otherwise low risk
        return 'Low'

    df['risk_segment'] = df.apply(risk_category, axis=1)

    # Extract issue month (YYYY-MM) if issue_date is available
    if 'issue_date' in df.columns:
        df['issue_month'] = pd.to_datetime(df['issue_date']).dt.to_period('M').astype(str)

    # Save final feature-engineered dataset
    df.to_csv(OUT_DIR/'train_features.csv', index=False)
    print('Saved train_features.csv')

if __name__ == '__main__':
    main()
