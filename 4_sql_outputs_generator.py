# Banking Loan Default Analysis - Deterministic Branches version

from pathlib import Path
import pandas as pd

# Set base and output directories
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(exist_ok=True)   # Ensure outputs folder exists

def safe_has(df, cols):
    """
    Helper function:
    Returns True only if ALL columns in `cols` exist in the dataframe.
    """
    return all(c in df.columns for c in cols)

def main():
    # Load the feature-engineered train CSV
    p = OUT_DIR / 'train_features.csv'
    if not p.exists():
        raise FileNotFoundError('Run 2_feature_engineering.py first')
    df = pd.read_csv(p)

    # --------------------------
    # Total loan amount disbursed per branch
    # --------------------------
    if 'branch_id' in df.columns and 'current_loan_amount' in df.columns:
        df.groupby('branch_id').agg(
            total_disbursed=('current_loan_amount','sum')
        ).reset_index().to_csv(OUT_DIR/'total_loans_by_branch.csv', index=False)

    # --------------------------
    # Customers with overdue loans
    # --------------------------
    if 'loan_status' in df.columns:
        df[df['loan_status'].str.contains('overdue', case=False, na=False)][
            ['customer_id','loan_id','loan_status']
        ].to_csv(OUT_DIR/'customers_with_overdue.csv', index=False)
    else:
        # If status column missing, save empty structure
        pd.DataFrame(columns=['customer_id','loan_id','loan_status']).to_csv(
            OUT_DIR/'customers_with_overdue.csv', index=False
        )

    # --------------------------
    # Average loan amount by purpose
    # --------------------------
    if safe_has(df, ['purpose','current_loan_amount']):
        df.groupby('purpose').agg(
            avg_amount=('current_loan_amount','mean')
        ).reset_index().to_csv(OUT_DIR/'avg_loan_by_purpose.csv', index=False)

    # --------------------------
    # Count of defaulted loans by risk segment
    # --------------------------
    if 'risk_segment' in df.columns:
        df[df['default_flag']==1].groupby('risk_segment').size().reset_index(
            name='default_count'
        ).to_csv(OUT_DIR/'defaulted_loans_by_segment.csv', index=False)

    # --------------------------
    # Rank branches based on number of defaulters
    # --------------------------
    if 'branch_id' in df.columns:
        df[df['default_flag']==1].groupby('branch_id').size().reset_index(
            name='defaulters'
        ).sort_values(
            'defaulters', ascending=False
        ).to_csv(OUT_DIR/'rank_branches_by_defaulters.csv', index=False)

    # --------------------------
    # Pivot table: approvals vs rejections per branch
    # Loan status categories become columns
    # --------------------------
    if 'branch_id' in df.columns and 'loan_status' in df.columns:
        df.pivot_table(
            index='branch_id',
            columns='loan_status',
            values='loan_id',
            aggfunc='count',
            fill_value=0
        ).reset_index().to_csv(OUT_DIR/'approved_vs_rejected_by_branch.csv', index=False)

    # --------------------------
    # High-risk customer extraction
    # Fallback defaults added if certain columns missing
    # --------------------------
    df_copy = df.copy()

    # Ensure required columns exist
    if 'credit_score' not in df_copy.columns:
        df_copy['credit_score'] = 9999     # Safe dummy value
    if 'dti' not in df_copy.columns:
        df_copy['dti'] = -1                # Safe dummy value
    if 'default_flag' not in df_copy.columns:
        df_copy['default_flag'] = 0        # Assume no default

    # Rule-based high-risk filter
    df_copy[
        (df_copy['credit_score'] < 580) |
        (df_copy['dti'] > 40) |
        (df_copy['default_flag'] == 1)
    ][['customer_id','credit_score','dti','default_flag']].to_csv(
        OUT_DIR/'high_risk_customers.csv', index=False
    )

    # --------------------------
    # Loan issuance trend by month + loan status
    # --------------------------
    if 'issue_month' in df.columns and 'loan_status' in df.columns:
        df.groupby(['issue_month','loan_status']).size().unstack(
            fill_value=0
        ).reset_index().to_csv(OUT_DIR/'loan_trend_by_status.csv', index=False)

    # --------------------------
    # Branch geo performance (default rate + coordinates)
    # --------------------------
    if (
        'branch_id' in df.columns and
        'branch_lat' in df.columns and
        'branch_lon' in df.columns
    ):
        g = df.groupby('branch_id').agg(
            defaults=('default_flag','sum'),
            total=('loan_id','count')
        ).reset_index()

        # Compute default rate
        g['default_rate'] = g['defaults'] / g['total']

        # Extract unique coordinates for each branch
        coords = df[['branch_id','branch_lat','branch_lon']].drop_duplicates(
            subset=['branch_id']
        )

        # Merge summary + coordinates
        g = g.merge(coords, on='branch_id', how='left')

        g.to_csv(OUT_DIR/'branch_geo_summary.csv', index=False)

    print('SQL-equivalent CSVs generated')

if __name__ == '__main__':
    main()
