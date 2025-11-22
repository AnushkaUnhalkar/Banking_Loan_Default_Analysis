# Banking Loan Default Analysis - Deterministic Branches version

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base directory and output folder
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'outputs'

def save_hist(series, out):
    """
    Saves a histogram plot for any numeric pandas Series.
    - Drops NaN values
    - Uses 50 bins by default
    - Saves the histogram to the specified output file
    """
    plt.figure(figsize=(8,4))
    plt.hist(series.dropna(), bins=50)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    # Load feature-engineered dataset
    p = OUT_DIR / 'train_features.csv'
    if not p.exists():
        raise FileNotFoundError('Run 2_feature_engineering.py first')
    df = pd.read_csv(p)

    # Histogram for loan amount distribution
    if 'current_loan_amount' in df.columns:
        save_hist(df['current_loan_amount'], OUT_DIR/'loan_amount_distribution.png')

    # Select all numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure default_flag is included even if non-numeric type originally
    if 'default_flag' in df.columns and 'default_flag' not in num_cols:
        num_cols.append('default_flag')

    # Filter only columns actually present in dataframe
    num_cols = [c for c in num_cols if c in df.columns]

    # Correlation matrix for numeric columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()

        plt.figure(figsize=(10,8))
        plt.imshow(corr, aspect='auto')
        plt.colorbar()

        # Axis labels
        plt.xticks(range(len(corr)), corr.columns, rotation=90)
        plt.yticks(range(len(corr)), corr.columns)

        plt.tight_layout()
        plt.savefig(OUT_DIR/'correlation_matrix.png')
        plt.close()

    # Purpose-wise default rate analysis
    if 'purpose' in df.columns:
        purpose = df.groupby('purpose').agg(
            total=('loan_id','count'),
            defaults=('default_flag','sum')
        ).reset_index()

        # Compute default rate
        purpose['default_rate'] = purpose['defaults'] / purpose['total']

        # Save top 10 highest default-rate purposes
        purpose.sort_values('default_rate', ascending=False).head(10).to_csv(
            OUT_DIR/'purpose_top10_default_rate.csv', index=False
        )

    # Risk segment summary stats
    if 'risk_segment' in df.columns:
        risk_summary = df.groupby('risk_segment').agg(
            count=('loan_id','count'),
            defaults=('default_flag','sum')
        )
        risk_summary['default_rate'] = risk_summary['defaults'] / risk_summary['count']
        risk_summary.to_csv(OUT_DIR/'risk_segment_summary.csv')

    # Branch-wise performance metrics (if branch_id exists)
    if 'branch_id' in df.columns and 'repayment_rate' in df.columns:
        branch_perf = df.groupby('branch_id').agg(
            avg_repayment_rate=('repayment_rate','mean'),
            defaults=('default_flag','sum'),
            total=('loan_id','count')
        ).reset_index()

        # Compute default rate
        branch_perf['default_rate'] = branch_perf['defaults'] / branch_perf['total']

        branch_perf.to_csv(OUT_DIR/'branch_performance.csv', index=False)

    print('EDA outputs saved')

if __name__ == '__main__':
    main()
