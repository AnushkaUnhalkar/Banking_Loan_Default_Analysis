# Banking Loan Default Analysis - Deterministic Branches version

from pathlib import Path
import pandas as pd
import numpy as np
import hashlib

# Base directory of current script
BASE_DIR = Path(__file__).resolve().parent

# Output directory for cleaned files
OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

# Fallback paths (for environments like Jupyter)
FALLBACK_TRAIN = Path('/mnt/data/credit_train.csv')
FALLBACK_TEST = Path('/mnt/data/credit_test.csv')

def normalize_columns(df):
    """
    Standardizes column names:
    - Strips spaces
    - Replaces non-alphanumeric chars with underscore
    - Converts to lowercase
    """
    df = df.copy()
    df.columns = (df.columns.str.strip()
                  .str.replace(r'\W+', '_', regex=True)
                  .str.replace(r'_+', '_', regex=True)
                  .str.strip('_')
                  .str.lower())
    return df

def parse_years_in_job(x):
    """
    Parses 'years in current job' values:
    - Converts formats like '<1 year', '10+ years', 'n/a'
    - Extracts digits and returns numeric years
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('n/a','na','none'):
        return np.nan
    if '<' in s:
        return 0.5  # convert "<1 year" to 0.5
    digits = ''.join(ch for ch in s if ch.isdigit())
    try:
        return float(digits) if digits else np.nan
    except:
        return np.nan

def load_csv_try(paths):
    """
    Tries loading CSV from a list of possible paths.
    Returns the first existing file.
    """
    for p in paths:
        p = Path(p)
        if p.exists():
            print(f"Loading: {p}")
            return pd.read_csv(p)
    raise FileNotFoundError(f"No file found in: {paths}")

def deterministic_branch(customer_id, n_branches=15):
    """
    Assigns a deterministic branch ID using SHA-256 hashing.
    Ensures same customer always maps to same branch.
    """
    if pd.isna(customer_id):
        return f"BR{0:03d}"
    s = str(customer_id).encode('utf-8')
    h = int(hashlib.sha256(s).hexdigest(), 16)
    idx = h % n_branches
    return f"BR{idx+1:03d}"

def main():
    # Load training CSV (with fallback)
    train = load_csv_try([BASE_DIR/'credit_train.csv', FALLBACK_TRAIN])
    
    # Load test CSV if available, else create empty DF
    try:
        test = load_csv_try([BASE_DIR/'credit_test.csv', FALLBACK_TEST])
    except FileNotFoundError:
        print('credit_test.csv not found; continuing with empty test set')
        test = pd.DataFrame()

    # Normalize column names in both datasets
    train = normalize_columns(train)
    if not test.empty:
        test = normalize_columns(test)

    # Extra normalization: replace spaces with underscores and lowercase
    train.columns = [c.strip().lower().replace(' ', '_') for c in train.columns]
    if not test.empty:
        test.columns = [c.strip().lower().replace(' ', '_') for c in test.columns]

    # Parse 'years_in_current_job' into numeric form if column exists
    if 'years_in_current_job' in train.columns:
        train['years_in_current_job_num'] = train['years_in_current_job'].apply(parse_years_in_job)
        if not test.empty and 'years_in_current_job' in test.columns:
            test['years_in_current_job_num'] = test['years_in_current_job'].apply(parse_years_in_job)

    # Replace placeholder invalid values (99999999) with NaN
    for col in train.select_dtypes(include=['number']).columns:
        train[col] = train[col].replace(99999999, np.nan)

    # Deterministic branching using customer_id
    N_BRANCHES = 15
    if 'customer_id' in train.columns:
        # Assign branch IDs deterministically
        train['branch_id'] = train['customer_id'].apply(lambda x: deterministic_branch(x, N_BRANCHES))
        
        # Generate fake branch latitude/longitude based on index
        unique = sorted(train['branch_id'].unique())
        lat_map = {b: 19.0 + (i * 0.1) for i, b in enumerate(unique)}
        lon_map = {b: 72.8 + (i * 0.1) for i, b in enumerate(unique)}

        train['branch_lat'] = train['branch_id'].map(lat_map)
        train['branch_lon'] = train['branch_id'].map(lon_map)
    else:
        print('customer_id not found in train; branch assignment skipped')

    # Assign branches in test dataset similarly
    if not test.empty and 'customer_id' in test.columns:
        test['branch_id'] = test['customer_id'].apply(lambda x: deterministic_branch(x, N_BRANCHES))

        # Use same mapping as train when possible
        if 'branch_lat' in train.columns:
            test['branch_lat'] = test['branch_id'].map(lat_map)
            test['branch_lon'] = test['branch_id'].map(lon_map)
        else:
            # Create new mapping if train lacks branch data
            unique_t = sorted(test['branch_id'].unique())
            lat_map_t = {b: 19.0 + (i * 0.1) for i, b in enumerate(unique_t)}
            lon_map_t = {b: 72.8 + (i * 0.1) for i, b in enumerate(unique_t)}
            test['branch_lat'] = test['branch_id'].map(lat_map_t)
            test['branch_lon'] = test['branch_id'].map(lon_map_t)

    # If issue_date missing, create synthetic dates
    if 'issue_date' not in train.columns:
        rng = pd.date_range('2021-01-01', periods=365)
        train['issue_date'] = pd.to_datetime(pd.Series(np.random.choice(rng, size=len(train))))
        if not test.empty:
            test['issue_date'] = pd.to_datetime(pd.Series(np.random.choice(rng, size=len(test))))

    # Save cleaned CSVs
    train.to_csv(OUT_DIR/'clean_train.csv', index=False)
    if not test.empty:
        test.to_csv(OUT_DIR/'clean_test.csv', index=False)
    print('Saved cleaned train/test to outputs/')

if __name__ == '__main__':
    main()
