# Banking Loan Default Analysis - Deterministic Branches version

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Define base directory and output directory
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / 'outputs'

def main():
    # Load engineered training data
    p = OUT_DIR / 'train_features.csv'
    if not p.exists():
        raise FileNotFoundError('Run 2_feature_engineering.py first')
    df = pd.read_csv(p)

    # Candidate list of features (only pick the ones present in dataset)
    candidate_features = [
        'current_loan_amount','term','credit_score','annual_income','years_in_current_job_num',
        'monthly_debt','dti','years_of_credit_history','months_since_last_delinquent',
        'number_of_open_accounts','number_of_credit_problems','current_credit_balance',
        'maximum_open_credit','bankruptcies','tax_liens','home_ownership','purpose','risk_segment'
    ]
    features = [f for f in candidate_features if f in df.columns]
    print('Using features:', features)

    # Input features X
    X = df[features].copy()

    # Target variable y (default_flag)
    y = df['default_flag'].astype(int) if 'default_flag' in df.columns else pd.Series([0]*len(df))

    # Train/validation split (stratified when possible)
    if y.nunique() > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    # Identify numeric and categorical feature lists
    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in X.columns if c not in numeric_feats]

    # Define numeric preprocessing: median imputation + scaling
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define categorical preprocessing: fill missing + OneHotEncoding
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Combine preprocessing pipelines
    preproc = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    # Build final pipeline: preprocessing + Logistic Regression model
    clf = Pipeline([
        ('preproc', preproc),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    print('Training model...')
    clf.fit(X_train, y_train)  # Train model
    print('Model trained.')

    # Evaluate model
    if len(set(y_val)) > 1:  # Avoid errors in case of constant target
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:,1]
        report = classification_report(y_val, y_pred, output_dict=True)

        # ROC AUC
        try:
            roc = roc_auc_score(y_val, y_proba)
        except Exception:
            roc = None

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
    else:
        # If only one class exists in validation split
        y_pred = clf.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        roc = None
        cm = confusion_matrix(y_val, y_pred)

    # Save model
    joblib.dump(clf, OUT_DIR/'logreg_default_model.joblib')

    # Save classification report
    pd.DataFrame(report).transpose().to_csv(OUT_DIR/'classification_report.csv')

    # Save ROC AUC summary
    pd.DataFrame([{'roc_auc': roc}]).to_csv(OUT_DIR/'metrics_summary.csv', index=False)

    # Plot and save confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks([0,1], ['Pred 0','Pred 1'])
    plt.yticks([0,1], ['True 0','True 1'])

    # Annotate values in confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i,j],
                ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black'
            )
    plt.tight_layout()
    plt.savefig(OUT_DIR/'confusion_matrix.png')
    plt.close()

    print('Model + metrics saved')

if __name__ == '__main__':
    main()
