# Banking Loan Default Analysis - Deterministic Branches version

import subprocess, sys

# List of script steps to run sequentially (each step depends on the previous one)
steps = [
    '1_data_loading_cleaning.py',          # Step 1: Load raw data and clean it
    '2_feature_engineering.py',            # Step 2: Generate additional cleaned features
    '3_eda_analysis.py',                   # Step 3: Perform EDA and create plots
    '4_sql_outputs_generator.py',          # Step 4: Generate SQL-like CSV summaries
    '5_model_training.py',                 # Step 5: Train ML model and save metrics
    '6_powerbi_export_preparation.py'      # Step 6: Prepare datasets for Power BI dashboards
]

# Loop through each script and execute it
for s in steps:
    print('\n--- Running', s, '---')

    # Run each Python file as a subprocess
    res = subprocess.run([sys.executable, s])

    # If any script fails, stop the pipeline
    if res.returncode != 0:
        print(f'Error: step {s} failed with return code {res.returncode}. Aborting.')
        sys.exit(res.returncode)
    else:
        print(f'Step {s} completed successfully.')

# Pipeline finished successfully
print('\nPipeline complete. Check the outputs/ folder for results.')