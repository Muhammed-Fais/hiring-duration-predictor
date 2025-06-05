import pandas as pd
import numpy as np
import os

def apply_ttf_rules(features_df):
    """
    Applies rule-based logic to a DataFrame to generate TimeToFill_days.
    Assumes features_df contains all necessary columns from X_features.csv.
    """
    # Ensure a copy is being worked on if features_df is a slice/view from a larger df elsewhere
    df_subset = features_df.copy()

    # Base Time-to-Fill
    ttf_series = pd.Series(30, index=df_subset.index, dtype=float)

    # --- Prerequisite Column Checks & Preparation ---
    cs_median_fallback = 0
    if 'Company Size' in df_subset.columns and pd.api.types.is_numeric_dtype(df_subset['Company Size']) and not df_subset['Company Size'].isnull().all():
        cs_median_fallback = df_subset['Company Size'].median()

    cols_for_rules_defs = {
        'MinExperience': 0,
        'MaxExperience': 0,
        'NumberOfSkills': 0,
        'Company Size': cs_median_fallback,
        'Role': 'Unknown',
        'Job Description_Word_Count': 0,
        'Responsibilities_Word_Count': 0
    }

    for col, default_val in cols_for_rules_defs.items():
        if col not in df_subset.columns:
            print(f"INFO: Rule prerequisite column '{col}' not found in X_features.csv. Creating with default: {default_val}.")
            df_subset[col] = default_val
        elif col in ['MinExperience', 'MaxExperience', 'NumberOfSkills', 'Job Description_Word_Count', 'Responsibilities_Word_Count', 'Company Size']:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce').fillna(default_val)
        elif col == 'Role':
            df_subset[col] = df_subset[col].fillna(default_val).astype(str)

    # --- Apply Rules ---
    # 1. Experience Impact
    avg_exp = (df_subset['MinExperience'] + df_subset['MaxExperience']) / 2
    ttf_series += avg_exp * 2

    # 2. Number of Skills Impact
    ttf_series += df_subset['NumberOfSkills'] * 1

    # 3. Company Size Impact (Tiered)
    company_size_adjustment = pd.Series(0, index=df_subset.index, dtype=float)
    current_company_size = df_subset['Company Size']
    company_size_adjustment[current_company_size >= 20000] = 15
    company_size_adjustment[(current_company_size >= 5000) & (current_company_size < 20000)] = 10
    company_size_adjustment[(current_company_size >= 500) & (current_company_size < 5000)] = 5
    ttf_series += company_size_adjustment

    # 4. Role Impact (Case-Insensitive, Additive)
    role_lower = df_subset['Role'].astype(str).str.lower()
    role_adj = pd.Series(0, index=df_subset.index, dtype=float)
    senior_mgmt_kws = ['manager', 'director', 'lead', 'vp', 'president', 'chief', 'head']
    senior_staff_kws = ['senior', r'sr\.'] 
    specialist_kws = ['principal', 'architect']
    junior_kws = ['intern', 'trainee', 'junior', r'jr\.'] 

    for kw in senior_mgmt_kws:
        role_adj[role_lower.str.contains(kw, na=False, regex=False)] += 20
    for kw in senior_staff_kws: 
        role_adj[role_lower.str.contains(kw, na=False, regex=True)] += 10
    for kw in specialist_kws:
        role_adj[role_lower.str.contains(kw, na=False, regex=False)] += 12
    for kw in junior_kws: 
        role_adj[role_lower.str.contains(kw, na=False, regex=True)] -= 10
    ttf_series += role_adj

    # 5. Job Content Word Count Impact
    word_count_sum = df_subset['Job Description_Word_Count'] + df_subset['Responsibilities_Word_Count']
    ttf_series += (word_count_sum) / 150 * 1

    # 6. Constraints
    ttf_series = ttf_series.clip(lower=10, upper=200)
    return ttf_series.round().astype(int)

def main():
    print("--- Starting Rule-Based Target Generation ---")
    input_processed_data_path = 'data/processed'  # Path for X_features.csv
    output_processed_data_path = 'data/processed_v2' # Path for new y_target.csv

    x_features_file = os.path.join(input_processed_data_path, 'X_features.csv')
    y_target_file = os.path.join(output_processed_data_path, 'y_target.csv')

    # Create the output directory if it doesn't exist
    os.makedirs(output_processed_data_path, exist_ok=True)
    print(f"Ensured output directory exists: {output_processed_data_path}")

    if not os.path.exists(x_features_file):
        print(f"ERROR: X_features.csv not found at {x_features_file}. Please ensure the notebook has been run to generate it.")
        return

    print(f"Loading X_features from {x_features_file}...")
    try:
        x_df = pd.read_csv(x_features_file)
    except Exception as e:
        print(f"Error loading X_features.csv: {e}")
        return

    if x_df.empty:
        print("ERROR: X_features.csv is empty.")
        return

    print(f"Successfully loaded X_features with shape: {x_df.shape}")

    print("Applying rules to generate 'TimeToFill_days'...")
    new_time_to_fill = apply_ttf_rules(x_df)

    print(f"Saving new 'TimeToFill_days' to {y_target_file}...")
    try:
        y_df_to_save = pd.DataFrame({'TimeToFill_days': new_time_to_fill})
        y_df_to_save.to_csv(y_target_file, index=False)
        print(f"Successfully saved new y_target.csv to {y_target_file} with {len(new_time_to_fill)} rows.")
        print("\nSummary of new 'TimeToFill_days':")
        print(new_time_to_fill.describe())
    except Exception as e:
        print(f"Error saving y_target.csv: {e}")

    print("--- Rule-Based Target Generation Finished ---")

if __name__ == '__main__':
    main() 