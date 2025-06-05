import pandas as pd
import numpy as np
import time
import os
import joblib # For saving the model and preprocessor
import xgboost as xgb # Import XGBoost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("--- Starting Model Training Script ---")

    # --- 1. Load Data ---
    x_processed_data_path = 'data/processed' # Path for X_features.csv
    y_processed_data_path = 'data/processed_v2' # Path for rule-based y_target.csv

    x_file_path = os.path.join(x_processed_data_path, 'X_features.csv')
    y_file_path = os.path.join(y_processed_data_path, 'y_target.csv')

    if not os.path.exists(x_file_path) or not os.path.exists(y_file_path):
        print(f"ERROR: Data files not found. Ensure '{x_file_path}' (from {x_processed_data_path}) and '{y_file_path}' (from {y_processed_data_path}) exist.")
        print("Please run the data generation scripts first (notebook for X_features, apply_rules_to_target.py for y_target).")
        return

    print(f"Loading X_features from {x_file_path}...")
    X = pd.read_csv(x_file_path)
    print(f"Loading y_target from {y_file_path}...")
    y = pd.read_csv(y_file_path).iloc[:, 0]

    # Removing the 1000-row limitation to use the full dataset
    # print("Selecting the first 1000 rows for training...")
    # X = X.head(1000)
    # y = y.head(1000)

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")

    if X.empty or y.empty:
        print("ERROR: Loaded data is empty.")
        return

    # --- 2. Define Feature Sets (must match notebook) ---
    # These lists should be identical to how they were defined in the notebook
    # when X was created.
    numerical_features = [
        'Company Size', 
        'MinExperience', 
        'MaxExperience', 
        'AverageSalary', 
        'NumberOfSkills',
        'Job Title_Word_Count',
        'Job Description_Word_Count',
        'Benefits_Word_Count',
        'Responsibilities_Word_Count',
        'Company Profile_Word_Count'
    ]
    # Filter to only include features present in the loaded X
    numerical_features = [col for col in numerical_features if col in X.columns]
    print(f"Using numerical features: {numerical_features}")

    categorical_features_for_ohe = [
        'Qualifications', 
        'Work Type', 
        'Role', 
        'Job Portal', 
        'Preference',
        'Country' 
    ]
    # Filter to only include features present in the loaded X
    categorical_features_for_ohe = [col for col in categorical_features_for_ohe if col in X.columns]
    print(f"Using categorical features for OHE: {categorical_features_for_ohe}")

    # Ensure all selected features are actually in X, otherwise ColumnTransformer will fail
    missing_num_feats = [f for f in numerical_features if f not in X.columns]
    missing_cat_feats = [f for f in categorical_features_for_ohe if f not in X.columns]
    if missing_num_feats:
        print(f"ERROR: Following numerical features defined but not in loaded X: {missing_num_feats}")
        return
    if missing_cat_feats:
        print(f"ERROR: Following categorical features defined but not in loaded X: {missing_cat_feats}")
        return


    # --- 3. Train-Test Split ---
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Shape of X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- 4. Preprocessing Pipeline ---
    print("Defining preprocessing pipeline...")
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features_for_ohe)
        ], 
        remainder='drop' # Drop any columns not specified, X should only contain these
    )
    
    print("Fitting preprocessor and transforming data...")
    # Fit on training data, transform both train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Shape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of X_test_processed: {X_test_processed.shape}")

    # --- 5. Model Training ---
    print("--- Training XGBoost Regressor Model ---")
    model = xgb.XGBRegressor(objective='reg:squarederror',
                               n_estimators=100,
                               random_state=42,
                               n_jobs=-1,
                               max_depth=6,
                               learning_rate=0.1,
                               subsample=0.8,
                               colsample_bytree=0.8)
    
    start_time = time.time()
    model.fit(X_train_processed, y_train)
    end_time = time.time()
    print(f"Model training completed in {end_time - start_time:.2f} seconds.")

    # --- 6. Model Evaluation ---
    print("--- Evaluating Model ---")
    y_train_pred = model.predict(X_train_processed)
    y_test_pred = model.predict(X_test_processed)
    
    print("Training Set Performance:")
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"  Mean Squared Error (MSE): {train_mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {train_rmse:.2f}")
    print(f"  R-squared (R2): {train_r2:.2f}")
    
    print("Test Set Performance:")
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"  Mean Squared Error (MSE): {test_mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {test_rmse:.2f}")
    print(f"  R-squared (R2): {test_r2:.2f}")

    print("--- Interpretation of Results ---")
    print("Evaluating model performance with rule-based 'TimeToFill_days'.")

    # --- 7. (Optional) Save the Model and Preprocessor ---
    models_path = 'models' # Path relative to the project root
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f"Created directory: {models_path}")

    preprocessor_path = os.path.join(models_path, 'preprocessor.joblib')
    model_path = os.path.join(models_path, 'xgboost_model.joblib')

    try:
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(model, model_path)
        print(f"Preprocessor saved to: {preprocessor_path}")
        print(f"Trained XGBoost model saved to: {model_path}")
    except Exception as e:
        print(f"Error saving model/preprocessor: {e}")
        
    print("--- Model Training Script Finished ---")

if __name__ == '__main__':
    main() 