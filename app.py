import flask
import pandas as pd
import numpy as np
import joblib
import os

# Initialize the Flask application
app = flask.Flask(__name__)

# --- Load Model and Preprocessor ---
# Construct absolute paths to model files
base_dir = os.path.abspath(os.path.dirname(__file__))
preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
model_path = os.path.join(base_dir, 'models', 'lightgbm_model.joblib')

try:
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    print("Preprocessor and model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model or preprocessor file not found. Searched paths:\nPreprocessor: {preprocessor_path}\nModel: {model_path}")
    print("Please ensure 'preprocessor.joblib' and 'lightgbm_model.joblib' are in the 'models' directory.")
    preprocessor = None
    model = None
except Exception as e:
    print(f"An error occurred while loading the model or preprocessor: {e}")
    preprocessor = None
    model = None

# Define the feature names exactly as used during training
base_numerical_features = [
    'Company Size', 'MinExperience', 'MaxExperience', 'AverageSalary', 'NumberOfSkills'
]
word_count_features_map = {
    'JobTitleText': 'Job Title_Word_Count',
    'JobDescriptionText': 'Job Description_Word_Count',
    'BenefitsText': 'Benefits_Word_Count',
    'ResponsibilitiesText': 'Responsibilities_Word_Count',
    'CompanyProfileText': 'Company Profile_Word_Count'
}
all_word_count_feature_names = list(word_count_features_map.values())

numerical_features_for_model = base_numerical_features + all_word_count_feature_names

categorical_features_for_ohe = [
    'Qualifications', 'Work Type', 'Role', 'Job Portal', 'Preference', 'Country'
]

expected_input_columns = numerical_features_for_model + categorical_features_for_ohe

@app.route('/', methods=['GET'])
def home():
    if not preprocessor or not model:
        return "Error: Model or preprocessor not loaded. Please check server logs.", 500
    return flask.render_template('index.html', 
                                 raw_text_inputs=word_count_features_map, # Pass the map for form generation
                                 direct_numerical_inputs=base_numerical_features,
                                 categorical_inputs=categorical_features_for_ohe)

@app.route('/predict', methods=['POST'])
def predict():
    if not preprocessor or not model:
        return "Error: Model or preprocessor not loaded. Please check server logs.", 500
    try:
        form_data = {}

        # Process direct numerical inputs
        for feature in base_numerical_features:
            value = flask.request.form.get(feature)
            if value is None or value == '':
                print(f"Warning: Missing value for numerical feature '{feature}', using 0.")
                form_data[feature] = 0.0
            else:
                try:
                    form_data[feature] = float(value)
                except ValueError:
                    print(f"Warning: Could not convert {feature} value '{value}' to float. Using 0.")
                    form_data[feature] = 0.0
        
        # Process categorical inputs
        for feature in categorical_features_for_ohe:
            value = flask.request.form.get(feature)
            if value is None:
                form_data[feature] = "Missing"
                print(f"Warning: Missing value for categorical feature '{feature}', using 'Missing'.")
            else:
                form_data[feature] = str(value)

        # Process raw text inputs and calculate word counts
        for form_input_name, model_feature_name in word_count_features_map.items():
            text_value = flask.request.form.get(form_input_name, '') # Default to empty string
            form_data[model_feature_name] = len(text_value.split())
            
        input_df = pd.DataFrame([form_data], columns=expected_input_columns)

        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        predicted_days = round(prediction[0], 1) # Get the first prediction and round it

        return flask.render_template('result.html', prediction=predicted_days)

    except Exception as e:
        print(f"Error during prediction: {e}")
        # You might want to redirect to an error page or show a generic error
        return f"An error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    # Create 'models' and 'templates' directory if they don't exist (for local testing)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'templates'), exist_ok=True)
    
    print(f"To run the app, execute: flask --app {os.path.basename(__file__).replace('.py','')} run")
    print(f"Or in Python: app.run(debug=True)")
    # For development server:
    # app.run(debug=True) 
    # For production, use a WSGI server like Gunicorn or Waitress. 