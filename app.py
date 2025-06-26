# from flask import Flask, request, render_template
# import pandas as pd
# import pickle

# app = Flask(__name__)

# # Load the preprocessor and model with error handling
# try:
#     with open('preprocessor.pkl', 'rb') as f:
#         preprocessor = pickle.load(f)
# except Exception as e:
#     preprocessor = None
#     print(f"Error loading preprocessor: {e}")

# try:
#     with open('best_model.pkl', 'rb') as f:
#         best_model = pickle.load(f)
# except Exception as e:
#     best_model = None
#     print(f"Error loading model: {e}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if model and preprocessor are loaded
#     if preprocessor is None or best_model is None:
#         return render_template('result.html', error="Model or preprocessor failed to load. Please check the server logs.")

#     try:
#         # Extract form data
#         data = {
#             'type': request.form['type'],
#             'amount': float(request.form['amount']),
#             'oldbalanceOrg': float(request.form['oldbalanceOrg']),
#             'newbalanceOrig': float(request.form['newbalanceOrig']),
#             'oldbalanceDest': float(request.form['oldbalanceDest']),
#             'newbalanceDest': float(request.form['newbalanceDest'])
#         }
        
#         # Convert to DataFrame
#         df = pd.DataFrame([data])
        
#         # Preprocess the data
#         df_preprocessed = preprocessor.transform(df)
        
#         # Make prediction
#         prediction = best_model.predict(df_preprocessed)[0]
#         probability = best_model.predict_proba(df_preprocessed)[0][1] if hasattr(best_model, 'predict_proba') else None
        
#         # Render result page with prediction and probability
#         return render_template('result.html', prediction=prediction, probability=probability)
#     except Exception as e:
#         return render_template('result.html', error=f"An error occurred during prediction: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Global variables to store model and preprocessor
preprocessor = None
best_model = None

# Function to load model and preprocessor with detailed error checking
def load_artifacts():
    global preprocessor, best_model
    try:
        # Define file paths (adjust these if files are in a different directory)
        model_path = os.path.join(os.getcwd(), 'best_model.pkl')
        preprocessor_path = os.path.join(os.getcwd(), 'preprocessor.pkl')

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")

        # Load preprocessor
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
            print("Preprocessor loaded successfully")

        # Load model
        with open(model_path, 'rb') as f:
            best_model = pickle.load(f)
            print("Model loaded successfully")

        # Verify that the model has a predict method
        if not hasattr(best_model, 'predict'):
            raise AttributeError("Loaded model does not have a 'predict' method")

    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        preprocessor, best_model = None, None

# Load artifacts when the app starts
load_artifacts()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if preprocessor is None or best_model is None:
        return render_template('result.html', error="Model or preprocessor failed to load. Check server logs for details.")

    try:
        # Extract form data
        data = {
            'type': request.form['type'],
            'amount': float(request.form['amount']),
            'oldbalanceOrg': float(request.form['oldbalanceOrg']),
            'newbalanceOrig': float(request.form['newbalanceOrig']),
            'oldbalanceDest': float(request.form['oldbalanceDest']),
            'newbalanceDest': float(request.form['newbalanceDest'])
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Preprocess the data
        df_preprocessed = preprocessor.transform(df)

        # Make prediction
        prediction = best_model.predict(df_preprocessed)[0]
        probability = best_model.predict_proba(df_preprocessed)[0][1] if hasattr(best_model, 'predict_proba') else None

        # Render result page
        return render_template('result.html', prediction=prediction, probability=probability)

    except Exception as e:
        return render_template('result.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


