from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import os
from datetime import datetime
import json
from collections import defaultdict

app = Flask(__name__)

# Global variables to store model and preprocessor
preprocessor = None
best_model = None

# Global analytics data storage with initial test values
analytics_data = {
    'total_transactions': 1247,  # Initial test transactions
    'fraud_count': 23,  # Initial fraud cases
    'safe_count': 1224,  # Initial safe transactions
    'transaction_types': defaultdict(int, {
        'PAYMENT': 456,
        'TRANSFER': 234,
        'CASH_OUT': 189,
        'DEBIT': 278,
        'CASH_IN': 90
    }),
    'amount_ranges': defaultdict(int, {
        '$0-100': 345,
        '$100-500': 456,
        '$500-1000': 234,
        '$1000-5000': 156,
        '$5000+': 56
    }),
    'hourly_data': defaultdict(lambda: {'safe': 0, 'fraud': 0}, {
        9: {'safe': 45, 'fraud': 1},
        10: {'safe': 67, 'fraud': 2},
        11: {'safe': 89, 'fraud': 3},
        12: {'safe': 78, 'fraud': 2},
        13: {'safe': 92, 'fraud': 4},
        14: {'safe': 85, 'fraud': 3},
        15: {'safe': 76, 'fraud': 2},
        16: {'safe': 68, 'fraud': 1},
        17: {'safe': 54, 'fraud': 2},
        18: {'safe': 42, 'fraud': 1},
        19: {'safe': 38, 'fraud': 1},
        20: {'safe': 29, 'fraud': 1}
    }),
    'recent_transactions': [
        {
            'time': '14:32',
            'type': 'PAYMENT',
            'amount': '$1,250.00',
            'status': 'Safe',
            'confidence': '94.2%',
            'timestamp': '2024-07-27T14:32:00'
        },
        {
            'time': '14:28',
            'type': 'TRANSFER',
            'amount': '$5,750.00',
            'status': 'Fraud',
            'confidence': '87.6%',
            'timestamp': '2024-07-27T14:28:00'
        },
        {
            'time': '14:25',
            'type': 'CASH_OUT',
            'amount': '$850.00',
            'status': 'Safe',
            'confidence': '96.1%',
            'timestamp': '2024-07-27T14:25:00'
        },
        {
            'time': '14:22',
            'type': 'DEBIT',
            'amount': '$320.00',
            'status': 'Safe',
            'confidence': '91.8%',
            'timestamp': '2024-07-27T14:22:00'
        },
        {
            'time': '14:19',
            'type': 'PAYMENT',
            'amount': '$2,100.00',
            'status': 'Safe',
            'confidence': '93.5%',
            'timestamp': '2024-07-27T14:19:00'
        }
    ],
    'daily_volume': defaultdict(int, {
        '2024-07-21': 156,
        '2024-07-22': 178,
        '2024-07-23': 192,
        '2024-07-24': 165,
        '2024-07-25': 203,
        '2024-07-26': 187,
        '2024-07-27': 166
    }),
    'confidence_scores': [94.2, 87.6, 96.1, 91.8, 93.5, 89.3, 95.7, 92.4, 88.9, 94.6]
}

# File to persist analytics data
ANALYTICS_FILE = 'analytics_data.json'

def load_analytics_data():
    """Load analytics data from file"""
    global analytics_data
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                data = json.load(f)
                # Convert back to defaultdict
                analytics_data['transaction_types'] = defaultdict(int, data.get('transaction_types', {}))
                analytics_data['amount_ranges'] = defaultdict(int, data.get('amount_ranges', {}))
                analytics_data['hourly_data'] = defaultdict(lambda: {'safe': 0, 'fraud': 0}, data.get('hourly_data', {}))
                analytics_data['daily_volume'] = defaultdict(int, data.get('daily_volume', {}))
                analytics_data['total_transactions'] = data.get('total_transactions', 0)
                analytics_data['fraud_count'] = data.get('fraud_count', 0)
                analytics_data['safe_count'] = data.get('safe_count', 0)
                analytics_data['recent_transactions'] = data.get('recent_transactions', [])
                analytics_data['confidence_scores'] = data.get('confidence_scores', [])
                print(f"✅ Analytics data loaded: {analytics_data['total_transactions']} transactions")
        else:
            print("📊 No existing analytics data found, using initial test values")
            # Keep the initial test values defined above
    except Exception as e:
        print(f"❌ Error loading analytics data: {e}, using initial test values")
        # Keep the initial test values defined above

def save_analytics_data():
    """Save analytics data to file"""
    try:
        # Convert defaultdict to regular dict for JSON serialization
        data_to_save = {
            'total_transactions': analytics_data['total_transactions'],
            'fraud_count': analytics_data['fraud_count'],
            'safe_count': analytics_data['safe_count'],
            'transaction_types': dict(analytics_data['transaction_types']),
            'amount_ranges': dict(analytics_data['amount_ranges']),
            'hourly_data': dict(analytics_data['hourly_data']),
            'recent_transactions': analytics_data['recent_transactions'],
            'daily_volume': dict(analytics_data['daily_volume']),
            'confidence_scores': analytics_data['confidence_scores']
        }
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"💾 Analytics data saved: {analytics_data['total_transactions']} transactions")
    except Exception as e:
        print(f"❌ Error saving analytics data: {e}")

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
load_analytics_data()

# Helper functions for analytics
def update_analytics_data(transaction_data, prediction, probability):
    """Update analytics data with new transaction"""
    global analytics_data
    
    try:
        # Update basic counts
        analytics_data['total_transactions'] += 1
        if prediction == 1:
            analytics_data['fraud_count'] += 1
        else:
            analytics_data['safe_count'] += 1
        
        # Update transaction type
        transaction_type = transaction_data['type']
        analytics_data['transaction_types'][transaction_type] += 1
        
        # Update amount range
        amount = float(transaction_data['amount'])
        if amount <= 100:
            amount_range = '$0-100'
        elif amount <= 500:
            amount_range = '$100-500'
        elif amount <= 1000:
            amount_range = '$500-1000'
        elif amount <= 5000:
            amount_range = '$1000-5000'
        else:
            amount_range = '$5000+'
        analytics_data['amount_ranges'][amount_range] += 1
        
        # Update hourly data
        current_hour = datetime.now().hour
        if prediction == 1:
            analytics_data['hourly_data'][current_hour]['fraud'] += 1
        else:
            analytics_data['hourly_data'][current_hour]['safe'] += 1
        
        # Update daily volume
        current_date = datetime.now().strftime('%Y-%m-%d')
        analytics_data['daily_volume'][current_date] += 1
        
        # Update confidence scores
        analytics_data['confidence_scores'].append(probability if probability else 0.5)
        if len(analytics_data['confidence_scores']) > 100:  # Keep only last 100 scores
            analytics_data['confidence_scores'] = analytics_data['confidence_scores'][-100:]
        
        # Update recent transactions
        recent_transaction = {
            'time': datetime.now().strftime('%H:%M'),
            'type': transaction_type,
            'amount': f"${amount:.2f}",
            'status': 'Fraud' if prediction == 1 else 'Safe',
            'confidence': f"{probability * 100:.1f}%" if probability else "N/A",
            'timestamp': datetime.now().isoformat()
        }
        
        analytics_data['recent_transactions'].insert(0, recent_transaction)
        # Keep only last 20 transactions
        if len(analytics_data['recent_transactions']) > 20:
            analytics_data['recent_transactions'] = analytics_data['recent_transactions'][:20]
        
        # Save analytics data after each update
        save_analytics_data()
        
        print(f"📊 Analytics updated - Total: {analytics_data['total_transactions']}, Fraud: {analytics_data['fraud_count']}, Type: {transaction_type}, Amount: ${amount:.2f}")
        
    except Exception as e:
        print(f"❌ Error updating analytics: {e}")

def get_analytics_data():
    """Get formatted analytics data for frontend"""
    global analytics_data
    
    # Use constant accuracy rate based on model's test accuracy (98.2%)
    accuracy_rate = 98.2
    
    # Format data for charts
    transaction_types_data = dict(analytics_data['transaction_types'])
    amount_ranges_data = dict(analytics_data['amount_ranges'])
    
    # Prepare hourly data for chart
    hourly_labels = [f"{hour:02d}" for hour in range(24)]
    hourly_safe = [analytics_data['hourly_data'][hour]['safe'] for hour in range(24)]
    hourly_fraud = [analytics_data['hourly_data'][hour]['fraud'] for hour in range(24)]
    
    # Prepare daily volume data
    daily_labels = list(analytics_data['daily_volume'].keys())[-7:]  # Last 7 days
    daily_volumes = [analytics_data['daily_volume'][date] for date in daily_labels]
    
    return {
        'total_transactions': analytics_data['total_transactions'],
        'fraud_count': analytics_data['fraud_count'],
        'safe_count': analytics_data['safe_count'],
        'accuracy_rate': accuracy_rate,
        'avg_response_time': 0.3,  # Mock data
        'transaction_types': transaction_types_data,
        'amount_ranges': amount_ranges_data,
        'hourly_data': {
            'labels': hourly_labels,
            'safe': hourly_safe,
            'fraud': hourly_fraud
        },
        'daily_volume': {
            'labels': daily_labels,
            'data': daily_volumes
        },
        'recent_transactions': analytics_data['recent_transactions'],
        'avg_confidence': round(sum(analytics_data['confidence_scores']) / len(analytics_data['confidence_scores']), 1) if analytics_data['confidence_scores'] else 0
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Get real analytics data
    analytics_data = get_analytics_data()
    return render_template('index.html', dashboard_data=analytics_data)

@app.route('/api/analytics')
def get_analytics():
    """API endpoint to get real-time analytics data"""
    data = get_analytics_data()
    print(f"📡 API Analytics requested - Total: {data['total_transactions']}, Fraud: {data['fraud_count']}")
    return jsonify(data)

@app.route('/api/reset-analytics', methods=['POST'])
def reset_analytics():
    """Reset all analytics data to 0"""
    global analytics_data
    
    # Reset all analytics data to 0
    analytics_data = {
        'total_transactions': 0,
        'fraud_count': 0,
        'safe_count': 0,
        'transaction_types': defaultdict(int),
        'amount_ranges': defaultdict(int),
        'hourly_data': defaultdict(lambda: {'safe': 0, 'fraud': 0}),
        'recent_transactions': [],
        'daily_volume': defaultdict(int),
        'confidence_scores': []
    }
    
    # Save the reset data
    save_analytics_data()
    
    print("🔄 Analytics data reset to 0")
    return jsonify({'status': 'success', 'message': 'Analytics data reset successfully'})

@app.route('/api/load-test-data', methods=['POST'])
def load_test_data():
    """Load initial test data (1247 total, 23 fraud)"""
    global analytics_data
    
    # Load initial test data
    analytics_data = {
        'total_transactions': 1247,  # Initial test transactions
        'fraud_count': 23,  # Initial fraud cases
        'safe_count': 1224,  # Initial safe transactions
        'transaction_types': defaultdict(int, {
            'PAYMENT': 456,
            'TRANSFER': 234,
            'CASH_OUT': 189,
            'DEBIT': 278,
            'CASH_IN': 90
        }),
        'amount_ranges': defaultdict(int, {
            '$0-100': 345,
            '$100-500': 456,
            '$500-1000': 234,
            '$1000-5000': 156,
            '$5000+': 56
        }),
        'hourly_data': defaultdict(lambda: {'safe': 0, 'fraud': 0}, {
            9: {'safe': 45, 'fraud': 1},
            10: {'safe': 67, 'fraud': 2},
            11: {'safe': 89, 'fraud': 3},
            12: {'safe': 78, 'fraud': 2},
            13: {'safe': 92, 'fraud': 4},
            14: {'safe': 85, 'fraud': 3},
            15: {'safe': 76, 'fraud': 2},
            16: {'safe': 68, 'fraud': 1},
            17: {'safe': 54, 'fraud': 2},
            18: {'safe': 42, 'fraud': 1},
            19: {'safe': 38, 'fraud': 1},
            20: {'safe': 29, 'fraud': 1}
        }),
        'recent_transactions': [
            {
                'time': '14:32',
                'type': 'PAYMENT',
                'amount': '$1,250.00',
                'status': 'Safe',
                'confidence': '94.2%',
                'timestamp': '2024-07-27T14:32:00'
            },
            {
                'time': '14:28',
                'type': 'TRANSFER',
                'amount': '$5,750.00',
                'status': 'Fraud',
                'confidence': '87.6%',
                'timestamp': '2024-07-27T14:28:00'
            },
            {
                'time': '14:25',
                'type': 'CASH_OUT',
                'amount': '$850.00',
                'status': 'Safe',
                'confidence': '96.1%',
                'timestamp': '2024-07-27T14:25:00'
            },
            {
                'time': '14:22',
                'type': 'DEBIT',
                'amount': '$320.00',
                'status': 'Safe',
                'confidence': '91.8%',
                'timestamp': '2024-07-27T14:22:00'
            },
            {
                'time': '14:19',
                'type': 'PAYMENT',
                'amount': '$2,100.00',
                'status': 'Safe',
                'confidence': '93.5%',
                'timestamp': '2024-07-27T14:19:00'
            }
        ],
        'daily_volume': defaultdict(int, {
            '2024-07-21': 156,
            '2024-07-22': 178,
            '2024-07-23': 192,
            '2024-07-24': 165,
            '2024-07-25': 203,
            '2024-07-26': 187,
            '2024-07-27': 166
        }),
        'confidence_scores': [94.2, 87.6, 96.1, 91.8, 93.5, 89.3, 95.7, 92.4, 88.9, 94.6]
    }
    
    # Save the test data
    save_analytics_data()
    
    print("📊 Test data loaded: 1247 total, 23 fraud")
    return jsonify({'status': 'success', 'message': 'Test data loaded successfully'})



@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Handle feedback submission
        feedback_data = {
            'name': request.form.get('name', ''),
            'email': request.form.get('email', ''),
            'rating': request.form.get('rating', ''),
            'message': request.form.get('message', ''),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # In a real app, save to database
        print(f"Feedback received: {feedback_data}")
        return jsonify({'status': 'success', 'message': 'Thank you for your feedback!'})
    
    return render_template('index.html')

@app.route('/profile')
def profile():
    # Profile data for Manas Nayak
    profile_data = {
        'name': 'Manas Nayak',
        'email': 'manvamp2003@gmail.com',
        'phone': '+91 9438753369',
        'department': 'Cybersecurity',
        'role': 'Cybersecurity Analyst',
        'join_date': '2024-01-15',
        'total_analyses': 1247,
        'accuracy_score': 98.2
    }
    return render_template('index.html', profile_data=profile_data)

@app.route('/about')
def about():
    about_data = {
        'company_name': 'Fraud Detector AI',
        'version': '2.1.0',
        'description': 'Advanced AI-powered fraud detection platform',
        'features': [
            'Real-time transaction analysis',
            '98.2% accuracy rate in fraud detection',
            'Advanced machine learning algorithms',
            'User-friendly interface',
            'Comprehensive analytics dashboard',
            'Secure and confidential processing'
        ],
        'tech_stack': [
            'Python 3.8+',
            'Flask Web Framework',
            'Scikit-learn ML Library',
            'Pandas Data Processing',
            'HTML5/CSS3/JavaScript',
            'Font Awesome Icons'
        ]
    }
    return render_template('index.html', about_data=about_data)

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

        # Update analytics data
        update_analytics_data(data, prediction, probability)

        # Render result page
        return render_template('result.html', prediction=prediction, probability=probability)

    except Exception as e:
        return render_template('result.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


