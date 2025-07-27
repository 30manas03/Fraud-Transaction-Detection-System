# Fraud Detection AI Application

An advanced AI-powered fraud detection platform built with Flask, featuring real-time transaction analysis, comprehensive analytics dashboard, and machine learning-based fraud detection.

## Features

- 🔍 **Real-time Fraud Detection**: Analyze transactions instantly with 98.2% accuracy
- 📊 **Interactive Dashboard**: Real-time analytics with charts and metrics
- 🎨 **Modern UI**: Responsive design with dark/light theme toggle
- 📱 **Mobile Friendly**: Optimized for all device sizes
- 🔄 **Real-time Updates**: Live dashboard updates with new predictions
- 📈 **Advanced Analytics**: Transaction volume, fraud distribution, and hourly activity charts

## Technology Stack

- **Backend**: Python, Flask, Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Data Processing**: Pandas, NumPy
- **Deployment**: Render, Gunicorn

## Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd fraud_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:5000`

## Deployment on Render

### Automatic Deployment (Recommended)

1. **Connect your GitHub repository to Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository

2. **Configure the deployment**
   - **Name**: `fraud-detection-app`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`

3. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### Manual Deployment

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Render"
   git push origin main
   ```

2. **Create a new Web Service on Render**
   - Follow the automatic deployment steps above

## File Structure

```
fraud_detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── Procfile             # Process file for deployment
├── runtime.txt          # Python runtime specification
├── templates/           # HTML templates
│   ├── index.html      # Main application page
│   └── result.html     # Prediction result page
├── static/             # Static assets
│   ├── logo.svg        # Application logo
│   └── *.png/jpg       # Images for About Us section
├── best_model.pkl      # Trained ML model
├── preprocessor.pkl    # Data preprocessor
└── analytics_data.json # Analytics data persistence
```

## Environment Variables

The application uses the following environment variables:
- `PORT`: Port number (automatically set by Render)
- `PYTHON_VERSION`: Python version (set to 3.10)
- `PYTHONPATH`: Python path (set to /opt/render/project/src)

## API Endpoints

- `GET /`: Main application page
- `POST /predict`: Fraud prediction endpoint
- `GET /api/analytics`: Get analytics data
- `POST /api/reset-analytics`: Reset analytics data
- `POST /api/load-test-data`: Load initial test data
- `POST /feedback`: Submit user feedback

## Model Information

- **Accuracy**: 98.2%
- **Algorithm**: Machine Learning (Scikit-learn)
- **Features**: Transaction type, amount, balance changes
- **Output**: Fraud/Safe classification with confidence score

## Support

For issues or questions:
- Check the application logs in Render dashboard
- Ensure all required files are present in the repository
- Verify the model files (`best_model.pkl`, `preprocessor.pkl`) are included

## License

This project is for educational and demonstration purposes. 