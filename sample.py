import pickle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
print("Model and preprocessor loaded successfully")