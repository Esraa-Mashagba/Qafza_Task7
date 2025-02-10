import joblib
import gradio as gr
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from huggingface_hub import login, HfApi

# --- Data Ingestion ---
def load_data():
    print("Loading datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

# --- Data Preprocessing ---
def preprocess_data(train, test):
    drop_columns = ['Id', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2']
    train.drop(columns=drop_columns, inplace=True, errors='ignore')
    test.drop(columns=drop_columns, inplace=True, errors='ignore')

    # Handle Missing Data
    print("Handling missing data...")
    Data = pd.concat((train.drop(columns=['SalePrice']), test)).reset_index(drop=True)
    Target = train['SalePrice']
    
    # Fill missing values
    categorical_cols = Data.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
       Data[col] = Data[col].fillna(Data[col].mode()[0])

    Data = pd.get_dummies(Data, drop_first=True)

    ntrain = train.shape[0]
    return Data, Target, ntrain

# --- Model Training ---
def train_models(X_train, y_train, X_val, y_val):
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0)
    }
    
    best_model = None
    best_rmse = float('inf')
    best_model_name = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"{model_name} RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name
    
    return best_model, best_model_name

# --- Save Model ---
def save_model(model, model_name):
    model_path = f"{model_name}.pkl"
    joblib.dump(model, model_path)
    return model_path

# --- Gradio Deployment ---
def predict_price(*input_data):
    df = pd.DataFrame([input_data], columns=X_train.columns)
    prediction = best_model.predict(df)
    return {"Predicted House Price": prediction[0]}

# Load and preprocess data
train, test = load_data()
Data, Target, ntrain = preprocess_data(train, test)
X_train, X_val, y_train, y_val = train_test_split(Data[:ntrain], Target, test_size=0.25, random_state=42)

# Train and save model
best_model, best_model_name = train_models(X_train, y_train, X_val, y_val)
model_path = save_model(best_model, best_model_name)

# --- Hugging Face Upload ---



login(token="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")  # Replace with your token
api = HfApi()
# api.create_repo("Qafza_Task", repo_type="model")
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=f"models/{model_path}",
    repo_id="EsrMash/Qafza_Task",
    repo_type="model"
)
print(f"Model {best_model_name} uploaded to Hugging Face!")

# --- Deploy with Gradio ---
inputs = [gr.Number(label=col) for col in X_train.columns]
outputs = gr.JSON()
app = gr.Interface(fn=predict_price, inputs=inputs, outputs=outputs, title="House Price Prediction")
app.launch(share=True)


# # --- Model Deployment ---
# app = Flask(__name__)
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     prediction = model.predict([[data['day']]])
#     return jsonify({'prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
