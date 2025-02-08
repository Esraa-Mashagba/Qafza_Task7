# Install necessary libraries
import pandas as pd
import numpy as np
import joblib
import gradio as gr
from huggingface_hub import HfApi, login
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mage_ai
from mage_ai.data_preparation.decorators import data_loader, transformer, model_trainer, model_saver
from mage_ai.data_preparation.models.pipeline import Pipeline
import pickle
import os
from flask import Flask, request, jsonify

# # Create a Mage.ai pipeline
# pipeline = Pipeline.create(
#     name='house_price_prediction',
#     repo_path='./mage_repo'
# )


model = joblib.load('house_price_model.pkl')

# --- Hugging Face Upload ---
login(token="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")  # Replace with your token
api = HfApi()
api.upload_file(
    path_or_fileobj=model,
    path_in_repo=model,
    repo_id="EsrMash/Qafza_Task",
    repo_type="space"
)


def predict_price(OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt, YearRemodAdd, Fireplaces, LotArea, MasVnrArea):
    input_data = pd.DataFrame([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath,
                                YearBuilt, YearRemodAdd, Fireplaces, LotArea, MasVnrArea]],
                              columns=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath',
                                       'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'LotArea', 'MasVnrArea'])
    prediction = model.predict(input_data)
    return {"Predicted House Price": prediction[0]}

inputs = [
    gr.Number(label='Overall Quality (1-10)'),
    gr.Number(label='Above Ground Living Area (sqft)'),
    gr.Number(label='Garage Cars'),
    gr.Number(label='Total Basement Area (sqft)'),
    gr.Number(label='Full Bathrooms'),
    gr.Number(label='Year Built'),
    gr.Number(label='Year Remodeled'),
    gr.Number(label='Fireplaces'),
    gr.Number(label='Lot Area (sqft)'),
    gr.Number(label='Masonry Veneer Area (sqft)')
]
outputs = gr.JSON()

app = gr.Interface(fn=predict_price, inputs=inputs, outputs=outputs, title="House Price Prediction")
app.launch(share=True)



# # --- Deploy with Flask ---
# # Step 4: Model Deployment
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

