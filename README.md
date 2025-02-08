Here’s a suggested README for your code:

---

# House Price Prediction

This project is designed to predict house prices based on various features using machine learning algorithms. The model leverages the **Random Forest** and **XGBoost** regression techniques, selecting the best model based on performance. It then exposes a user-friendly interface using **Gradio**, allowing users to input property features and get predicted house prices.

## Project Structure

1. **Data Loading**:
   - Loads training and test datasets from CSV files (`train.csv` and `test.csv`).
   
2. **Data Preprocessing**:
   - Selects important features (`OverallQual`, `GrLivArea`, `GarageCars`, etc.) for training.
   - Handles missing values by filling them with the median of each column.
   - Splits the data into features (`Data`) and target (`SalePrice`).

3. **Model Training**:
   - Trains both **Random Forest Regressor** and **XGBoost Regressor** on the training data.
   - Compares the models using RMSE (Root Mean Squared Error) and selects the best-performing model.

4. **Gradio Interface**:
   - A Gradio app that allows users to input features of a house and get the predicted price.
   
## Requirements

To run this project, you need the following libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- gradio
- joblib


