
# House Price Prediction

This project is designed to predict house prices based on various features using machine learning algorithms. The model utilizes both Random Forest and XGBoost regression techniques, selecting the best model based on performance. It then exposes a user-friendly interface using Gradio, allowing users to input property features and get predicted house prices.

## Project Structure

### Data Loading
- Loads training and test datasets from CSV files (`train.csv` and `test.csv`).

### Data Preprocessing
- Selects important features (e.g., `OverallQual`, `GrLivArea`, `GarageCars`, etc.) for training.
- Handles missing values by filling them with the median of each column.
- Splits the data into features (`Data`) and target (`SalePrice`).

### Model Training
- Trains both `RandomForestRegressor` and `XGBRegressor` on the training data.
- Compares the models using RMSE (Root Mean Squared Error) and selects the best-performing model.

### Gradio Interface
- A Gradio app that allows users to input features of a house and get the predicted price.

### Model Deployment (Hugging Face & Flask)
- **Hugging Face**: The trained model is uploaded to the Hugging Face repository for future use and sharing.
- **Flask (Optional)**: Provides an API for deploying the model in a web application.



## How to Use
1. Clone the repository.
2. Install the required libraries using the `pip` command.
3. Run the script to start the Gradio interface.
4. Enter the house features into the Gradio interface to get the predicted price.

For deployment in production, you can also use the Flask API for integration into a web application.

