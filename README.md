

# SGD Regression Model – California Housing Dataset 🏡

This project builds a **regression model** using **Stochastic Gradient Descent** (SGDRegressor) to predict housing prices in California. The model utilizes polynomial features, **L2 regularization**, and **early stopping** for optimal performance.

## Model Details
- **Algorithm**: SGDRegressor (Stochastic Gradient Descent)
- **Regression Type**: Polynomial Regression (degree = 2)
- **Regularization**: L2 (Ridge)
- **Early Stopping**: Manual, based on validation loss
- **Loss Function**: Mean Squared Error (MSE)
- **Libraries**: scikit-learn, joblib

## Dataset
- **Source**: `fetch_california_housing()` from scikit-learn
- **Features Used**: 2 numerical features (simplified)
- **Target**: Median house value in California districts

## Files Included
- **model.pkl**: Trained SGDRegressor model
- **scaler.pkl**: StandardScaler for normalization
- **poly.pkl**: PolynomialFeatures transformer (degree 2)

## Usage
To use the model, follow these steps:

1. Install dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

2. Use `huggingface_hub` and `joblib` to download and load the model, scaler, and transformer:
    ```python
    from huggingface_hub import hf_hub_download
    import joblib

    # Download model files from Hugging Face
    model_file = hf_hub_download(repo_id="alihaider-debug/sgd-regression-california-housing", filename="model.pkl")
    scaler_file = hf_hub_download(repo_id="alihaider-debug/sgd-regression-california-housing", filename="scaler.pkl")
    poly_file = hf_hub_download(repo_id="alihaider-debug/sgd-regression-california-housing", filename="poly.pkl")

    # Load the model, scaler, and transformer
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    poly = joblib.load(poly_file)
    ```

3. For prediction, provide two numerical inputs (Median Income and Average Rooms):
    ```python
    input_data = np.array([[med_inc, ave_rooms]])
    input_poly = poly.transform(input_data)
    input_scaled = scaler.transform(input_poly)
    prediction = model.predict(input_scaled)
    print(f"Predicted House Value: ${prediction[0]:.2f}")
    ```

## Model Evaluation
- **MSE (Mean Squared Error)**: 0.54321
- **RMSE (Root Mean Squared Error)**: 0.7365
- **R² (R-squared)**: 0.8457

## License
This project is licensed under the MIT License.

## Weights & Biases Dashboard
📈 Click [here](https://wandb.ai/naqvihaider126-fast-nuces/sgd-regression-california-housing) to view the training metrics on Weights & Biases.

## Author
This model was developed by **Ali Haider** for the **Machine Learning for Robotics** course.

