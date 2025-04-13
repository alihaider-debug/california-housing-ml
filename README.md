# SGD Regression Model – California Housing Dataset 🏡

This is a regression model trained using Stochastic Gradient Descent (SGDRegressor) on the California Housing dataset. The model uses polynomial features, L2 regularization, and early stopping to optimize performance.

## Model Details

- Algorithm: SGDRegressor (Stochastic Gradient Descent)
- Regression Type: Polynomial Regression (degree = 2)
- Regularization: L2 (Ridge)
- Early Stopping: Manual, based on validation loss
- Loss Function: Mean Squared Error (MSE)
- Libraries: scikit-learn, joblib

## Dataset

- Source: fetch_california_housing() from scikit-learn
- Features Used: 2 numerical features (simplified)
- Target: Median house value in California districts

## Files Included

- model.pkl: Trained SGDRegressor model
- scaler.pkl: StandardScaler for normalization
- poly.pkl: PolynomialFeatures transformer (degree 2)

## Usage

Use huggingface_hub and joblib to download and load the model, scaler, and transformer. Input should be a 2-feature list, which is scaled and transformed before prediction.

## Evaluation

- MSE: (0.54321)
- RMSE: (0.7365)
- R^2: (0.8457)

## License

MIT License

## Weights & Biases Dashboard

📈 [Click here to view training metrics on W&B](https://wandb.ai/naqvihaider126-fast-nuces/sgd-housing-regression)


## Author

This model was developed by Ali Haider for the Machine Learning for Robotics course.
