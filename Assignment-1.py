import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

df = pd.read_csv("dataset/insurance.csv")

print("First Five Records:\n")
print(df.head())
print("\nDataset Information:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())
print("\nNumerical Features:")
print(df.select_dtypes(include=['int64', 'float64']).columns.tolist())

print("\nCategorical Features:")
print(df.select_dtypes(include=['object', 'string']).columns.tolist())
print("\nTarget Variable:")
print("charges")

df = pd.get_dummies(df, drop_first=True)

print("\nDataset After Encoding:\n")
print(df.head())

X = df.drop("charges", axis=1)

y = df["charges"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data :", X_test.shape)

model = LinearRegression()

print("\nLinear Regression model created successfully.")

model.fit(X_train, y_train)

print("Model training completed successfully.")

y_pred = model.predict(X_test)

print("\nFirst 10 Predicted Charges:\n")
print(y_pred[:10])

comparison = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": y_pred
})

print("\nActual vs Predicted (First 10 Rows):\n")
print(comparison.head(10))

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Evaluation ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

#actual vs predicted graph

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.title("Actual vs Predicted Insurance Charges")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()