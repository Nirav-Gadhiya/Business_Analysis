import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (adjust the file path)
data = pd.read_csv("D:/Nirav/business_analysis/generated_data.csv")

# Ensure that the date column is in datetime format
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')

# Extract useful features from the date (Year, Month, Day)
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Drop the original date column (as it's not directly needed for the model)
data.drop(columns=['date'], inplace=True)

# Feature Engineering (if required)
data.fillna(0, inplace=True)  # Handle missing values by filling with 0

# Define the features (X) and target (y)
X = data[["expense", "employee_salary", "amount_earned", "total_profit", "year", "month", "day"]]
y = data["quantity_sold"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (important for some models, not essential for RandomForest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Optional: Save the model if needed
import joblib
joblib.dump(model, 'random_forest_model_with_expense_salary.pkl')
