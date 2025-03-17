import numpy as np
from model3digitsJupiter import ThreeDigitLotteryPredictor as ModelJupiter
from model3digits import ThreeDigitLotteryPredictor as ModelRegular
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define CSV path
CSV_PATH = "playground/Lotto/lotto3digits.csv"

# Initialize models
model_jupiter = ModelJupiter()
model_regular = ModelRegular()

# Load data
data_jupiter = model_jupiter.load_data(CSV_PATH)
data_regular = model_regular.load_data(CSV_PATH)

# Prepare data
X_j, y_j = model_jupiter.prepare_data(data_jupiter)
X_r, y_r = model_regular.prepare_data(data_regular)

# Split data
X_train_j, X_test_j, y_train_j, y_test_j = train_test_split(X_j, y_j, test_size=0.2, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

# Build models
model_jupiter.build_model(input_shape=(X_j.shape[1], X_j.shape[2]))
model_regular.build_model(input_shape=(X_r.shape[1], X_r.shape[2]))

# Train models
model_jupiter.train(X_train_j, y_train_j, epochs=50, batch_size=32)
model_regular.train(X_train_r, y_train_r, epochs=50, batch_size=32)

# Make predictions
y_pred_j = model_jupiter.model.predict(X_test_j)
y_pred_r = model_regular.model.predict(X_test_r)

# Calculate MAE
mae_jupiter = mean_absolute_error(y_test_j, y_pred_j)
mae_regular = mean_absolute_error(y_test_r, y_pred_r)

# Print comparison results
print(f"Model Jupiter MAE: {mae_jupiter:.4f}")
print(f"Model Regular MAE: {mae_regular:.4f}")

# Predict next numbers
recent_data_j = X_test_j[-1]
recent_data_r = X_test_r[-1]

# Ensure predictions are numeric and positive
predicted_numbers_j = np.abs(np.round(np.array(model_jupiter.predict(recent_data_j), dtype=float)).astype(int))
predicted_numbers_r = np.abs(np.round(np.array(model_regular.predict(recent_data_r), dtype=float)).astype(int))

print(f"Predicted Numbers (Jupiter Model): {predicted_numbers_j}")
print(f"Predicted Numbers (Regular Model): {predicted_numbers_r}")
