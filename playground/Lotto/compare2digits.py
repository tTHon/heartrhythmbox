import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
import random

# Import models from the provided files
from model2 import ImprovedLotteryPredictor as ModelDateNumber
from model2withZodiac import ZodiacLotteryPredictor as ModelDateNumberZodiac

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def evaluate_model(y_true, y_pred):
    """
    Calculate and return model evaluation metrics.
    
    Parameters:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    
    Returns:
    dict: Dictionary with evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2
    }

def main():
    # Define CSV paths for each model
    csv_paths = {
        'ModelDateNumber': r'playground/Lotto/lotteryLast30Yrs.csv',
        'ModelDateNumberZodiac': r'playground/Lotto/lotteryLast30YrsJupiter.csv'
    }
    
    # Initialize models
    models = {
        'ModelDateNumber': ModelDateNumber(),
        'ModelDateNumberZodiac': ModelDateNumberZodiac()
    }
    
    results = {}
    predicted_numbers = {}
    
    for model_name, model in models.items():
        try:
            # Load and prepare data
            data = model.load_data(csv_paths[model_name])
            if model_name == 'ModelDateNumber':
                X, y = model.prepare_data(data, window_size=10)
            else:
                X, y = model.prepare_data(data)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Build and train the model
            if model_name == 'ModelDateNumber':
                model.build_lstm_model(input_shape=(X_train.shape[1], 1))
            else:
                model.build_lstm_model(input_shape=(X_train.shape[1], 3))
            
            model.train(X_train, y_train, epochs=50, batch_size=32)
            
            # Make predictions on the test set
            y_pred = model.model.predict(X_test)
            
            # Ensure predicted numbers are not negative
            y_pred = np.maximum(y_pred, 0)
            
            # Evaluate the model
            metrics = evaluate_model(y_test, y_pred)
            results[model_name] = metrics
            
            # Predict a sample lottery number
            if model_name == 'ModelDateNumberZodiac':
                recent_data = X_test[-1][:, 0]
                recent_zodiac = 'Gemini'  # example zodiac sign
                combined_features = np.column_stack((recent_data, [model.zodiac_encoder.transform([recent_zodiac])[0]] * len(recent_data), np.zeros(len(recent_data))))
                predicted_number = model.predict(combined_features, recent_zodiac)
            else:
                recent_data = X_test[-1]
                predicted_number = model.predict(recent_data)
            
            # Ensure predicted numbers are numeric, not negative, and handle NaN values
            predicted_number = np.array(predicted_number, dtype=float)  # Ensure numeric
            predicted_number = np.nan_to_num(predicted_number, nan=0)  # Replace NaN with 0
            predicted_number = np.maximum(predicted_number, 0)  # Ensure non-negative
            predicted_number = np.round(predicted_number).astype(int)  # Convert to integers
            
            predicted_numbers[model_name] = predicted_number
            
            print(f"\n{model_name} Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print(f"Predicted Lottery Number: {predicted_number}")
        
        except Exception as e:
            print(f"An error occurred with {model_name}: {e}")
    
    # Create a summary table
    summary_table = pd.DataFrame(results).T
    summary_table['Predicted Number'] = pd.Series(predicted_numbers)
    summary_table.columns = ['MAE', 'RMSE', 'MSE', 'R2', 'Predicted Number']
    print("\nSummary Table:")
    print(summary_table)
    
    # Convert summary table to CSV format and print it
    csv_output = summary_table.to_csv(index=True)
    print("\nCSV Format for Excel:")
    print(csv_output)

if __name__ == "__main__":
    main()
