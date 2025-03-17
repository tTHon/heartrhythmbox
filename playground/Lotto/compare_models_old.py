import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import random
import matplotlib as mpl

# Import models from the provided files
from model2 import ImprovedLotteryPredictor as Model3
from model2withZodiac import ZodiacLotteryPredictor as Model5

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
        'Model3': r'playground/Lotto/lotteryLast30Yrs.csv',
        'Model5': r'playground/Lotto/lotteryLast30YrsJupiter.csv'
    }
    
    # Initialize models
    models = {
        'Model3': Model3(),
        'Model5': Model5()
    }
    
    results = {}
    predicted_numbers = {}
    histories = {}
    
    # Set font properties
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Inter']
    mpl.rcParams['font.size'] = 16
    
    # Ensure the save directory exists
    save_dir = r'playground/Lotto/results/autogenerate'
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, model in models.items():
        try:
            # Load and prepare data
            data = model.load_data(csv_paths[model_name])
            X, y = model.prepare_data(data, window_size=10)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Build and train the model
            if model_name == 'Model3':
                model.build_lstm_model(input_shape=(X_train.shape[1], 1))
            elif model_name == 'Model5':
                model.build_lstm_model(input_shape=(X_train.shape[1], 3))
            
            history = model.train(X_train, y_train, epochs=50, batch_size=32)
            histories[model_name] = history
            
            # Make predictions on the test set
            y_pred = model.model.predict(X_test)
            
            # Ensure predicted numbers are not negative
            y_pred = np.maximum(y_pred, 0)
            
            # Evaluate the model
            metrics = evaluate_model(y_test, y_pred)
            results[model_name] = metrics
            
            # Predict a sample lottery number
            if model_name == 'Model5':
                recent_data = X_test[-1][:, 0]
                recent_zodiac = 'Gemini'  # example zodiac sign
                combined_features = np.column_stack((recent_data, [model.zodiac_encoder.transform([recent_zodiac])[0]] * len(recent_data), np.zeros(len(recent_data))))
                predicted_number = model.predict(combined_features, recent_zodiac)
            else:
                recent_data = X_test[-1]
                predicted_number = model.predict(recent_data)
            
            # Ensure predicted numbers are not negative
            predicted_number = np.maximum(predicted_number, 0)
            
            predicted_numbers[model_name] = predicted_number
            
            print(f"\n{model_name} Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            print(f"Predicted Lottery Number: {predicted_number}")
            
            # Remove plot drawing section
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(history.history['mae'], label='Train MAE')
            # ax.plot(history.history['val_mae'], label='Validation MAE')
            # ax.axhline(y=1, color='lightgrey', linestyle='--', label='MAE = 1')
            # if model_name == 'Model3':
            #     ax.set_title('Model 2, a more rigorous model')
            # elif model_name == 'Model5':
            #     ax.set_title('Model 2 with Jupiter Transits Data')
            # ax.set_ylabel('Mean Absolute Error')
            # ax.legend()
            # plt.xlabel('Epochs')
            # plt.tight_layout()
            # plt.savefig(os.path.join(save_dir, f'{model_name}_mae_by_epochs.png'), transparent=True)
            # plt.show()
        
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

