import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import random

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

class ImprovedLotteryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()

    def load_data(self, csv_path, number_column='lottery_number'):
        try:
            data = pd.read_csv(csv_path)
            if number_column not in data.columns:
                raise ValueError(f"Column '{number_column}' not found in the CSV file. Available columns are: {list(data.columns)}")
            
            data[number_column] = pd.to_numeric(data[number_column], errors='coerce')
            data = data.dropna(subset=[number_column])
            
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data = data.sort_values('date')
            
            return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def prepare_data(self, historical_data, number_column='lottery_number', window_size=10):
        numbers = historical_data[number_column].values
        scaled_numbers = self.scaler.fit_transform(numbers.reshape(-1, 1)).flatten()
        features, labels = [], []
        for i in range(len(scaled_numbers) - window_size):
            features.append(scaled_numbers[i:i + window_size])
            labels.append(scaled_numbers[i + window_size])
        return np.array(features), np.array(labels)

    def build_lstm_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mean_squared_error',
                      metrics=['mae'])
        self.model = model
        return model

    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        os.makedirs('models', exist_ok=True)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_lstm_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.show()

    def predict(self, recent_data):
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        recent_data = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()
        prediction = self.model.predict(recent_data.reshape(1, -1))
        predicted_number = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        return round(predicted_number)

def evaluate_model(y_true, y_pred, scaler):
    """
    Calculate and print model evaluation metrics.
    
    Parameters:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    scaler (MinMaxScaler): Scaler used to inverse transform the data
        """
    # Inverse transform the scaled predictions and true values
    y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)
    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")



# Example usage in main function
def main():
    CSV_PATH = 'playground/Lotto/lotteryLast30Yrs.csv'
    predictor = ImprovedLotteryPredictor()

    try:
        data = predictor.load_data(CSV_PATH)
        X, y = predictor.prepare_data(data, window_size=10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        predictor.build_lstm_model(input_shape=(X.shape[1], 1))
        history = predictor.train(X_train, y_train, epochs=50, batch_size=32)
        predictor.plot_training_history(history)
        
        # Make predictions on the test set
        y_pred = predictor.model.predict(X_test)

        # Evaluate the model
        evaluate_model(y_test, y_pred, predictor.scaler)

        recent_data = X_test[-1]
        predicted_number = predictor.predict(recent_data)
        print(f"Predicted Lottery Number: {predicted_number}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
