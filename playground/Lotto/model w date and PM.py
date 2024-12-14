import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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

class DateAndPMIncorporatedLotteryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.date_scaler = MinMaxScaler()
        self.pm_encoder = LabelEncoder()

    def load_data(self, csv_path, number_column='lottery_number', pm_column='prime_minister'):
        try:
            data = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = [number_column, 'date', pm_column]
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in the CSV file. Available columns are: {list(data.columns)}")
            
            # Convert lottery number to numeric
            data[number_column] = pd.to_numeric(data[number_column], errors='coerce')
            data = data.dropna(subset=[number_column])
            
            # Process date
            data['date'] = pd.to_datetime(data['date'])
            data['date_ordinal'] = data['date'].map(lambda x: x.toordinal())
            
            # Encode prime minister names
            data['pm_encoded'] = self.pm_encoder.fit_transform(data[pm_column])
            
            # Sort by date
            data = data.sort_values('date')
            
            return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def prepare_data(self, historical_data, number_column='lottery_number', window_size=10):
        numbers = historical_data[number_column].values
        dates = historical_data['date_ordinal'].values
        pms = historical_data['pm_encoded'].values
        
        # Scale features
        scaled_numbers = self.scaler.fit_transform(numbers.reshape(-1, 1)).flatten()
        scaled_dates = self.date_scaler.fit_transform(dates.reshape(-1, 1)).flatten()
        
        features, labels = [], []
        for i in range(len(scaled_numbers) - window_size):
            number_window = scaled_numbers[i:i + window_size]
            date_window = scaled_dates[i:i + window_size]
            pm_window = pms[i:i + window_size]
            
            # Combine all features: lottery numbers, dates, and prime ministers
            combined_features = np.column_stack((number_window, date_window, pm_window))
            features.append(combined_features)
            labels.append(scaled_numbers[i + window_size])
        
        return np.array(features), np.array(labels)

    def build_lstm_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
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
            'models/best_date_pm_lstm_model.keras',
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

    def predict(self, recent_data, recent_dates, recent_pms):
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        scaled_numbers = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()
        scaled_dates = self.date_scaler.transform(recent_dates.reshape(-1, 1)).flatten()
        
        # Ensure PM names are encoded consistently
        scaled_pms = self.pm_encoder.transform(recent_pms)
        
        combined_features = np.column_stack((scaled_numbers, scaled_dates, scaled_pms))
        
        prediction = self.model.predict(combined_features.reshape(1, -1, 3))
        predicted_number = self.scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        
        # Constrain to two digits using modulo operation
        return f"{int(round(predicted_number) % 100):02d}"

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
    CSV_PATH = 'lottery_data w PM.csv'
    predictor = DateAndPMIncorporatedLotteryPredictor()

    try:
        # Load and prepare data
        data = predictor.load_data(CSV_PATH)
        X, y = predictor.prepare_data(data, window_size=10)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train the model
        predictor.build_lstm_model(input_shape=(X.shape[1], 3))
        history = predictor.train(X_train, y_train, epochs=50, batch_size=32)
        
        # Plot training history
        predictor.plot_training_history(history)
        
        # Make predictions on the test set
        y_pred = predictor.model.predict(X_test)
        
        # Evaluate the model
        evaluate_model(y_test, y_pred, predictor.scaler)
        
        # Predict a sample lottery number
        recent_data = X_test[-1][:, 0]
        recent_dates = X_test[-1][:, 1]
        recent_pms = predictor.pm_encoder.inverse_transform(X_test[-1][:, 2].astype(int))
        
        predicted_number = predictor.predict(recent_data, recent_dates, recent_pms)
        print(f"Predicted Lottery Number: {predicted_number}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
       

if __name__ == "__main__":
    main()