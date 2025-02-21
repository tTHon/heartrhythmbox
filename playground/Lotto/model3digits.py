import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import random

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

class ThreeDigitLotteryPredictor:
    def __init__(self):
        self.model = None
        self.number_scaler = MinMaxScaler()
        self.zodiac_encoder = LabelEncoder()
        self.date_scaler = MinMaxScaler()

    def load_data(self, csv_path, number1_column='number1', number2_column='number2', zodiac_column='zodiac'):
        try:
            data = pd.read_csv(csv_path)

            # Check required columns
            required_columns = [number1_column, number2_column, zodiac_column, 'date', 'month', 'year']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in the CSV file. Available columns are: {list(data.columns)}")

            # Convert number columns to numeric
            data[number1_column] = pd.to_numeric(data[number1_column], errors='coerce')
            data[number2_column] = pd.to_numeric(data[number2_column], errors='coerce')
            data = data.dropna(subset=[number1_column, number2_column, zodiac_column])

            # Encode zodiac signs
            data['zodiac_encoded'] = self.zodiac_encoder.fit_transform(data[zodiac_column])

            # Combine date components into a single datetime column
            data['date'] = pd.to_datetime(data[['year', 'month', 'date']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
            data = data.sort_values('date')

            # Extract date features
            data['day'] = data['date'].dt.day
            data['month'] = data['date'].dt.month
            data['year'] = data['date'].dt.year

            return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def prepare_data(self, historical_data, number1_column='number1', number2_column='number2', zodiac_column='zodiac_encoded', window_size=10):
        # Prepare number data
        number1 = historical_data[number1_column].values
        number2 = historical_data[number2_column].values
        scaled_number1 = self.number_scaler.fit_transform(number1.reshape(-1, 1)).flatten()
        scaled_number2 = self.number_scaler.fit_transform(number2.reshape(-1, 1)).flatten()

        # Prepare zodiac data
        zodiacs = historical_data[zodiac_column].values

        # Prepare date data
        dates = historical_data[['day', 'month', 'year']].values
        scaled_dates = self.date_scaler.fit_transform(dates)

        features, labels = [], []
        for i in range(len(scaled_number1) - window_size):
            # Combine number, zodiac, date, and additional features
            number1_window = scaled_number1[i:i + window_size]
            number2_window = scaled_number2[i:i + window_size]
            zodiac_window = zodiacs[i:i + window_size]
            date_window = scaled_dates[i:i + window_size]
            combined_features = np.column_stack((number1_window, number2_window, zodiac_window, date_window))

            features.append(combined_features)
            labels.append((scaled_number1[i + window_size], scaled_number2[i + window_size]))

        return np.array(features), np.array(labels)

    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')  # Two outputs for number1 and number2
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

        # Scale number and prepare zodiac
        recent_number1 = self.number_scaler.transform(recent_data[:, 0].reshape(-1, 1)).flatten()
        recent_number2 = self.number_scaler.transform(recent_data[:, 1].reshape(-1, 1)).flatten()
        recent_zodiac = recent_data[:, 2]
        recent_dates = self.date_scaler.transform(recent_data[:, 3:])

        # Combine features
        combined_features = np.column_stack((recent_number1, recent_number2, recent_zodiac, recent_dates))

        # Predict
        prediction = self.model.predict(combined_features.reshape(1, -1, combined_features.shape[1]))
        predicted_numbers = self.number_scaler.inverse_transform(prediction)

        # Format predicted numbers as 3 digits
        predicted_numbers = [f"{int(num):03d}" for num in predicted_numbers.flatten()]

        return predicted_numbers

def evaluate_model(y_true, y_pred, scaler):
    """
    Calculate and print model evaluation metrics.
    
    Parameters:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    scaler (MinMaxScaler): Scaler used to inverse transform the data
    """
    # Inverse transform the scaled predictions and true values
    y_true_original = scaler.inverse_transform(y_true)
    y_pred_original = scaler.inverse_transform(y_pred)

    # Calculate metrics
    mae = mean_absolute_error(y_true_original, y_pred_original)
    mse = mean_squared_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_original, y_pred_original)

    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

# Example usage in main function
def main():
    CSV_PATH = r'playground/Lotto/lotto3digits.csv'
    predictor = ThreeDigitLotteryPredictor()

    try:
        data = predictor.load_data(CSV_PATH)
        X, y = predictor.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
        history = predictor.train(X_train, y_train, epochs=50, batch_size=32)
        predictor.plot_training_history(history)

        # Make predictions on the test set
        y_pred = predictor.model.predict(X_test)

        # Evaluate the model
        evaluate_model(y_test, y_pred, predictor.number_scaler)

        recent_data = X_test[-1]
        predicted_numbers = predictor.predict(recent_data)
        print(f"Predicted Lottery Numbers: {predicted_numbers}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
