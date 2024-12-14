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

class LotteryNumberPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
    
    def load_data(self, csv_path, number_column='lottery_number'):
        """
        Load lottery data from a CSV file
        
        :param csv_path: Path to the CSV file
        :param number_column: Name of the column with lottery numbers
        :return: DataFrame with lottery data
        """
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
        """
        Prepare data for training with sliding window
        
        :param historical_data: DataFrame with lottery numbers
        :param number_column: Name of the column with lottery numbers
        :param window_size: Number of previous draws to use as features
        :return: Prepared input features and labels
        """
        numbers = historical_data[number_column].values
        features = []
        labels = []
        for i in range(len(numbers) - window_size):
            window = numbers[i:i+window_size]
            features.append(window)
            labels.append(numbers[i+window_size])
        return np.array(features), np.array(labels)
    
    def build_model(self, input_shape):
        """
        Create neural network model for prediction
        
        :param input_shape: Shape of input features
        """
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        self.model = model
        return model
    
    def train(self, X, y, epochs=100, batch_size=16, validation_split=0.2):
        """
        Train the neural network model
        
        :param X: Input features
        :param y: Target values
        :param epochs: Number of training epochs
        :param batch_size: Size of batches for training
        :param validation_split: Portion of data used for validation
        :return: Training history
        """
        os.makedirs('models', exist_ok=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras', 
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
        """
        Plot training and validation loss
        
        :param history: Training history from model.fit()
        """
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
        """
        Predict lottery numbers based on recent data
        
        :param recent_data: Recent lottery draw numbers
        :return: Predicted lottery number
        """
        prediction = self.model.predict(recent_data.reshape(1, -1))
        predicted_number = int(round(prediction[0][0] % 100))
        return f"{predicted_number:02d}"
    
def evaluate_model(y_true, y_pred, scaler):
    """
    Calculate and print model evaluation metrics.
    
    Parameters:
    y_true (array-like): True target values
    y_pred (array-like): Predicted target values
    scaler (MinMaxScaler): Scaler used to inverse transform the data
        """
    # Flatten predictions
    y_pred_original = y_pred.flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred_original)
    r2 = r2_score(y_true, y_pred_original)
    mse = mean_squared_error(y_true, y_pred_original)
    rmse = np.sqrt(mse)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")




def main():
    CSV_PATH = 'lottery_data.csv'
    predictor = LotteryNumberPredictor()
    try:
        data = predictor.load_data(CSV_PATH)
        X, y = predictor.prepare_data(data)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Prepare the model, scale the features
        predictor.build_model(input_shape=X_train.shape[1])
        history = predictor.train(X_train, y_train, epochs=100, batch_size=16)
        predictor.plot_training_history(history)
        
        # Make predictions on the test set
        y_pred = predictor.model.predict(X_test)
        
        # Evaluate the model
        evaluate_model(y_test, y_pred, predictor.scaler)
        
        # Predict a lottery number using the last sequence
        recent_data = X_test[-1]
        predicted_number = predictor.predict(recent_data)
        print(f"Predicted Lottery Number: {predicted_number}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
