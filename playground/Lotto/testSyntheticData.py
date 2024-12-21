import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LotteryNumberPredictor:
    def __init__(self):
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.label_scaler = MinMaxScaler()
    
    def generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic lottery data for training
        
        :param num_samples: Number of data points to generate
        :return: DataFrame with synthetic lottery data
        """
        # Simulate historical lottery data with some patterns
        np.random.seed(42)
        dates = pd.date_range(start='1/1/2020', periods=num_samples)
        
        # Create a base series with some randomness
        base_series = np.random.randint(0, 100, num_samples)
        
        # Add some cyclical patterns
        seasonal_pattern = np.sin(np.arange(num_samples) * 0.1) * 10
        trend_pattern = np.linspace(0, 5, num_samples)
        
        lottery_numbers = base_series + seasonal_pattern + trend_pattern
        lottery_numbers = lottery_numbers % 100  # Ensure numbers are between 0-99
        
        data = pd.DataFrame({
            'date': dates,
            'lottery_number': lottery_numbers.astype(int)
        })
        
        return data
    
    def prepare_data(self, historical_data, window_size=10):
        """
        Prepare data for training with sliding window
        
        :param historical_data: DataFrame with lottery numbers
        :param window_size: Number of previous draws to use as features
        :return: Prepared input features and labels
        """
        features = []
        labels = []
        
        for i in range(len(historical_data) - window_size):
            # Use previous 'window_size' draws as features
            window = historical_data['lottery_number'].values[i:i+window_size]
            features.append(window)
            labels.append(historical_data['lottery_number'].values[i+window_size])
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Fit the scalers on the features and labels
        self.feature_scaler.fit(features)
        self.label_scaler.fit(labels.reshape(-1, 1))
        
        features = self.feature_scaler.transform(features)
        labels = self.label_scaler.transform(labels.reshape(-1, 1)).flatten()
        
        return features, labels
    
    def build_model(self, input_shape):
        """
        Create neural network model for prediction
        
        :param input_shape: Shape of input features
        """
        # Ensure input_shape is a tuple
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Use Adam optimizer with learning rate scheduling
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.9
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer, 
            loss='mean_squared_error', 
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, epochs=200, batch_size=32, validation_split=0.2):
        """
        Train the neural network model
        
        :param X: Input features
        :param y: Target values
        :param epochs: Number of training epochs
        :param batch_size: Size of batches for training
        :param validation_split: Portion of data used for validation
        :return: Training history
        """
        # Ensure the save directory exists
        os.makedirs('models', exist_ok=True)
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True
        )
        
        # Model checkpoint to save best model (use .keras extension)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras', 
            monitor='val_loss', 
            save_best_only=True
        )
        
        # Train the model
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
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot training & validation MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set and print evaluation metrics
        
        :param X_test: Test input features
        :param y_test: Test target values
        """
        y_pred = self.model.predict(X_test)
        y_pred = y_pred.flatten()
        
        # Inverse transform the scaled predictions and true values
        y_test_original = self.label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = self.label_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)
        
        # Print evaluation metrics
        print("\nModel Evaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R-squared (R2) Score: {r2:.4f}")
    
    def predict(self, recent_data):
        """
        Predict lottery numbers based on recent data
        
        :param recent_data: Recent lottery draw numbers
        :return: Predicted lottery number
        """
        recent_data_scaled = self.feature_scaler.transform(recent_data.reshape(1, -1))
        prediction = self.model.predict(recent_data_scaled)
        predicted_number = self.label_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
        predicted_number = int(round(predicted_number % 100))
        return f"{predicted_number:02d}"

def main():
    # For development, use synthetic data
    # In production, you would load your own data
    predictor = LotteryNumberPredictor()
    data = predictor.generate_synthetic_data()
    
    # Prepare data for training
    X, y = predictor.prepare_data(data, window_size=10)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build the model - pass in the correct input shape
    predictor.build_model(input_shape=X_train.shape[1])
    
    # Train the model
    history = predictor.train(X_train, y_train)
    
    # Plot training history
    predictor.plot_training_history(history)
    
    # Evaluate the model on the test set
    predictor.evaluate_model(X_test, y_test)
    
    # Example prediction
    recent_data = X_test[-1]  # Use last window of test data
    predicted_number = predictor.predict(recent_data)
    print(f"Predicted Lottery Number: {predicted_number}")

if __name__ == "__main__":
    main()