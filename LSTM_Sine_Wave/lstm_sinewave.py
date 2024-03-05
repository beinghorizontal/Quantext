import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

# Create a sine wave dataset with noise
np.random.seed(42)

t = np.arange(0, 100, 0.1)  # time vector
sin_wave = np.sin(t)

# Add noise to the sine wave
noise = 0.1 * np.random.normal(size=len(t))
sin_wave_noisy = sin_wave + noise

# Plot the clean and noisy sine wave
plt.figure(figsize=(10, 6))
plt.plot(t, sin_wave, label='Clean Sine Wave')
plt.plot(t, sin_wave_noisy, label='Noisy Sine Wave', alpha=0.6)
plt.title('Sine Wave with Noise')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# df_noisy = pd.DataFrame(sin_wave_noisy)
# df_noisy['target'] = (df_noisy[0].shift(-1) > df_noisy[0]).astype(int)
# df_clean = pd.DataFrame(sin_wave)
# df_clean['target'] = (df_clean[0].shift(-1) > df_clean[0]).astype(int)

# Create the target variable indicating whether the next wave will go up (1) or down (0)
target = np.where(sin_wave_noisy[1:] > sin_wave_noisy[:-1], 1, 0)

# Reshape the data for LSTM input (sequence_length, features)
sequence_length = 10
X = np.array([sin_wave_noisy[i:i+sequence_length] for i in range(len(sin_wave_noisy)-sequence_length)])
y = target[sequence_length-1:]
# Split the dataset into training and validation sets
split_index = int(0.7 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Basic LSTM network
model = Sequential()
model.add(LSTM(8, return_sequences=False, input_shape=(sequence_length, 1), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)


# Define the LSTM model
# model = Sequential()
# model.add(LSTM(32, return_sequences=True, input_shape=(sequence_length, 1), activation='relu'))
# model.add(LSTM(16, return_sequences=True, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(LSTM(32, return_sequences=True, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(LSTM(32, return_sequences=True, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(Dropout(0.4))
# # model.add(LSTM(16, return_sequences=True, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(LSTM(8, return_sequences=True, activation='relu'))
# # model.add(Dropout(0.4))
# # # model.add(LSTM(16, return_sequences=True, activation='relu'))
# model.add(LSTM(8, return_sequences=True, activation='relu'))
# model.add(LSTM(8))  # Last LSTM layer (no return_sequences)
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

# For regression problem As opposed to classification problem where we want to predict the next outcome
# as a number instead of binary bullish or bearish
# we use the activation function in the final dense layer  = 'Linear' So the values will be in the numbers


# Compile the model


# For the regression problem we use the loss function equal to mean squared error or MSE
# The loss function quantifies the difference between the predicted outputs of a neural network 
# and the actual target values.
#And for the metrics you can use mean absolute error MAE or mean absolute percentage error or MAPE


# Visualize model network

early_stopping = EarlyStopping(monitor='val_loss',    # Monitor the val loss
                               patience=150,           # Num epochs; if no improvement training will be stopped
                               verbose=1,
                               mode='min',            # The training will stop when the quantity monitored has stopped decreasing
                               restore_best_weights=True) # Restores model weights from the epoch with the best value of the monitored quantity.

# Model checkpoint callback
model_checkpoint = ModelCheckpoint('d:/demos/best_model.keras',   # Path where to save the model
                                   monitor='val_loss',   # Monitor the validation loss
                                   save_best_only=True,  # The latest best model according to the quantity monitored will not be overwritten
                                   mode='min',           # The training will save the model when the quantity monitored has decreased
                                   verbose=1)

# Add callbacks to the fit function
history = model.fit(X_train, y_train,
                    epochs=1000,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1)

# # Train the model
# history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val))

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# // to reuse the model
from keras.models import load_model
model = load_model('d:/demos/my_model.keras')
