import tensorflow as tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

model = Sequential(
    [
        Dense(units = 16, input_shape=(1,), activation = 'relu'),
        Dense(units = 32, activation = 'relu'),
        Dense(units = 2, activation = 'softmax')
    ]
)

model.summary()