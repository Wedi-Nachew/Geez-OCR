import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import Dropout

def build_simpler_cnn():
    """Builds a simple convolutional neural network"""
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same", input_shape=(32,32)),
        MaxPool2D((2,2)),
        
        Flatten(),
        Dense(64, activation="relu"),
        Dense(287, activation="linear")
    ], name="Basic CNN")
    
    model.compile(
        optimizer=Adam(0.01),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model

def build_deeper_cnn():
    """Builds a deeper convolutional neural network"""
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same", input_shape=(32,32)),
        MaxPool2D((2,2)),
        
        Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D((2,2)),
        
        Flatten(),
        Dense(128, activation="relu"),
        Dense(287, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(0.01),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    return model

def build_regularized_deeper_cnn():
    """Builds a regularized version of the deeper convolutional neural network model"""
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3,3), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding="same", input_shape=(32,32)),
        BatchNormalization(),
        tf.keras.layers.Relu(),
        
        MaxPool2D((2,2)),

        Conv2D(filters=64, kernel_size=(3,3), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding="same"),
        BatchNormalization(),
        tf.keras.layers.Relu(),

        MaxPool2D((2,2)),
        
        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.4),
        
        Dense(287, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(0.01),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    return model