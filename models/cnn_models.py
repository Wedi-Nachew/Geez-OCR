import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def build_simpler_cnn():
    """Builds a simple convolutional neural network"""
    model = Sequential([
        Input(shape=(32,32,1)),
        Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D((2,2)),
        
        Flatten(),
        Dense(64, activation="relu"),
        Dense(287, activation="linear")
    ], name="Basic_CNN")
    
    model.compile(
        optimizer=Adam(0.0005),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy")]
    )

    return model

def build_deeper_cnn():
    """Builds a deeper convolutional neural network"""
    model = Sequential([
        Input(shape=(32,32,1)),
        Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D((2,2)),
        
        Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
        MaxPool2D((2,2)),
        
        Flatten(),
        Dense(128, activation="relu"),
        Dense(287, activation="linear"),
    ], name = "Deeper_CNN_Model")

    model.compile(
        optimizer=Adam(0.0005),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy")]
    )
    
    return model

def build_regularized_deeper_cnn():
    """Builds a regularized version of the deeper convolutional neural network model"""
    model = Sequential([
        Input(shape=(32,32,1)),
        Conv2D(filters=32, kernel_size=(3,3), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding="same"),
        BatchNormalization(),
        tf.keras.layers.ReLU(),
        
        MaxPool2D((2,2)),

        Conv2D(filters=64, kernel_size=(3,3), kernel_regularizer=tf.keras.regularizers.l2(1e-4), padding="same"),
        BatchNormalization(),
        tf.keras.layers.ReLU(),

        MaxPool2D((2,2)),
        
        Flatten(),

        Dense(128, activation="relu"),
        Dropout(0.4),
        
        Dense(287, activation="linear"),
    ], name = "Regularized_Deeper_Model")

    model.compile(
        optimizer=Adam(0.0005),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy")]
    )
    
    return model