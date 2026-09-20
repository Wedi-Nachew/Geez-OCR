import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def build_simpler_ann():
    """Builds a simple artificial neural network"""
    model = Sequential([
        Input(shape=(32,32)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(287, activation="linear")
    ], name="Basic_ANN")

    model.compile(
       optimizer=Adam(0.01),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=["accuracy"]
    )
    
    return model

def build_deeper_ann():
   """Builds a deeper artificial neural network""" 
   model = Sequential([
       Input(shape=(32,32)),
       Flatten(),
       Dense(256, activation="relu"),
       Dense(128, activation="relu"),
       Dense(287, activation="linear")
   ], name="Deeper_ANN")

   model.compile(
       optimizer=Adam(0.01),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=["accuracy"]
   )

   return model

def build_regularized_deeper_ann():
    """Builds a regularized version of the deeper ANN model"""
    model = Sequential([
       Input(shape=(32,32)),
       Flatten(),

       Dense(256, activation="relu"),
       Dropout(0.3),
       
       Dense(128, activation="relu"),
       Dropout(0.3),
       
       Dense(287, activation="linear")
   ], name="Regularized_Deeper_ANN")

    model.compile(
       optimizer=Adam(0.01),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=["accuracy"]
    )

    return model