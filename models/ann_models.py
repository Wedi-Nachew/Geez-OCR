import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense  
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import Dropout

def build_simpler_ann():
    """Builds a simple artificial neural network"""
    model = Sequential([
        Flatten(input_shape=(32,32)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(287, activation="linear")
    ], name="Basic ANN")

    model.compile(
       optimizer=Adam(0.01),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=["accuracy"]
    )
    
    return model

def build_deeper_ann():
   """Builds a deeper artificial neural network""" 
   model = Sequential([
       Flatten(input_shape=(32,32)),
       Dense(256, activation="relu"),
       Dense(128, activation="relu"),
       Dense(287, activation="linear")
   ], name="Deeper ANN")

   model.compile(
       optimizer=Adam(0.01),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=["accuracy"]
   )

   return model

def build_regularized_ann():
    """Builds a regularized version of the deeper ANN model"""
    model = Sequential([
       Flatten(input_shape=(32,32)),

       Dense(256, activation="relu"),
       Dropout(0.3),
       
       Dense(128, activation="relu"),
       Dropout(0.3),
       
       Dense(287, activation="linear")
   ], name="Regularized Deeper ANN")

    model.compile(
       optimizer=Adam(0.01),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=["accuracy"]
    )

    return model