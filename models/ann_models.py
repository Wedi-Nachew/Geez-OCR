import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Rescaling, ReLU, LayerNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy

def build_simpler_ann():
    """Builds a simple artificial neural network"""
    model = Sequential([
        Input(shape=(32,32,1)),

        # Normalizes the 0-225 pixels to optimal [-1.0,1.0]
        Rescaling(scale=1./127.5, offset=-1.0),

        Flatten(),
        Dense(512, activation="relu"),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(287, activation="linear")
    ], name="Basic_ANN")

    model.compile(
       optimizer=Adam(0.0005),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=[SparseCategoricalAccuracy(name="accuracy"), SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )
    
    return model

def build_deeper_ann():
   """Builds a deeper artificial neural network""" 
   model = Sequential([
       Input(shape=(32, 32, 1)),

        Rescaling(scale=1./127.5, offset=-1.0),

        Flatten(),

        Dense(
            1024,
            kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        LayerNormalization(),
        ReLU(),

        Dense(
            512,
            kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        LayerNormalization(),
        ReLU(),
        
        Dense(
            256,
            kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        LayerNormalization(),
        ReLU(),

        Dense(
            128,
            kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        LayerNormalization(),
        ReLU(),

        Dense(287)

   ], name="Deeper_ANN")

   model.compile(
       optimizer=Adam(0.0005),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=[SparseCategoricalAccuracy(name="accuracy"), SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
   )

   return model

def build_regularized_deeper_ann():
    """Builds a regularized version of the deeper ANN model"""
    model = Sequential([
       Input(shape=(32,32,1)),
       
       # Normalizes the 0-225 pixels to optimal [-1.0,1.0]
       Rescaling(scale=1./127.5, offset=-1.0),
       
       Flatten(),


       Dense(
           1024, 
           kernel_initializer=tf.keras.initializers.HeNormal()
        ),
       LayerNormalization(),
       ReLU(),

       Dense(
           512, 
           kernel_initializer=tf.keras.initializers.HeNormal()
        ),
       LayerNormalization(),
       ReLU(),
       Dropout(0.2),

       Dense(
           256, 
           kernel_initializer=tf.keras.initializers.HeNormal()
        ),
       LayerNormalization(),
       ReLU(),
       Dropout(0.3),
       
       Dense(
           128, 
           kernel_initializer=tf.keras.initializers.HeNormal()
        ),
       LayerNormalization(),
       ReLU(),
       Dropout(0.3),
       
       Dense(287, activation="linear")
   ], name="Regularized_Deeper_ANN")

    model.compile(
       optimizer=Adam(0.0005),
       loss=SparseCategoricalCrossentropy(from_logits=True),
       metrics=[SparseCategoricalAccuracy(name="accuracy"), SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )

    return model