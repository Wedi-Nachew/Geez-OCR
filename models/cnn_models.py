import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy

def build_simpler_cnn():
    """Builds a simple convolutional neural network"""
    model = Sequential([
        layers.Input(shape=(32,32,1)),
        
        # Normalizes the 0-225 pixels to optimal [-1.0,1.0]
        layers.Rescaling(scale=1./127.5, offset=-1.0),

        layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
        layers.MaxPool2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(287, activation="linear")
    ], name="Basic_CNN")
    
    model.compile(
        optimizer=Adam(0.0005),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy"), SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )

    return model

def build_deeper_cnn():
    """Builds a deeper convolutional neural network"""
    model = Sequential([
        layers.Input(shape=(32,32,1)),

        # Normalizes the 0-225 pixels to optimal [-1.0,1.0]
        layers.Rescaling(scale=1./127.5, offset=-1.0),

        layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
        layers.MaxPool2D((2,2)),
        
        layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"),
        layers.MaxPool2D((2,2)),


        layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", padding="same"),
        layers.MaxPool2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(287, activation="linear"),
    ], name = "Deeper_CNN_Model")

    model.compile(
        optimizer=Adam(0.0005),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy"), SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )
    
    return model

def build_regularized_deeper_cnn():
    """Builds a regularized version of the deeper convolutional neural network model"""
    l2_regularizer = tf.keras.regularizers.l2(1e-3)
    
    model = Sequential([
        layers.Input(shape=(32,32,1)),
        
        # Data Augmentation
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        # Normalizes the 0-225 pixels to optimal [-1.0,1.0]
        layers.Rescaling(scale=1./127.5, offset=-1.0),
        

        # Block 1: 32 filters
        layers.Conv2D(
            filters=32, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        
        layers.Conv2D(
            filters=64, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2)),
        layers.SpatialDropout2D(0.10),


        # Block 2: 64 filters
        layers.Conv2D(
            filters=64, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        
        layers.Conv2D(
            filters=64, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2)),
        layers.SpatialDropout2D(0.15),
        
        # Block 3: 128 filters
        layers.Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        
        layers.Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",            
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2)),
        layers.SpatialDropout2D(0.20),

        # Block 4: 128 filters with no pooling to extract deeper features
        layers.Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            kernel_initializer=tf.keras.initializers.HeNormal, 
            padding="same",
            kernel_regularizer=l2_regularizer
        ),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        
        # Dense Layers
        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128, 
            activation="relu", 
            kernel_initializer=tf.keras.initializers.HeNormal,
            kernel_regularizer=l2_regularizer
        ),
        layers.Dropout(0.4),
        
        
        layers.Dense(287, activation="linear"),
    ], name = "Regularized_Deeper_Model")

    model.compile(
        optimizer=Adam(2.5e-4),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="accuracy"), SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy")]
    )
    
    return model