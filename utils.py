import numpy as np
import tensorflow as tf
from pathlib import Path

def extract_images_labels(dataset):
    """A helper function to extract images and labels from tf.data.Dataset"""
    # Iterate over the dataset and concatnate the batches directly 
    images_list = []
    labels_list = []

    for images, labels in dataset:
        images_list.append(images.numpy())
        labels_list.append(labels.numpy())
    
    # Combines batches into a single large Numpy array
    X = np.concatenate(images_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    return X, y

def load_data(dir):
    """Loads and preprocess the images and returns a numpy array of the train and test splits"""
    train_path = Path(f"{dir}/train")
    test_path = Path(f"{dir}/test")
  
    # Load the train data from the dataset files with inferred label
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="int",
        image_size=(32,32),
        batch_size=1024, # Using a larger batch size to quickly load the images in chunks
        shuffle=False     
    )

    # Load the test data from the dataset files with inferred label
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        image_size=(32,32),
        batch_size=1024, # Using a larger batch size to quickly load the images in chunks
        shuffle=False
    )

    X_train, y_train = extract_images_labels(train_ds)
    X_test, y_test = extract_images_labels(test_ds) 
    
    return X_train, y_train, X_test, y_test
