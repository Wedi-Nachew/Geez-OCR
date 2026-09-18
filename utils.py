import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        color_mode = "grayscale",
        image_size=(32,32),
        batch_size=1024, # Using a larger batch size to quickly load the images in chunks
        shuffle=False     
    )

    # Load the test data from the dataset files with inferred label
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        color_mode = "grayscale",
        image_size=(32,32),
        batch_size=1024, # Using a larger batch size to quickly load the images in chunks
        shuffle=False
    )

    X_train, y_train = extract_images_labels(train_ds)
    X_test, y_test = extract_images_labels(test_ds) 
    
    return X_train, y_train, X_test, y_test

def display_image(images, titles=None, suptitle=None, cols=10, img_shape=(32,32)):
    """Core rendering method any collection of 1D/2D image arrays"""
    m = len(images)
    if m == 0:
        print("No images to display")
        return
    
    rows = math.ceil(m / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.4), sequeeze=False)
    fig.tight_layout(pad=0.2, rect=[0, 0.03, 1, 0.93])
    
    axes_flat = axes.flatten()
    
    for i in range(m):
        ax = axes_flat[i]
        ax.imshow(images[i].reshape(img_shape), cmap="gray")
        if titles is not None:
            ax.set_title(str(titles[i]))
        ax.set_axis_off()
    
    # Hide any unused trailing subplots in a column if any
    for j in range(m, len(axes_flat)):
        axes_flat[j].set_axis_off()
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.show()

def show_images(X, Y, num_of_images=20, cols=10):
    """Show randomly selected images"""
    random_indices = np.random.choice(len(X), size=num_of_images, replace=False)
    titles = [f"{Y[i]}" for i in random_indices]

    display_image(
        images=X[random_indices],
        titles=titles,
        cols=cols
    )
