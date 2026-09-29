import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from constants import GEEZ_CHARACTERS
from sklearn.model_selection import StratifiedKFold

# Prevents tensorflow from treating the folder names as strings and sorting them 
# alphabetically("0", "1", "10"..) and forces it to load the folders in exact numerical order
ordered_directory_names = [str(i) for i in range(len(GEEZ_CHARACTERS))] 

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
        shuffle=False,     
        class_names=ordered_directory_names
    )

    # Load the test data from the dataset files with inferred label
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        labels="inferred",
        label_mode="int",
        color_mode = "grayscale",
        image_size=(32,32),
        batch_size=1024, # Using a larger batch size to quickly load the images in chunks
        shuffle=False,
        class_names=ordered_directory_names
    )

    # Extract image tensors and labels
    X_train, y_train = extract_images_labels(train_ds)
    X_test, y_test = extract_images_labels(test_ds) 

    # Extract original file paths
    train_paths = np.array(train_ds.file_paths)
    test_paths = np.array(test_ds.file_paths)
    
    return (X_train, y_train, train_paths), (X_test, y_test, test_paths)

def cross_validate_model(build_model, X, y, epochs=10, batch_size=64, callback=None):
    if callback is None:
        callbacks_list = []
    elif isinstance(callback, list):
        callbacks_list = callback
    else:
        callbacks_list = [callback]
    
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []
    
    for fold, (train_idx, cv_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):
        print("\n","="*21, "Fold", fold,  "="*21)

        X_fold_train = X[train_idx]
        X_fold_cv = X[cv_idx]
        
        y_fold_train = y[train_idx] 
        y_fold_cv = y[cv_idx]
        
        # build the model
        model = build_model()
        
        history = model.fit(
            X_fold_train,
            y_fold_train,
            
            validation_data=(
                X_fold_cv,
                y_fold_cv
            ),
            
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks = callbacks_list
        )

        _, train_accuracy, train_top5_accuracy = model.evaluate(
            X_fold_train,
            y_fold_train,
            verbose=1
        )

        _, cv_accuracy, cv_top5_accuracy = model.evaluate(
            X_fold_cv,
            y_fold_cv,
            verbose=1
        )

        results.append({
            "fold": fold,
            "train_accuracy": train_accuracy,
            "cv_accuracy": cv_accuracy,
            "gap": train_accuracy - cv_accuracy,
            "train_top5_accuracy": train_top5_accuracy,
            "cv_top5_accuracy": cv_top5_accuracy
        })

        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Cross validation accuracy: {cv_accuracy:.4f}")
        print(f"Train-Validation gap: {train_accuracy - cv_accuracy:.4f}")
        print(f"Train top 5 accuracy: {train_top5_accuracy:.4f}")
        print(f"Cross top 5 accuracy: {cv_top5_accuracy:.4f}")
    
    return results

def summarize_results(results):
    cv_scores = [result['cv_accuracy'] for result in results]
    train_scores = [result['train_accuracy'] for result in results]
    gaps = [result["gap"] for result in results]
    train_top5_scores = [result["train_top5_accuracy"] for result in results]
    cv_top5_scores = [result["cv_top5_accuracy"] for result in results]
    
    return {
        "mean_train_accuracy": np.mean(train_scores),
        "mean_cv_accuracy": np.mean(cv_scores),
        "std_train_accuracy": np.std(train_scores),
        "std_cv_accuracy": np.std(cv_scores),
        "mean_gap": np.mean(gaps),
        "mean_train_top5_accuracy": np.mean(train_top5_scores),
        "mean_cv_top5_accuracy": np.mean(cv_top5_scores)
    }

def display_images(images, titles=None, suptitle=None, cols=10, img_shape=(32,32)):
    """Core rendering method any collection of 1D/2D image arrays"""
    m = len(images)
    if m == 0:
        print("No images to display")
        return
    
    rows = math.ceil(m / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.4), squeeze=False)
    fig.tight_layout(pad=0.2, rect=[0, 0.03, 1, 0.93])
    
    axes_flat = axes.flatten()
    
    for i in range(m):
        ax = axes_flat[i]
        ax.imshow(images[i].reshape(img_shape), cmap="gray")
        if titles is not None:
            ax.set_title(str(titles[i]), fontname="Ebrima")
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
    titles = [f"{GEEZ_CHARACTERS[Y[i]]}" for i in random_indices]

    display_images(
        images=X[random_indices],
        titles=titles,
        cols=cols
    )

def show_images_with_prediction(model, X, Y, num_of_images=20, cols=10):
    """Show randomly selected images along with the a given model's predictions"""
    random_indices = np.random.choice(len(X), size=num_of_images, replace=False)
    predictions = model.predict(X[random_indices])
    softmax_predictions = tf.nn.softmax(predictions)
    predicted_labels = np.argmax(softmax_predictions, axis=1)
    titles = [f"{GEEZ_CHARACTERS[Y[i]]} | {GEEZ_CHARACTERS[predicted_labels[idx]]}" for idx, i in enumerate(random_indices)]
    
    display_images(
        images=X[random_indices],
        titles=titles,
        cols=cols,
        suptitle="Actual vs. Predicted"
    )

def show_misclassifications(model, X, Y, num_of_images=20, cols=10):
    """Finds and displays misclassified examples"""
    predictions = model.predict(X)
    softmax_predictions = tf.nn.softmax(predictions)
    predicted_labels = np.argmax(softmax_predictions, axis=1)
    
    errors = np.where(predicted_labels != Y)[0]
    total_errors = len(errors)
    
    if total_errors == 0:
        print("No misclassifications found")
        return

    # Limit the number of displayed misclassification using num_of_images as a limit
    errors = errors[:num_of_images]
    titles = [f"{GEEZ_CHARACTERS[Y[i]]} | {GEEZ_CHARACTERS[predicted_labels[i]]}" for i in errors]

    display_images(
        images=X[errors],
        titles=titles,
        cols=cols,
        suptitle=f"Actual | Predicted {len(errors)} misclassifications shown out of {total_errors} misclassifications."
    )