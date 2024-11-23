import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    # Correct the path to match the directory structure
    yes_dir = 'data/yes'
    no_dir = 'data/no'

    # Ensure the directories exist
    if not os.path.exists(yes_dir):
        raise FileNotFoundError(f"Directory {yes_dir} not found.")
    if not os.path.exists(no_dir):
        raise FileNotFoundError(f"Directory {no_dir} not found.")
    
    # List to hold the images and labels
    images = []
    labels = []

    # Load 'yes' images (with tumor)
    for img_name in os.listdir(yes_dir):
        img_path = os.path.join(yes_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
        if img is not None:
            img_resized = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img_resized)
            labels.append(1)  # Label 1 for tumor
        else:
            print(f"Skipping invalid image {img_name}")

    # Load 'no' images (without tumor)
    for img_name in os.listdir(no_dir):
        img_path = os.path.join(no_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (128, 128))  # Resize to 128x128
            images.append(img_resized)
            labels.append(0)  # Label 0 for no tumor
        else:
            print(f"Skipping invalid image {img_name}")

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalize pixel values to [0, 1]
    images = images / 255.0

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val
