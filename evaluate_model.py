# evaluate_model.py
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model (adjust the path to where your model is saved)
try:
    model = load_model("D:\\Learning\\Code\\Projects\\Medical Image Segmentation\\Medical-Image-Segmentation\\fuzzy_cnn_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


# Load your validation data (replace with your actual data loading function)
# Example:
# X_val, y_val = load_your_data()  # Make sure you load your validation data

# Make predictions on validation data
y_pred = model.predict(X_val)

# Flatten the true and predicted masks (assuming binary segmentation)
y_true_flat = y_val.flatten()
y_pred_flat = np.round(y_pred.flatten())  # Convert probabilities to binary 0/1 for classification

# Compute the confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Normalize the confusion matrix and plot
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
plt.figure(figsize=(6, 5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Background', 'Foreground'], yticklabels=['Background', 'Foreground'])
plt.title("Normalized Confusion Matrix")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# If you have history object from training, you can plot the accuracy and loss curves
# Example:
# plot_metrics(history)