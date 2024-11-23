import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.preprocess import load_data
from models.fuzzy_cnn import build_fuzzy_cnn
from config.config import BATCH_SIZE, EPOCHS, IMG_SIZE

def main():
    # Load and preprocess data
    print("Loading data...")
    X_train, X_val, y_train, y_val = load_data()
    print(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples.")

    # Build the Fuzzy CNN model
    print("Building the model...")
    model = build_fuzzy_cnn((128, 128, 1))  # Grayscale images, 128x128
    
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint('fuzzy_cnn_model.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[checkpoint])

    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
