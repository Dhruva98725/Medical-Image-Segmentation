import tensorflow as tf
from tensorflow.keras import layers, models
from utils.fuzzify import fuzzify_image
from config.config import MEMBERSHIP_FUNC, SIGMA

def build_fuzzy_cnn(input_shape):
    """
    Build a simple Fuzzy CNN model for tumor classification.
    
    Args:
    - input_shape: Shape of the input images (height, width, channels).
    
    Returns:
    - model: The compiled CNN model.
    """
    inputs = layers.Input(shape=input_shape)

    # Assuming IMG_SIZE = (128, 128)
    IMG_SIZE = (128, 128)

    # Modify the Lambda layer by specifying the output shape
    fuzzified_input = layers.Lambda(
        lambda x: fuzzify_image(x, MEMBERSHIP_FUNC, SIGMA),
        output_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)  # Specify the output shape explicitly
    )(inputs)

    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(fuzzified_input)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Output layer (binary classification: tumor or no tumor)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Compile model
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
