import numpy as np

import tensorflow as tf

def fuzzify_image(image, membership_func='triangular', sigma=1.0):
    """
    Apply fuzzification to the pixel intensities using a specified membership function.
    
    Args:
    - image: Input image (grayscale or RGB).
    - membership_func: Type of fuzzy membership function ('triangular', 'gaussian', etc.).
    - sigma: Standard deviation for 'gaussian'.
    
    Returns:
    - fuzzified_image: Fuzzified image with membership values between 0 and 1.
    """
    # Normalize the image to the range [0, 1]
    image = tf.cast(image, tf.float32) / 255.0  # Ensure the image is a float and normalized
    
    if membership_func == 'triangular':
        fuzzified_image = triangular_membership(image)
    elif membership_func == 'gaussian':
        fuzzified_image = gaussian_membership(image, sigma)
    else:
        raise ValueError("Unsupported membership function")

    # Ensure the fuzzified image values are clipped to the range [0, 1]
    fuzzified_image = tf.clip_by_value(fuzzified_image, 0.0, 1.0)

    return fuzzified_image


def triangular_membership(image):
    """
    Apply a triangular membership function to the image.
    Image values are fuzzified between 0 and 1.

    Args:
    - image: Input image (normalized between 0 and 1).

    Returns:
    - fuzzified_image: Image fuzzified using a triangular membership function.
    """
    # Example of a triangular membership function
    fuzzified_image = tf.abs(image - 0.5)  # Some example operation
    fuzzified_image = tf.clip_by_value(fuzzified_image, 0.0, 1.0)
    return fuzzified_image


def gaussian_membership(image, sigma=1.0):
    """
    Apply Gaussian fuzzy membership function to fuzzify image intensities.
    """
    fuzzified = np.exp(-(image - 128)**2 / (2 * sigma**2))  # Assume mid-intensity value of 128
    return fuzzified
