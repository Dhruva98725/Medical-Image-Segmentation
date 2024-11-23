# config/config.py

# General settings
IMG_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 16
EPOCHS = 10

# Fuzzy CNN settings
MEMBERSHIP_FUNC = 'triangular'  # Choose 'triangular', 'gaussian', etc.
SIGMA = 1.0  # Standard deviation for Gaussian membership function

# Dataset directories
DATA_DIR = '../data/'  # Root directory for images ('yes' and 'no')
