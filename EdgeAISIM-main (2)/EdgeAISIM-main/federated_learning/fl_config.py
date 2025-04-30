import os

# Server Configuration
SERVER_ADDRESS = "0.0.0.0:9090"  # Change if needed
NUM_ROUNDS = 5  # Number of FL training rounds

# Dataset Configuration
DATASET_PATH = os.path.join(os.getcwd(), "sample_dataset1.json")  # Ensure this file exists

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5

# Federated Learning Strategy
MIN_FIT_CLIENTS = 2  # Minimum clients for training
MIN_EVALUATE_CLIENTS = 2  # Minimum clients for evaluation
MIN_AVAILABLE_CLIENTS = 2  # Minimum required clients

# Power Consumption Tracking
ENABLE_POWER_TRACKING = True  # Set to False to disable
