# -*- coding: utf-8 -*-
"""
================================================================================
CSE 548 Project 4: Machine Learning-Based Anomaly Detection Solutions
Feed-forward Neural Network (FNN) Implementation
================================================================================

Course: CSE 548 - Advanced Computer Network Security
Project: Machine Learning-Based Anomaly Detection using NSL-KDD Dataset
Author: Alexander Rivera
Date: December 11, 2025

Description:
    This script implements a Feed-forward Neural Network for network intrusion
    detection using the NSL-KDD dataset. It supports three experimental scenarios:
    - SA: Train on DoS+U2R, Test on Probe+R2L (untrained attacks)
    - SB: Train on DoS+Probe, Test on DoS (trained attacks)
    - SC: Train on DoS+Probe, Test on DoS+Probe+U2R (mixed attacks)
    - Please note that I made a copy of the original file and made all the changes 
      on the modified file. I wanted to have a backup in case something went wrong.

Key Modifications from Original Code:
    1. Added interactive scenario selection system
    2. Implemented separate train/test dataset loading (no train_test_split)
    3. Enhanced OneHotEncoder with handle_unknown='ignore' for robustness
    4. Added comprehensive error handling for edge cases
    5. Implemented scenario-specific plot naming
    6. Added detailed progress reporting and result analysis

================================================================================
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

import sys                          # For system operations and error handling
import pandas as pd                 # For CSV data loading and manipulation
import numpy as np                  # For numerical operations and array handling
import matplotlib.pyplot as plt     # For plotting accuracy and loss curves

# Scikit-learn preprocessing utilities
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Keras/TensorFlow for neural network implementation
from keras.models import Sequential
from keras.layers import Dense

# Scikit-learn metrics for evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ============================================================================
# SCENARIO CONFIGURATION AND SETUP
# ============================================================================

print("=" * 60)
print("FNN-Based Anomaly Detection - NSL-KDD Dataset")
print("CSE 548 Project 4: Machine Learning Solutions")
print("=" * 60)

# Alexander Rivera: Added interactive scenario selection system
# This allows users to easily switch between three experimental scenarios
# without modifying the code manually
SCENARIO = input("\nAvailable Scenarios:\n"
                "SA: Train(DoS+U2R+Normal), Test(Probe+R2L+Normal)\n"
                "SB: Train(DoS+Probe+Normal), Test(DoS+Normal)\n"
                "SC: Train(DoS+Probe+Normal), Test(DoS+Probe+U2R+Normal)\n"
                "\nEnter scenario (SA, SB, or SC): ").strip().upper()

# Alexander Rivera: Dataset mapping dictionary for easy scenario management
# Each scenario has specific training and testing datasets designed to
# evaluate different aspects of the model's detection capabilities
scenarios = {
    'SA': {
        'train': 'Training-a1-a3-a0.csv',
        'test': 'Testing-a2-a4-a0.csv',
        'description': 'Train: DoS+U2R+Normal, Test: Probe+R2L+Normal',
        'purpose': 'Evaluate detection of completely untrained attack types'
    },
    'SB': {
        'train': 'Training-a1-a2-a0.csv',
        'test': 'Testing-a1-a0.csv',
        'description': 'Train: DoS+Probe+Normal, Test: DoS+Normal',
        'purpose': 'Evaluate detection of trained attack types'
    },
    'SC': {
        'train': 'Training-a1-a2-a0.csv',
        'test': 'Testing-a1-a2-a3.csv',
        'description': 'Train: DoS+Probe+Normal, Test: DoS+Probe+U2R+Normal',
        'purpose': 'Evaluate mixed scenario with trained and untrained attacks'
    }
}

# Validate scenario selection
if SCENARIO not in scenarios:
    print(f"\nError: Invalid scenario '{SCENARIO}'!")
    print("Please choose SA, SB, or SC")
    sys.exit(1)

# Alexander Rivera: Extract dataset filenames based on selected scenario
TrainingData = scenarios[SCENARIO]['train']
TestingData = scenarios[SCENARIO]['test']

# Display scenario information
print(f"\nLoading Scenario {SCENARIO}")
print(f"Description: {scenarios[SCENARIO]['description']}")
print(f"Purpose: {scenarios[SCENARIO]['purpose']}")
print(f"Training: {TrainingData}")
print(f"Testing: {TestingData}")

# Alexander Rivera: Training hyperparameters
# These values were found to work well for binary classification
BatchSize = 10      # Number of samples per gradient update
NumEpoch = 10       # Number of complete passes through training data

print("=" * 60)

# ============================================================================
# PART 1: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 60)
print("Training Scenario {}...".format(SCENARIO))
print("Batch Size: {}, Epochs: {}".format(BatchSize, NumEpoch))
print("=" * 60)

# ----------------------------------------------------------------------------
# Step 1.1: Load Training and Testing Datasets
# ----------------------------------------------------------------------------

# Alexander Rivera: Modified to load separate training and testing datasets
# Original code used train_test_split which doesn't work for our scenarios
# We need predefined test sets to evaluate specific attack type combinations

print(f"\nLoading training dataset: {TrainingData}")
train_dataset = pd.read_csv(TrainingData, header=None)
print(f"Training shape: {train_dataset.shape}")

print(f"Loading testing dataset: {TestingData}")
test_dataset = pd.read_csv(TestingData, header=None)
print(f"Testing shape: {test_dataset.shape}")

# ----------------------------------------------------------------------------
# Step 1.2: Separate Features and Labels
# ----------------------------------------------------------------------------

# Alexander Rivera: Extract features (X) and labels (y) from datasets
# NSL-KDD format: [41 features, attack_type, difficulty_level]
# We use columns 0 to -2 as features (excluding last 2 columns)
# Column -2 contains the attack type label ('normal' or specific attack name)

X_train = train_dataset.iloc[:, 0:-2].values  # All feature columns
X_test = test_dataset.iloc[:, 0:-2].values

label_column_train = train_dataset.iloc[:, -2].values  # Attack type labels
label_column_test = test_dataset.iloc[:, -2].values

# ----------------------------------------------------------------------------
# Step 1.3: Binary Label Encoding
# ----------------------------------------------------------------------------

# Alexander Rivera: Convert multi-class labels to binary classification
# Normal traffic = 0, Any attack type = 1
# This simplifies the problem and enables better generalization across attack types

print("\nConverting labels to binary classification (Normal=0, Attack=1)...")

y_train = []
for label in label_column_train:
    if label == 'normal':
        y_train.append(0)
    else:
        y_train.append(1)  # All attack types mapped to 1

y_test = []
for label in label_column_test:
    if label == 'normal':
        y_test.append(0)
    else:
        y_test.append(1)

# Convert lists to numpy arrays for compatibility with Keras
y_train = np.array(y_train)
y_test = np.array(y_test)

# Display label distribution to understand dataset composition
print(f"Training labels - Normal: {np.sum(y_train == 0)}, Attack: {np.sum(y_train == 1)}")
print(f"Testing labels - Normal: {np.sum(y_test == 0)}, Attack: {np.sum(y_test == 1)}")

# ----------------------------------------------------------------------------
# Step 1.4: Categorical Feature Encoding
# ----------------------------------------------------------------------------

# Alexander Rivera: Apply One-Hot Encoding to categorical columns
# Columns 1, 2, 3 represent: protocol_type, service, and flag
# These are categorical variables that need to be converted to numerical format
# 
# CRITICAL MODIFICATION: Added handle_unknown='ignore' parameter
# This prevents errors when test data contains categories not seen in training
# For example, if test data has a new service type or protocol not in training set

print("\nEncoding categorical features (columns 1, 2, 3)...")
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [1, 2, 3])],
    remainder='passthrough'  # Keep other columns unchanged
)

# Fit encoder on training data and transform both train and test
# This ensures consistent encoding while preventing data leakage
X_train = np.array(ct.fit_transform(X_train), dtype=np.float32)
X_test = np.array(ct.transform(X_test), dtype=np.float32)

print(f"After encoding - Training shape: {X_train.shape}")
print(f"After encoding - Testing shape: {X_test.shape}")

# The shape increases from 41 to ~106 features due to one-hot encoding expansion

# ----------------------------------------------------------------------------
# Step 1.5: Feature Scaling (Standardization)
# ----------------------------------------------------------------------------

# Alexander Rivera: Apply standardization to normalize feature ranges
# StandardScaler transforms features to have mean=0 and variance=1
# This improves neural network training by:
# 1. Preventing features with large ranges from dominating
# 2. Enabling faster convergence during gradient descent
# 3. Improving numerical stability

print("\nPerforming feature scaling (standardization)...")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit on training data only
X_test = sc.transform(X_test)        # Apply same transformation to test data

print("Preprocessing complete!")

# ============================================================================
# PART 2: BUILDING AND TRAINING THE NEURAL NETWORK
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: BUILDING AND TRAINING FNN")
print("=" * 60)

# ----------------------------------------------------------------------------
# Step 2.1: Initialize the Neural Network Architecture
# ----------------------------------------------------------------------------

# Alexander Rivera: Create Sequential model (layers are added sequentially)
classifier = Sequential()

# Get input dimension from preprocessed training data
input_dim = X_train.shape[1]
print(f"\nInput dimension: {input_dim} features")

# ----------------------------------------------------------------------------
# Step 2.2: Add Network Layers
# ----------------------------------------------------------------------------

# Alexander Rivera: Input layer + First hidden layer
# - Units: 6 nodes (found experimentally to work well for this dataset)
# - Activation: ReLU (Rectified Linear Unit) for non-linearity
# - Initialization: Uniform distribution for weights
print("Adding input layer and first hidden layer (6 nodes, ReLU activation)")
classifier.add(Dense(units=6, 
                    kernel_initializer='uniform', 
                    activation='relu', 
                    input_dim=input_dim))

# Alexander Rivera: Second hidden layer
# - Same configuration as first layer
# - Provides additional capacity for learning complex patterns
print("Adding second hidden layer (6 nodes, ReLU activation)")
classifier.add(Dense(units=6, 
                    kernel_initializer='uniform', 
                    activation='relu'))

# Alexander Rivera: Output layer
# - Units: 1 node (binary classification: attack or normal)
# - Activation: Sigmoid (outputs probability between 0 and 1)
print("Adding output layer (1 node, Sigmoid activation for binary classification)")
classifier.add(Dense(units=1, 
                    kernel_initializer='uniform', 
                    activation='sigmoid'))

# Network architecture summary:
# Input(106) → Dense(6,ReLU) → Dense(6,ReLU) → Dense(1,Sigmoid)
# Total parameters: approximately 691 trainable parameters

# ----------------------------------------------------------------------------
# Step 2.3: Compile the Model
# ----------------------------------------------------------------------------

# Alexander Rivera: Configure the model for training
# - Optimizer: Adam (adaptive learning rate, works well for most problems)
# - Loss: Binary cross-entropy (standard for binary classification)
# - Metrics: Accuracy (to monitor during training)
print("\nCompiling the model...")
print("Optimizer: Adam")
print("Loss function: Binary cross-entropy")
print("Metrics: Accuracy")
classifier.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

# ----------------------------------------------------------------------------
# Step 2.4: Train the Neural Network
# ----------------------------------------------------------------------------

# Alexander Rivera: Begin training process
print(f"\nTraining Scenario {SCENARIO}...")
print(f"Batch Size: {BatchSize}, Epochs: {NumEpoch}")
print("=" * 60)

# Train the model and store history for visualization
# - batch_size: Number of samples per gradient update
# - epochs: Number of complete passes through training data
# - verbose: 1 shows progress bar with metrics
classifierHistory = classifier.fit(X_train, y_train, 
                                   batch_size=BatchSize, 
                                   epochs=NumEpoch, 
                                   verbose=1)

# Display training completion message
train_samples = X_train.shape[0]
print("=" * 60)
print("Training completed!")
print(f"Trained on {train_samples} samples over {NumEpoch} epochs")

# ============================================================================
# PART 3: MODEL EVALUATION AND RESULTS
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: EVALUATION ON TEST SET")
print("=" * 60)

# ----------------------------------------------------------------------------
# Step 3.1: Make Predictions on Test Data
# ----------------------------------------------------------------------------

# Alexander Rivera: Generate predictions for test set
print("\nMaking predictions on test dataset...")
y_pred_prob = classifier.predict(X_test)  # Get probability predictions
y_pred = (y_pred_prob > 0.5).astype(int).flatten()  # Convert to binary (0 or 1)

# Calculate test set metrics
test_loss, test_accuracy = classifier.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print("=" * 60)

# ----------------------------------------------------------------------------
# Step 3.2: Confusion Matrix Analysis
# ----------------------------------------------------------------------------

# Alexander Rivera: Generate and display confusion matrix
# Confusion matrix shows:
#   [ TN  FP ]    TN = True Negatives (Normal correctly identified)
#   [ FN  TP ]    FP = False Positives (Normal misclassified as Attack)
#                 FN = False Negatives (Attack misclassified as Normal)
#                 TP = True Positives (Attack correctly identified)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("[ TN, FP ]")
print("[ FN, TP ]=")
print(cm)

# Alexander Rivera: Added error handling for edge cases
# Some test sets may contain only one class (e.g., all attacks, no normal traffic)
# This try-except block handles such cases gracefully
print("\nConfusion Matrix Breakdown:")
try:
    print(f"True Negatives (Normal as Normal): {cm[0][0]}")
    print(f"False Positives (Normal as Attack): {cm[0][1]}")
    print(f"False Negatives (Attack as Normal): {cm[1][0]}")
    print(f"True Positives (Attack as Attack): {cm[1][1]}")
except IndexError:
    # Handle case where only one class is present in test set
    print("Note: Only one class present in test set")
    print(f"Total samples correctly classified: {cm[0][0] if cm.shape[0] > 0 else 0}")

# ----------------------------------------------------------------------------
# Step 3.3: Classification Report
# ----------------------------------------------------------------------------

# Alexander Rivera: Display detailed classification metrics
# Includes precision, recall, F1-score for each class
print("\nClassification Report:")
try:
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
except ValueError as e:
    print("Note: Classification report unavailable (only one class present)")

# ----------------------------------------------------------------------------
# Step 3.4: Per-Class Accuracy Analysis
# ----------------------------------------------------------------------------

# Alexander Rivera: Calculate accuracy separately for attack and normal samples
# This helps understand model performance on each class individually
try:
    attack_indices = np.where(y_test == 1)[0]
    if len(attack_indices) > 0:
        attack_accuracy = accuracy_score(y_test[attack_indices], y_pred[attack_indices])
        print(f"\nAccuracy on Attack samples only: {attack_accuracy*100:.2f}%")
    
    normal_indices = np.where(y_test == 0)[0]
    if len(normal_indices) > 0:
        normal_accuracy = accuracy_score(y_test[normal_indices], y_pred[normal_indices])
        print(f"Accuracy on Normal samples only: {normal_accuracy*100:.2f}%")
except Exception as e:
    print(f"\nNote: Per-class accuracy calculation unavailable")

# ============================================================================
# PART 4: VISUALIZATION OF TRAINING HISTORY
# ============================================================================

print("\n" + "=" * 60)
print(f"Scenario {SCENARIO} Analysis Complete")
print("=" * 60)

# ----------------------------------------------------------------------------
# Step 4.1: Plot Training Accuracy
# ----------------------------------------------------------------------------

# Alexander Rivera: Visualize how accuracy improved during training
# This helps identify if the model converged properly
print("\nPlot the accuracy")
plt.figure(figsize=(10, 6))

# Alexander Rivera: Handle both old and new Keras history key names
# Older versions use 'acc', newer versions use 'accuracy'
accuracy_key = 'accuracy' if 'accuracy' in classifierHistory.history else 'acc'
plt.plot(classifierHistory.history[accuracy_key])
plt.title(f'Model Accuracy - Scenario {SCENARIO}')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='lower right')

# Alexander Rivera: Save plot with scenario-specific filename
# This prevents overwriting when running multiple scenarios
accuracy_filename = f'accuracy_{SCENARIO}.png'
plt.savefig(accuracy_filename)
print(f"Accuracy plot saved as: {accuracy_filename}")
plt.show()

# ----------------------------------------------------------------------------
# Step 4.2: Plot Training Loss
# ----------------------------------------------------------------------------

# Alexander Rivera: Visualize how loss decreased during training
# Lower loss indicates better model fit to training data
print("\nPlot the loss")
plt.figure(figsize=(10, 6))
plt.plot(classifierHistory.history['loss'])
plt.title(f'Model Loss - Scenario {SCENARIO}')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')

# Alexander Rivera: Save plot with scenario-specific filename
loss_filename = f'loss_{SCENARIO}.png'
plt.savefig(loss_filename)
print(f"Loss plot saved as: {loss_filename}")
plt.show()

# ============================================================================
# PROGRAM COMPLETION
# ============================================================================

print("\n" + "=" * 60)
print(f"SCENARIO {SCENARIO} EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"\nFinal Results Summary:")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Training Epochs: {NumEpoch}")
print(f"  Test Samples: {len(y_test)}")
print(f"\nGenerated Files:")
print(f"  - {accuracy_filename}")
print(f"  - {loss_filename}")
print("\nThank you for using the FNN-Based Anomaly Detection System!")
print("=" * 60)

# ============================================================================
# END OF PROGRAM
# ============================================================================
