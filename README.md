# 96-Well Localization and Cis/Trans Concentration Prediction Pipeline

This repository contains the image processing and machine learning pipeline for localizing 96 sensing wells from fluorescence images and predicting the concentrations of cis- and trans-isomers based on RGB channel intensities.

The pipeline is divided into two primary stages:
1. Geometric Detection: A Convolutional Neural Network (CNN) that detects the coordinates of the 96 wells.
2. Concentration Regression: Machine learning models (Traditional ML and an Artificial Neural Network) that predict the isomer concentrations from the extracted RGB values of those wells.

---

## Environment and Dependencies

To run the scripts, ensure you have a Python environment with the following libraries installed:
* torch, torchvision (PyTorch)
* opencv-python (cv2)
* numpy, pandas
* scikit-learn
* matplotlib, seaborn
* albumentations
* tqdm

It is recommended to run the neural network scripts (CNN-detect-circle.py and ANN.py) on a CUDA-enabled GPU, though they will automatically fall back to CPU if no GPU is detected.

---

## File Descriptions

### 1. CNN-detect-circle.py
This script trains a custom Convolutional Neural Network (KeypointNet) to localize the centers of the 96 sensing wells.
* Architecture: 4 convolutional blocks (filter sizes 32 to 256) followed by 3 fully connected layers predicting 192 output values (x, y coordinates for 96 points).
* Data Processing: Applies extensive data augmentation (rotations, flips, brightness adjustments) using albumentations.
* Output: Saves the trained weights (best_model.pth), training history, validation metrics, and visualizes the predicted keypoints on test images.

### 2. traditinal_model.py
This script evaluates multiple traditional machine learning algorithms for multi-output regression to predict Cis and Trans concentrations from exactly 6 RGB feature columns (A_B, A_G, A_R, B_B, B_G, B_R).
* Models Tested: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, Decision Tree, and K-Nearest Neighbors (KNN).
* Evaluation: Uses an 80/20 train-test split and 5-fold cross-validation.
* Output: Automatically generates and saves highly detailed, publication-ready PDF plots (editable in Adobe Illustrator), including R² performance comparisons, error distributions, and feature correlation heatmaps.

### 3. ANN.py
This script trains a feedforward Artificial Neural Network (ANN) to perform the same Cis and Trans concentration prediction task.
* Architecture: A 3-hidden-layer network (32 -> 16 -> 8 neurons) with ReLU activations and Dropout (p=0.1) after the first layer.
* Training: Trains for up to 4000 epochs using the Adam optimizer, implementing early stopping with a patience of 200 epochs based on validation loss.
* Output: Saves the trained PyTorch model and scalers (.pth and .pkl files), alongside publication-ready PDF charts of the loss history, residual distributions, and Cis:Trans ratio accuracy.

---

## Data Requirements

The scripts expect your data to be organized in specific formats and directories. Update the base directories within the code if your paths differ.

For CNN Well Detection:
* Images must be .jpg files located in: DATA_96/circle/
* Labels must be .npz files (containing x and y coordinate arrays) located in: DATA_96/loc/

For Concentration Prediction:
* Both regression scripts expect a CSV file named all_colors_data.csv in the root directory.
* The CSV must contain the 6 RGB feature columns (A_B, A_G, A_R, B_B, B_G, B_R) and target variable columns (which the scripts will automatically map to Cis and Trans).

---

## How to Run

Step 1: Well Localization
Run the CNN script to train the model for 96-well detection.
> python CNN-detect-circle.py
(Check the dynamically created results/experiment_[timestamp] folder for model weights and visualizations.)

Step 2: Concentration Prediction Analysis
After extracting the mean RGB values from your detected wells into all_colors_data.csv, run the traditional ML script to establish baselines and identify important features.
> python traditinal_model.py

Step 3: Neural Network Concentration Prediction
Train the final multi-output ANN on the RGB data.
> python ANN.py
(Check the model_output_plots/ folder for comprehensive PDF reports of model performance.)
