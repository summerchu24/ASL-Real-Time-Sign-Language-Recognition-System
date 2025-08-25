# ASL Real-Time Sign Language Recognition System

This repository contains a machine learning project for gesture recognition using Support Vector Machine (SVM) and CNN-LSTM models. The project focuses on recognizing sign language gestures from video data by extracting key hand and body landmarks with the help of **Google Mediapipe**.

## Project Overview

### Models Implemented

- **CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory)**:
  - This model uses convolutional layers to extract spatial features from video frames and LSTM layers to capture temporal dependencies between frames.

- **SVM (Support Vector Machine)**:
  - An SVM model was built using key points from the hand and body landmarks. This model was implemented using both linear and non-linear kernels for classification.

- **MP-LSTM (Mediapipe + LSTM)**:
  - Combines Google's Mediapipe for feature extraction with an LSTM model to handle time-series data from video frames.

## Dataset
The dataset used in this project is divided into three parts and contains videos of sign language gestures. The dataset is stored in three zip files:

- `dataset1.zip`
- `dataset2.zip`
- `dataset3.zip`

To fully run the project, you need to combine these datasets.

## Key Features
- **Feature Extraction**: Key points for hand and pose are extracted using Google Mediapipe.
- **Frame Selection**: From each video, 20 frames are selected for analysis.
- **Data Preprocessing**: The selected frames are normalized using MinMaxScaler to ensure data consistency across the videos.
- **Evaluation**: The models are evaluated based on accuracy, recall, and F1-score.

## Tools and Technologies Used
- **Python**: Data manipulation and analysis.
- **Jupyter Notebooks**: Used for experimentation and model building.
- **Mediapipe**: For extracting hand and body landmarks.
- **Scikit-learn**: For SVM model implementation.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/summerchu24/SEP788_FinalProject.git
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the datasets and place them in the project directory.
4. Run the Jupyter notebooks to train and evaluate the models:
  - `SVM with Key Points.ipynb`
  - `SL_CNN_LSTM_New.ipynb`

## Results
The project showcases different models with varying performance metrics. For detailed results, refer to the output of each Jupyter notebook.
