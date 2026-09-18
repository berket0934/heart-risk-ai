# HeartRisk AI

Deep neural network for heart disease risk prediction using structured clinical patient data.

## Project Overview

HeartRisk AI is a binary classification project built using the UCI Cleveland Heart Disease dataset.

The project applies preprocessing techniques to numerical and categorical clinical features before training a deep neural network to predict the presence of heart disease.

## Dataset

The model uses 13 clinical features, including:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope
- Number of major vessels
- Thalassemia

## Data Preprocessing

The preprocessing pipeline includes:

- Removing missing values
- Converting the target into binary classes
- Train/test split
- Standardizing numerical features
- One-hot encoding categorical features

## Model Architecture

The deep neural network consists of:

- Dense layer: 64 neurons + ReLU
- Dropout: 0.30
- Dense layer: 32 neurons + ReLU
- Dropout: 0.20
- Sigmoid output layer

Training configuration:

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 50
- Batch Size: 32

## Results

- Accuracy: **87%**
- Recall: **88%**

The model demonstrated strong classification performance on unseen test data.

## Technologies

- Python
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Deep Learning
- Data Preprocessing

## Disclaimer

This project was developed for educational and machine learning research purposes. It is not intended for medical diagnosis or clinical decision-making.

## Author

Berke Tüylek
