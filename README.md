# ANN Species Classifier

This repository contains code for building and evaluating an Artificial Neural Network (ANN) classifier for species classification based on provided features. The classifier is developed using TensorFlow and Keras libraries in Python.

## Dataset

The dataset used for training and evaluation is stored in a CSV file named `sample.csv`. It includes features such as length-width ratio, stem height, number of leaves, and angle of leaf, along with the corresponding species label.

## Preprocessing

- The dataset is loaded into a Pandas DataFrame.
- Categorical columns are encoded using LabelEncoder.
- Features are normalized using StandardScaler.
- The dataset is split into training and testing sets using K-cross validation.

## Model Architecture

The neural network model consists of multiple Dense layers with ReLU activation functions. Dropout layers are incorporated for regularization to prevent overfitting. The output layer uses the softmax activation function for multiclass classification.

## Training

The model is trained using the Adam optimizer with categorical cross-entropy loss. An early stopping callback is implemented to prevent overfitting. Training progress is monitored using a validation split.

## Evaluation

The trained model is evaluated on both training and testing sets. Metrics such as loss and accuracy are computed and printed. A classification report and confusion matrix are generated to assess model performance.


## Results

- The model achieves an accuracy of approximately 98.8%.
- The classification report and confusion matrix provide insights into the model's performance across different classes.
- Further analysis and optimization can be performed to enhance the model's accuracy and generalization capabilities.
