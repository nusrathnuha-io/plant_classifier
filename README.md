Neural Network Classifier with Hyperparameter Tuning and K-Fold Cross-Validation
This project demonstrates how to build and optimize a neural network classifier using Keras and scikit-learn. We employ hyperparameter tuning with grid search and k-fold cross-validation to enhance model performance. The final model is trained on the entire dataset using the best hyperparameters identified during the tuning process.

Overview
Model Architecture: Defined using Keras Sequential API.
Hyperparameter Tuning: Performed using GridSearchCV from scikit-learn.
Cross-Validation: Utilized K-fold cross-validation to ensure robust model evaluation.
Visualization: Plotted training and validation accuracy for each fold to monitor performance.
Getting Started
Prerequisites
Ensure you have the following libraries installed:

numpy
scikit-learn
keras
tensorflow
matplotlib
You can install these packages using pip:

bash
Copy code
pip install numpy scikit-learn keras tensorflow matplotlib
Project Structure
create_model.py: Defines the neural network architecture.
hyperparameter_tuning.py: Contains code for hyperparameter tuning using grid search.
kfold_cross_validation.py: Implements k-fold cross-validation.
plot_accuracy.py: Visualizes training and validation accuracy.
train_final_model.py: Trains the final model on the entire dataset using the best hyperparameters.
final_model.h5: Saved final model after training on the entire dataset.
Data
Ensure you have the features and labels data available in the required format:

features_normalized: Numpy array of normalized feature data.
labels_onehot: Numpy array of one-hot encoded labels.
Usage
Define Model Architecture
In create_model.py, define the model architecture:

python
Copy code
from keras import models, layers, regularizers

def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model
Hyperparameter Tuning with Grid Search
In hyperparameter_tuning.py, perform hyperparameter tuning:

python
Copy code
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(learning_rate=0.01, num_hidden_layers=1, input_layer_units=64, hidden_layer_units=32, l2_reg=0.01):
    # Model definition code here

model = KerasClassifier(build_fn=create_model)

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'num_hidden_layers': [1, 2, 3],
    'input_layer_units': [64, 128, 256],
    'hidden_layer_units': [32, 64, 128],
    'l2_reg': [0.001, 0.01, 0.1]
}

gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
gs.fit(train_features, train_labels)

best_params = gs.best_params_
print("Best Hyperparameters:", best_params)
K-Fold Cross-Validation
In kfold_cross_validation.py, implement k-fold cross-validation:

python
Copy code
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint

splitter = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold_number, (train_index, test_index) in enumerate(splitter.split(features_normalized), start=1):
    # Data splitting and model training code here
    history = model.fit(train_features, train_labels, epochs=300, batch_size=32,
                        validation_split=0.2, callbacks=[early_stopping, model_checkpoint], verbose=0)

    history_per_fold.append(history)
    test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=0)
    fold_accuracies.append(test_accuracy)

average_test_accuracy = np.mean(fold_accuracies)
print(f"Average Test Accuracy with Hyperparameter Tuning: {average_test_accuracy}")
Plot Training and Validation Accuracy
In plot_accuracy.py, plot the accuracy:

python
Copy code
import matplotlib.pyplot as plt

def plot_accuracy_per_fold(history_per_fold):
    for fold_number, history in enumerate(history_per_fold):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title(f'Training and validation accuracy for Fold {fold_number + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

plot_accuracy_per_fold(history_per_fold)
Train Final Model
In train_final_model.py, train the final model:

python
Copy code
final_model = create_model(input_shape=features_normalized.shape[1:], num_classes=labels_onehot.shape[1])
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
final_model.fit(features_normalized, labels_onehot, epochs=300, batch_size=32, validation_split=0.2, verbose=1)
final_model.save('final_model.h5')
Results
Average Test Accuracy: Achieved high average test accuracy through k-fold cross-validation and hyperparameter tuning.
Best Hyperparameters: Identified optimal hyperparameters for model training.
Visualization: Plotted training and validation accuracy for each fold to monitor performance.
Conclusion
This project demonstrates an effective approach to building and optimizing a neural network classifier using hyperparameter tuning and k-fold cross-validation. By systematically searching through a range of hyperparameters and leveraging cross-validation, we achieve robust and reliable model performance.
