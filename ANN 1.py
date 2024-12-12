import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Load the dataset
df = pd.read_csv("C:/Users/yew04/OneDrive/Desktop/UMS UNI LIFE/FYP/DATASET/DATASET WSN 1 ed.csv")

# Extract the features and the target variable from the dataset
X = df.drop('Class', axis=1)  # Exclude the 'Class' column
y = df['Class']

# Apply label encoding to the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the ANN classifier
ann = MLPClassifier()

# Variables to track training progress
total_epochs = 10  # Specify the total number of epochs
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_precisions = []
val_precisions = []

# Training loop
start_time = time.time()
for epoch in range(total_epochs):
    # Train the model using the training data
    ann.fit(X_train, y_train)

    # Training evaluation
    train_pred = ann.predict(X_train)
    train_loss = 1 - accuracy_score(y_train, train_pred)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_precision = precision_score(y_train, train_pred, average='weighted', zero_division=1)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)

    # Validation evaluation
    val_pred = ann.predict(X_test)
    val_loss = 1 - accuracy_score(y_test, val_pred)
    val_accuracy = accuracy_score(y_test, val_pred)
    val_precision = precision_score(y_test, val_pred, average='weighted', zero_division=1)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)

    print(
        f"Epoch {epoch + 1}/{total_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Train Precision: {train_precision:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Val Precision: {val_precision:.4f}")

end_time = time.time()

# Evaluate the model on the testing set
y_pred = ann.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
print(f"Total Testing Accuracy: {accuracy:.4f}")
print(f"Total Testing Precision: {precision:.4f}")

# Calculate total validation accuracy and precision
val_accuracy_total = accuracy_score(y_test, val_pred)
val_precision_total = precision_score(y_test, val_pred, average='weighted', zero_division=1)
print(f"Total Validation Accuracy: {val_accuracy_total:.4f}")
print(f"Total Validation Precision: {val_precision_total:.4f}")

# Calculate average training and validation losses
avg_train_loss = np.mean(train_losses)
avg_val_loss = np.mean(val_losses)
print(f"Average Training Loss: {avg_train_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")

# Calculate average training and validation accuracies
avg_train_accuracy = np.mean(train_accuracies)
avg_val_accuracy = np.mean(val_accuracies)
print(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")

# Calculate average training and validation precisions
avg_train_precision = np.mean(train_precisions)
avg_val_precision = np.mean(val_precisions)
print(f"Average Training Precision: {avg_train_precision:.4f}")
print(f"Average Validation Precision: {avg_val_precision:.4f}")

# Calculate total time elapsed
total_time = end_time - start_time
print(f"Total Time Elapsed: {total_time:.2f} seconds")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
