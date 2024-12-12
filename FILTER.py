import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('C:/Users/yew04/OneDrive/Desktop/UMS UNI LIFE/FYP/DATASET/DATASET WSN 1.csv')

# Separate the target variable (the variable you want to predict) from the features
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to your data
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature names and their importance scores
print("Feature Importance Scores:")
print(feature_importance_df)

# Define a list of threshold values to test
thresholds = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]

# Initialize variables to store the best threshold and its associated performance
best_threshold = None
best_score = 0.0

# Initialize a dictionary to store threshold-accuracy pairs
threshold_accuracy_dict = {}

for threshold in thresholds:
    # Select features based on the current threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()
    X_selected = X[selected_features]

    # Train and evaluate your model with the selected features using cross-validation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy')

    # Calculate the mean cross-validated score
    mean_score = np.mean(scores)

    # Store the threshold-accuracy pair in the dictionary
    threshold_accuracy_dict[threshold] = mean_score

    # Update the best threshold and score if the current threshold performs better
    if mean_score > best_score:
        best_score = mean_score
        best_threshold = threshold

# Print the thresholds and their corresponding accuracy scores
print("\nThresholds and Accuracy Scores:")
for threshold, accuracy in threshold_accuracy_dict.items():
    print(f"Threshold: {threshold}, Accuracy: {accuracy:.4f}")

# Select features based on the best threshold
selected_features = feature_importance_df[feature_importance_df['Importance'] > best_threshold]['Feature'].tolist()
X_selected = X[selected_features]

# Print the best threshold
print("\nBest Threshold:", best_threshold)

# Print the selected feature names
print("\nSelected Features :")
print(selected_features)
