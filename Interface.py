import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def train_with_feature_selection(X_train, y_train, X_test, y_test, feature_columns, classification_column):
    # Train a Random Forest classifier
    start_time_train_rf = datetime.now()
    st.write("\nProcessing with Feature Selection...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Perform cross-validation to select important features
    cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)

    # Select important features using the trained Random Forest model
    feature_selector = SelectFromModel(rf_classifier, prefit=True)
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)

    # Display selected features
    selected_feature_indices = feature_selector.get_support(indices=True)
    selected_features = feature_columns[selected_feature_indices]
    st.write("Selected Features:", list(selected_features))

    # Plot bar graph for feature importances
    feature_importances = rf_classifier.feature_importances_
    st.bar_chart(pd.Series(feature_importances, index=feature_columns).sort_values(ascending=False))
    st.write("\n")

    # Calculate and print the feature selection time
    end_time_rf = datetime.now()
    execution_time_rf = (end_time_rf - start_time_train_rf).total_seconds()
    st.write(f"Feature Selection TIme: {execution_time_rf:.4f} seconds")

    # Train a KNN classifier with the important features
    start_time_train_knn = datetime.now()
    st.write("\nTraining KNN classifier...")
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_selected, y_train)

    # Calculate and print the training time for KNN
    end_time_train_knn = datetime.now()
    execution_time_train_knn = (end_time_train_knn - start_time_train_knn).total_seconds()
    st.write(f"Training KNN time: {execution_time_train_knn:.4f} seconds")

    # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test_selected)

    # Inverse transform the encoded labels to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Accuracy calculation
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy: {:.4f}".format(accuracy))

    # Display the prediction over true case graph
    fig, ax = plt.subplots()
    ax.scatter(range(len(y_test)), y_test_labels, label='True Case', color='blue')
    ax.scatter(range(len(y_test)), y_pred_labels, label='Predicted Case', color='red', marker='x')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class')
    ax.legend()
    ax.set_title('True vs Predicted DDoS Attacks')
    st.pyplot(fig)

    # Print the total true attacks and total predicted attacks
    total_true_attacks = sum(y_test)
    total_predicted_attacks = sum(y_pred)

    st.write("\nTotal True DDoS Attacks:", total_true_attacks)
    st.write("Total Predicted DDoS Attacks:", total_predicted_attacks)

    # Calculate the total loss attacks
    total_loss_attacks = total_true_attacks - total_predicted_attacks
    st.write("Total Lost DDoS Attacks:", total_loss_attacks)

    # Calculate the percentage of predicted attacks
    percentage_predicted_attacks = (total_predicted_attacks / total_true_attacks) * 100
    st.write("Percentage of Predicted DDoS Attacks: {:.2f}%".format(percentage_predicted_attacks))

    # Calculate the percentage of loss attacks
    percentage_loss_attacks = (total_loss_attacks / total_true_attacks) * 100
    st.write("Percentage of Lost DDoS Attacks: {:.2f}%".format(percentage_loss_attacks))


def train_without_feature_selection(X_train, y_train, X_test, y_test):
    # Train a KNN classifier without feature selection
    start_time_train_knn = datetime.now()
    st.write("\nTraining KNN classifier without feature selection...")
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)

    # Calculate and print the training time for KNN without feature selection
    end_time_train_knn = datetime.now()
    execution_time_train_knn = (end_time_train_knn - start_time_train_knn).total_seconds()
    st.write(f"Training KNN time without feature selection: {execution_time_train_knn:.4f} seconds")

    # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test)

    # Inverse transform the encoded labels to original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    # Accuracy calculation
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy without feature selection: {:.4f}".format(accuracy))

    # Display the prediction over true case graph
    fig, ax = plt.subplots()
    ax.scatter(range(len(y_test)), y_test_labels, label='True Case', color='blue')
    ax.scatter(range(len(y_test)), y_pred_labels, label='Predicted Case', color='red', marker='x')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class')
    ax.legend()
    ax.set_title('True vs Predicted DDoS Attacks without Feature Selection')
    st.pyplot(fig)

    # Print the total true attacks and total predicted attacks
    total_true_attacks = sum(y_test)
    total_predicted_attacks = sum(y_pred)

    st.write("\nTotal True DDoS Attacks without feature selection:", total_true_attacks)
    st.write("Total Predicted DDoS Attacks without feature selection:", total_predicted_attacks)

    # Calculate the total loss attacks
    total_loss_attacks = total_true_attacks - total_predicted_attacks
    st.write("Total Lost DDoS Attacks without feature selection:", total_loss_attacks)

    # Calculate the percentage of predicted attacks
    percentage_predicted_attacks = (total_predicted_attacks / total_true_attacks) * 100
    st.write("Percentage of Predicted DDoS Attacks without feature selection: {:.2f}%".format(percentage_predicted_attacks))

    # Calculate the percentage of loss attacks
    percentage_loss_attacks = (total_loss_attacks / total_true_attacks) * 100
    st.write("Percentage of Lost DDoS Attacks without feature selection: {:.2f}%".format(percentage_loss_attacks))

# Create a Streamlit app title and description
st.title("WSN DDoS Attack Prediction Dashboard")
st.write("This dashboard allows you to upload a CSV dataset and perform WSN DDoS attack detection prediction.")

# Add an upload file widget to allow users to upload a CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the uploaded dataset
    st.write("Uploaded Dataset:")
    st.write(df)

    # Add the "Dashboard View" section
    st.subheader("Dashboard View")
    st.write("-------------------------------")

    # Data Exploration
    st.subheader("Current Network Status")
    st.write(f"Total Rows: {df.shape[0]}")
    st.write(f"Total Features: {df.shape[1]}")
    st.write("-------------------------------")

    # Let the user select the classification column
    classification_column = st.selectbox("Select the Classification Column", df.columns)

    if classification_column is not None:

        # Check if the user has selected a classification column
        if classification_column:
            # Record the start time for loading dataset
            start_time_load_data = datetime.now()

            # Visualize class distribution
            st.subheader("Classification Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            df[classification_column].value_counts().plot(kind='bar', color=['blue', 'red'], ax=ax)
            plt.xlabel(classification_column)
            plt.ylabel('Count')
            plt.title('Classification Distribution')
            st.pyplot(fig)

            # Separate features (X) and target variable (y)
            X = df.drop(classification_column, axis=1)
            y = df[classification_column]

            # Convert string labels to numerical values using Label Encoding
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            # Let the user choose between training with and without feature selection
            if st.button("Train with Feature Selection"):
                train_with_feature_selection(X_train, y_train, X_test, y_test, X.columns, classification_column)

            if st.button("Train without Feature Selection"):
                train_without_feature_selection(X_train, y_train, X_test, y_test)

    else:
        st.warning("Please select a classification column.")
