# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
print("Loading dataset...")
try:
    data = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit(1)

# Step 2: Preprocess and normalize the data
print("Preprocessing and normalizing data...")
X = data.drop("Class", axis=1)
y = data["Class"]

# Normalize the 'Amount' column
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))

# Step 3: Handle class imbalance
print("Handling class imbalance...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Step 4: Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest Classifier
print("Training the model...")
try:
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    print("Model trained successfully.")
except Exception as e:
    print("Error during training:", str(e))
    exit(1)

# Step 6: Evaluate the model
print("Evaluating the model...")
try:
    y_pred = rf_classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
except Exception as e:
    print("Error during evaluation:", str(e))
