# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv('IRIS.csv')

# Print the first few rows to check the dataset
print(df.head())

# Step 2: Preprocess the data
# This step is necessary if species are strings; convert them to integer codes if not already done.
if df['species'].dtype == 'object':
    df['species'] = pd.Categorical(df['species']).codes

X = df.drop('species', axis=1).values
y = df['species'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the model using KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = knn.predict(X_test)

# Print evaluation results
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 5: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
# Manually specify the names of the species in your dataset.
species_names = ['setosa', 'versicolor', 'virginica']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=species_names, yticklabels=species_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Visualization')
plt.show()
