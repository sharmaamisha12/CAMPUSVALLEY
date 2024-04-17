# Step 1: Import necessary libraries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Step 3: Data Exploration - Print the first few rows and some statistics
print(titanic.head())
print(titanic.describe())
print(titanic.info())

# Step 4: Data Preprocessing

## Handle Missing Values
# Fill numeric columns with their median
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['fare'] = titanic['fare'].fillna(titanic['fare'].median())

# Fill categorical columns with their mode
titanic['embarked'] = titanic['embarked'].fillna(titanic['embarked'].mode()[0])
titanic['deck'] = titanic['deck'].fillna(titanic['deck'].mode()[0])

# Fill string column 'embark_town' with a placeholder
titanic['embark_town'] = titanic['embark_town'].fillna('Unknown')

# Confirm no more missing values
print(titanic.isnull().sum())

## Encode Categorical Data
# Convert categorical and boolean columns to numeric using get_dummies
titanic = pd.get_dummies(titanic, drop_first=True)

# Step 5: Prepare Data for Modeling

# Separate features and target variable
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build the Model

# Initialize a Logistic Regression model
model = LogisticRegression(max_iter=500)

# Fit the model on the training data
model.fit(X_train, y_train)

# Step 7: Model Evaluation

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 8: Visualize the Confusion Matrix

# Visualize the confusion matrix using heatmap
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
