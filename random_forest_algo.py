# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Create a sample customer churn dataset
data = {'Tenure': [12, 24, 36, 48, 60, 8, 16, 30, 45, 55],
        'MonthlyCharges': [50, 80, 60, 100, 120, 30, 40, 70, 90, 110],
        'ContractType': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'Two year',
                          'Month-to-month', 'One year', 'Month-to-month', 'Two year', 'One year'],
        'Churn': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0]}

df = pd.DataFrame(data)

# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['ContractType'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract feature importances
#feature_importance = model.feature_importances
feature_importance = model.feature_importances_

# Display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nFeature Importances:\n", feature_importance_df)