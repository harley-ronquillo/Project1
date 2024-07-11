# File path for the CSV file
file_path = 'test.csv'
new_data_file_path = 'new_data.csv'
predictions_output_path = 'predictions.csv'

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load and preprocess the data
# Load the data from the CSV file
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
print(data.head())

# Handle missing values (if any)
# Here we are dropping rows with any missing values. This can be changed based on the dataset and requirements.
data = data.dropna()

# Encode categorical features (if any). Assuming 'satisfaction' is the target variable.
# This is a simple example. You might need to use more sophisticated encoding methods depending on your dataset.
target = 'satisfaction'
features = data.drop(columns=[target]).select_dtypes(include=['number']).columns.tolist()

# Step 2: Feature selection and splitting the dataset
# Using only the first 10000 rows for the project
data_sample = data.sample(n=10000, random_state=42)

# Splitting the dataset into training and testing sets
X = data_sample[features]
y = data_sample[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model training
# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Model evaluation
# Predict on the test data
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# Step 5: Generate new data for prediction
# Pick 100 random data points from the original data
new_data_sample = data.sample(n=100, random_state=42)

# Save the new data to a CSV file for prediction
new_data_sample.to_csv(new_data_file_path, index=False)

# Load the new CSV file
new_data = pd.read_csv(new_data_file_path)

# Ensure the new data has the same feature columns as the original training data
new_data_features = new_data[features]

# Make predictions on the new data
new_data_predictions = model.predict(new_data_features)

# Add predictions to the new data dataframe
new_data['predicted_satisfaction'] = new_data_predictions

# Save the predictions to a new CSV file
new_data.to_csv(predictions_output_path, index=False)

print(f'Predictions saved to {predictions_output_path}')
