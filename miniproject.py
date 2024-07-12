import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data from the CSV files
train = pd.read_csv('/mnt/data/train.csv')
test = pd.read_csv('/mnt/data/test.csv')

# Check the columns in the DataFrame
print("Columns in train dataset:", train.columns)
print("Columns in test dataset:", test.columns)

# Drop unnecessary columns
train.drop(columns=["Unnamed: 0", "id"], inplace=True)
test.drop(columns=["Unnamed: 0", "id"], inplace=True)

# Create a new feature for total satisfaction level
columns_to_sum = ["Inflight wifi service", "Departure/Arrival time convenient", 
                  "Ease of Online booking", "Gate location", "Food and drink", 
                  "Online boarding", "Seat comfort", "Inflight entertainment", 
                  "On-board service", "Leg room service", "Baggage handling", 
                  "Checkin service", "Inflight service", "Cleanliness"]
train["total_satisfaction_level"] = train[columns_to_sum].sum(axis=1)
test["total_satisfaction_level"] = test[columns_to_sum].sum(axis=1)

# Check the columns after creating the new feature
print("Columns in train dataset after adding total_satisfaction_level:", train.columns)

# One-hot encode categorical features
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Ensure train and test have the same columns after one-hot encoding
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Split the data into features (X) and target (y)
X = train.drop(columns=["satisfaction"], errors='ignore')
y = train["satisfaction"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

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
