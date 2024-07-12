import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# To load the data from the CSV file
data = pd.read_csv('test.csv')
# Function to handle missing values and drop them
data = data.dropna()
# Preprocessing the data for training the model
target = 'satisfaction'
features = data.drop(columns=[target]).select_dtypes(include=['number']).columns.tolist()

# Declaring the sample size
data_sample = data.sample(n=100, random_state=42)

# Splitting the dataset into training and testing sets
# X is for the attributes to predict the Y which is the target
X = data_sample[features]
y = data_sample[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics to evaluate the accuracy of the training of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

