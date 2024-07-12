import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('test.csv')

# Function to check the percentages of satisfied and dissatisfied customers
def check_satisfaction_percentages(data, target, encoding=False):
    satisfaction_counts = data[target].value_counts(normalize=True) * 100
    if encoding:
        print("Percentage of Satisfied Customers:", satisfaction_counts.get(1, 0))
        print("Percentage of Dissatisfied Customers:", satisfaction_counts.get(0, 0))
    else:
        print("Percentage of Satisfied Customers:", satisfaction_counts.get('satisfied', 0))
        print("Percentage of Dissatisfied Customers:", satisfaction_counts.get('neutral or dissatisfied', 0))

# Preprocess the data
def preprocess_data(df):
    df = df.drop(["Unnamed: 0", "id"], axis=1, errors='ignore')
    df = df.dropna()
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df

# Check the percentages of satisfied and dissatisfied customers before any processing
print("Before Processing:")
check_satisfaction_percentages(data, 'satisfaction')

# Apply preprocessing
data = preprocess_data(data)

# Check the percentages of satisfied and dissatisfied customers after preprocessing
print("\nAfter Preprocessing:")
check_satisfaction_percentages(data, 'satisfaction', encoding=True)

# Prepare features and target
target = 'satisfaction'
features = data.drop(columns=[target]).columns.tolist()

# Declare the sample size
data_sample = data.sample(n=10000, random_state=42)

# Check the percentages of satisfied and dissatisfied customers after sampling
print("\nAfter Sampling:")
check_satisfaction_percentages(data_sample, 'satisfaction', encoding=True)

# Split the dataset into training and testing sets
X = data_sample[features]
y = data_sample[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Perform 5-fold cross-validation
cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_precision = cross_val_score(model, X, y, cv=5, scoring='precision_weighted')
cv_recall = cross_val_score(model, X, y, cv=5, scoring='recall_weighted')
cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')

print(f'\nCross-Validated Accuracy: {cv_accuracy.mean()}')
print(f'Cross-Validated Precision: {cv_precision.mean()}')
print(f'Cross-Validated Recall: {cv_recall.mean()}')
print(f'Cross-Validated F1-Score: {cv_f1.mean()}')

# Calculate and print evaluation metrics to evaluate the accuracy of the training of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'\nAccuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')



# 1. Feature Importance Plot
feature_importance = model.feature_importances_
feature_importance_sorted = sorted(zip(feature_importance, features), reverse=True)
plt.figure(figsize=(10, 6))
plt.bar([x[1] for x in feature_importance_sorted], [x[0] for x in feature_importance_sorted])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
plt.show()

# 3. Pairplot for top 5 important features
top_5_features = [x[1] for x in feature_importance_sorted[:5]]
top_5_df = X[top_5_features]
top_5_df['Target'] = y
sns.pairplot(top_5_df, hue='Target', height=2.5)
plt.suptitle('Pairplot of Top 5 Important Features', y=1.02)
plt.tight_layout()
plt.show()

# 4. Box plots for top 5 features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(top_5_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=y, y=X[feature])
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()
# Create and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()