import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and preprocess the data
file_path = 'test.csv'
data = pd.read_csv(file_path)

# Handle missing values
data = data.dropna()

# Encode categorical features
target = 'satisfaction'
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if col != target:
        data[col] = LabelEncoder().fit_transform(data[col])

# Feature selection
features = data.drop(columns=[target]).columns.tolist()
X = data[features]
y = data[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model training with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_rf_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Generate new data for prediction
new_data_sample = data.sample(n=100, random_state=42)
new_data_sample.to_csv('new_data.csv', index=False)

# Load new data for prediction
new_data = pd.read_csv('new_data.csv')
new_data_features = new_data[features]
new_data_features_scaled = scaler.transform(new_data_features)

# Make predictions
new_data_predictions = best_rf_model.predict(new_data_features_scaled)
new_data['predicted_satisfaction'] = new_data_predictions

# Save predictions
new_data.to_csv('predictions.csv', index=False)
print(f'Predictions saved to predictions.csv')
