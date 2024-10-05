import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load train and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data Preprocessing
# One-hot encoding for categorical variables
train_data = pd.get_dummies(train_data, columns=['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation'])
test_data = pd.get_dummies(test_data, columns=['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation'])

# Ensure 'Class/ASD' is included in test data
if 'Class/ASD' not in test_data.columns:
    test_data['Class/ASD'] = 0

# Ensure both datasets have the same columns after one-hot encoding
missing_cols_train = set(train_data.columns) - set(test_data.columns)
missing_cols_test = set(test_data.columns) - set(train_data.columns)

for col in missing_cols_train:
    test_data[col] = 0

for col in missing_cols_test:
    train_data[col] = 0

# Reorder columns to match the order in the training data
test_data = test_data[train_data.columns]

# Separate features and target variable
X_train = train_data.drop(['ID', 'Class/ASD'], axis=1)
y_train = train_data['Class/ASD']
X_test = test_data.drop(['ID', 'Class/ASD'], axis=1)  # Exclude 'ID' and 'Class/ASD' columns from test data

# Train Support Vector Machine classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Cross-validation to evaluate the model
cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Predict outcomes on test data
predictions = svm.predict(X_test)

# Save predictions to CSV
submission = pd.DataFrame({'ID': test_data['ID'], 'Class/ASD': predictions})
submission.to_csv('svm_predictions.csv', index=False)
