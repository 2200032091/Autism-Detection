import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Load train and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data Preprocessing
# One-hot encoding for categorical variables
train_data = pd.get_dummies(train_data, columns=['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation'])
test_data = pd.get_dummies(test_data, columns=['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation'])

# Ensure both datasets have the same columns after one-hot encoding
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Reorder columns to match the order in the training data
test_data = test_data[train_data.columns]

# Separate features and target variable
X_train = train_data.drop(['ID', 'Class/ASD'], axis=1)
y_train = train_data['Class/ASD']

# Train Decision Tree classifier with limited depth
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Perform k-fold cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

# Output cross-validation scores
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Optional: Evaluate the model on the training data
train_predictions = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)
