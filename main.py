import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"C:\Users\Yashvardhan Singh\Downloads\wine\winequality-red.csv")
df = df.drop(['fixed acidity', 'chlorides', 'free sulfur dioxide', 'pH', 'residual sugar'], axis=1)

# Convert quality to binary classification (1 for good, 0 for bad)
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use selected features from your feature selection
selected_features = ['alcohol', 'sulphates', 'density', 'citric acid']

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train[selected_features], y_train)
best_model = grid_search.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test[selected_features])
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("RandomForest Model Performance")
print("-" * 30)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", acc)
print("F1-score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['0 (bad)', '1 (good)'], yticklabels=['0 (bad)', '1 (good)'])
plt.title("RandomForest Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature importance plot
importances = best_model.feature_importances_
feature_names = selected_features
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# Test with a sample input
sample = pd.DataFrame({
    'alcohol': [10.5],
    'sulphates': [0.75],
    'density': [0.998],
    'citric acid': [0.3]
})

predicted_class = best_model.predict(sample)
predicted_proba = best_model.predict_proba(sample)

print("\nSample Input Prediction")
print("-" * 30)
print("Predicted Quality:", predicted_class[0], "(good)" if predicted_class[0] == 1 else "(bad)")
print("Class Probabilities:", predicted_proba[0])