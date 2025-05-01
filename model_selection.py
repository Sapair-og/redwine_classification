import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Yashvardhan Singh\Downloads\archive (2)\winequality-red.csv")
df = df.drop(['fixed acidity', 'chlorides', 'free sulfur dioxide', 'pH', 'residual sugar'], axis=1)

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selected_features_dt = ['alcohol', 'density', 'sulphates']
selected_features_rf = ['alcohol', 'sulphates', 'density', 'citric acid']
selected_features_knn = ['alcohol', 'density', 'sulphates']

models = {
    'DecisionTree': (DecisionTreeClassifier(random_state=42), selected_features_dt),
    'RandomForest': (RandomForestClassifier(random_state=42), selected_features_rf),
    'KNN': (KNeighborsClassifier(), selected_features_knn)
}

results = {}
conf_matrices = {}

for name, (model, features) in models.items():
    model.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results[name] = {'Accuracy': acc, 'F1-score': f1}
    conf_matrices[name] = confusion_matrix(y_test, y_pred)

# Bar chart for Accuracy and F1-score
metrics = ['Accuracy', 'F1-score']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in models.keys()]
    axes[i].bar(models.keys(), values, color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1)
    for j, v in enumerate(values):
        axes[i].text(j, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Confusion Matrices heatmaps
for name in models.keys():
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrices[name], annot=True, fmt="d", cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
