import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # This will suppress all warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


df = pd.read_csv(r"C:\Users\Yashvardhan Singh\Downloads\archive (2)\winequality-red.csv")


df = df.drop(['fixed acidity', 'chlorides', 'free sulfur dioxide', 'pH', 'residual sugar'], axis=1)

X = df.drop('quality', axis=1)
y = df['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Remaining columns:", X_train.columns.tolist())


models = {
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}


for name, model in models.items():
    print(f"\nFeature Selection using {name}:")
    sfs = SFS(model,
              k_features='best',
              forward=True,
              floating=False,
              scoring='accuracy',
              cv=5)
    sfs = sfs.fit(X_train, y_train)

    selected_features = list(sfs.k_feature_names_)
    print("Selected Features:", selected_features)
