import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif, mutual_info_classif


file_path =r"C:\Users\Yashvardhan Singh\Downloads\archive (2)\winequality-red.csv"


df = pd.read_csv(file_path)

X = df.drop('quality', axis=1)
y = df['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fischer test
f_scores, p_values = f_classif(X_train, y_train)

# mutial info test
mi_scores = mutual_info_classif(X_train, y_train)


scores_df = pd.DataFrame({
    'Feature': X_train.columns,
    'F_score': f_scores,
    'Mutual_Info': mi_scores
})


scores_df = scores_df.sort_values(by='F_score', ascending=False)

print(scores_df)
