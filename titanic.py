import os
import pandas as pd
import numpy as np
os.getcwd()
titanic = pd.read_csv("C:/Users/sihyun/Desktop/HandsOn/titanic/train.csv")
df = pd.DataFrame(titanic, columns=["Survived", "Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"])
df = df.dropna(axis=0, how='any')
y_train = df["Survived"]
X = df[["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]]
X_train = df[["Pclass", "SibSp", "Parch", "Fare"]]

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
L_enc = LabelEncoder()
O_enc = OneHotEncoder()
S = L_enc.fit_transform(X['Sex'])
E = L_enc.fit_transform(X['Embarked'])
S = S.reshape(-1, 1)
E = E.reshape(-1, 1)
X_train['Sex'] = O_enc.fit_transform(S)
X_train['Embarked'] = O_enc.fit_transform(E)


# SettingWithCopy warning 끄기
pd.set_option('mode.chained_assignment', None)

# X['Embarked'] = X['Embarked'].astype(str)
# X.loc[:, 'Embarked'] = X.loc[:, 'Embarked'].astype(str)
# X.dtypes
from sklearn.linear_model import SGDClassifier
y_train
X_train[["Embarked"]]
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
