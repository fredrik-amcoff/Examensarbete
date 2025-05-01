import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#load training data
df = pd.read_csv('text_statistics_eng.csv')
#df = pd.read_csv('text_statistics_trans.csv')

#split off y
y = df["ai"]
X = df.drop("ai", axis=1)

#train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initialize model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

#train
rf.fit(X_train, y_train)

#evaluate
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)