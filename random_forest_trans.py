import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#load training data
df_train = pd.read_csv('text_statistics_trans.csv')
df_eval = pd.read_csv('text_statistics_swe.csv') #Can use all 10k articles for trainin because of eval split

#split off y
y_train = df_train["ai"]
X_train = df_train.drop("ai", axis=1)

y_eval = df_eval["ai"]
X_eval = df_eval.drop("ai", axis=1)

#initialize model
rf = RandomForestClassifier(n_estimators=5, max_depth=20, random_state=42) #set hyperparameters

#train
rf.fit(X_train, y_train)

#evaluate
y_pred = rf.predict(X_eval)

accuracy = accuracy_score(y_eval, y_pred)
precision = precision_score(y_eval, y_pred)
recall = recall_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_eval, y_pred)
print(conf_matrix)