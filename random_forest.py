import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#load training data
df_train_eng = pd.read_csv('text_stats_eng_train.csv')
df_train_trans = pd.read_csv('text_stats_trans_train.csv')

df_eval_eng = pd.read_csv('text_stats_eng_eval.csv')
df_eval_trans = pd.read_csv('text_stats_trans_eval.csv')
df_eval_sv = pd.read_csv('text_stats_sv_eval.csv')

#remove old features
df_train_eng = df_train_eng.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)
df_train_trans = df_train_trans.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)

df_eval_eng = df_eval_eng.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)
df_eval_trans = df_eval_trans.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)
df_eval_sv = df_eval_sv.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)

#remove non-numeric
df_train_eng = df_train_eng.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)
df_train_trans = df_train_trans.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)

df_eval_eng = df_eval_eng.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)
df_eval_trans = df_eval_trans.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)
df_eval_sv = df_eval_sv.drop(['title', 'words', 'chars'], axis=1)

def run_model(train_set, eval_set):
    #split
    y_train = train_set["ai"]
    X_train = train_set.drop("ai", axis=1)

    y_eval = eval_set["ai"]
    X_eval = eval_set.drop("ai", axis=1)

    #initialize model
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42) #set hyperparameters

    #train
    rf.fit(X_train, y_train)

    #evaluate ENG --> ENG
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


run_model(df_train_eng, df_eval_eng)