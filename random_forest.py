import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#load training data
train_eng = pd.read_csv('text_stats_eng_train.csv')
train_trans = pd.read_csv('text_stats_trans_train.csv')

eval_eng = pd.read_csv('text_stats_eng_eval.csv')
eval_trans = pd.read_csv('text_stats_trans_eval.csv')
eval_swe = pd.read_csv('text_stats_sv_eval.csv')

#remove old features
train_eng = train_eng.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)
train_trans = train_trans.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)

eval_eng = eval_eng.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)
eval_trans = eval_trans.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)
eval_swe = eval_swe.drop(['char_std', 'word_std', 'temporal_burstiness', 'syntactic_burstiness', 'wd_burstiness', 'semantic_burstiness'], axis=1)

#remove non-numeric
train_eng = train_eng.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)
train_trans = train_trans.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)

eval_eng = eval_eng.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)
eval_trans = eval_trans.drop(['title', 'topic', 'section', 'words', 'chars'], axis=1)
eval_swe = eval_swe.drop(['title', 'words', 'chars'], axis=1)

def run_model(train_set, eval_set, n, depth, threshold):
    #split
    y_train = train_set["ai"]
    X_train = train_set.drop("ai", axis=1)

    y_eval = eval_set["ai"]
    X_eval = eval_set.drop("ai", axis=1)

    #initialize model
    rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)

    #train
    rf.fit(X_train, y_train)

    #evaluate
    y_probs = rf.predict_proba(X_eval)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    conf_matrix = confusion_matrix(y_eval, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(conf_matrix)


run_model(train_set=train_trans, eval_set=eval_trans, n=89, depth=18, threshold=0.5)

#Hyperparams from random seach using 80/20 test split from training data and 100 iterations.

#ENG HYPERPARAMETERS: (train_set=train_eng, eval_set=eval_eng, n=89, depth=18, threshold=0.5)
#Generally little difference for different parameters (except very low values)
#'n_estimators': range(1, 500, 1)
#'max_depth': range(1, 100, 1)

#TRANSLATED HYPERPARAMETERS: (train_set=train_trans, eval_set=eval_swe, n=56, depth=33, threshold=0.5)
#Same random search arguments