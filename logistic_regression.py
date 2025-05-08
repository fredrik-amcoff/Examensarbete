import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import RobustScaler

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

def run_model(train_set, eval_set, C, penalty, iter, threshold):
    #Split
    y_train = train_set["ai"]
    X_train = train_set.drop("ai", axis=1)

    y_eval = eval_set["ai"]
    X_eval = eval_set.drop("ai", axis=1)

    #standardize
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    #Initialize
    lr = LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=iter, random_state=42)

    #Train
    lr.fit(X_train, y_train)

    #Evaluate
    y_probs = lr.predict_proba(X_eval)[:, 1]
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

run_model(train_set=train_eng, eval_set=eval_swe, C=385.1107002325569, penalty="l1", iter=100, threshold=0.5)

#Hyperparams from random seach using 80/20 test split from training data and 100 iterations.

#ENG HYPERPARAMETERS: (train_set=train_eng, eval_set=eval_eng, C=385.1107002325569, penalty="l1", iter=100, threshold=0.5)
#'C': np.logspace(-3, 3, 20)
#penalty': ['l1', 'l2']
#max_iter': [50, 100, 200] (converged at 100)

#TRANSLATED HYPERPARAMETERS: (train_set=train_trans, eval_set=eval_swe, C=2.27697025538168, penalty="l1", iter=100, threshold=0.5)
#Same random search arguments