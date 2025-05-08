import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

def run_model(train_set, eval_set, batch, rate, epochs, hidden_1, hidden_2, threshold):
    #split
    y_train = train_set["ai"].values
    X_train = train_set.drop("ai", axis=1).values
    y_eval = eval_set["ai"].values
    X_eval = eval_set.drop("ai", axis=1).values

    #Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    #Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_eval = torch.tensor(X_eval, dtype=torch.float32)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32).unsqueeze(1)

    #Prepare dataset and dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    class FeedforwardNN(nn.Module):                                                        #HYPERPARAM: number of layers, activation functions
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
            super(FeedforwardNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size_1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size_2, output_size)

        def forward(self, x):
                out = self.fc1(x)
                out = self.relu1(out)
                out = self.fc2(out)
                out = self.relu2(out)
                out = self.fc3(out)
                return out

    #Initialization
    model = FeedforwardNN(input_size=X_train.shape[1], hidden_size_1=hidden_1, hidden_size_2=hidden_2, output_size=1)
    criterion = nn.BCEWithLogitsLoss()                                                                      #HYPERPARAM: loss function
    optimizer = optim.Adam(model.parameters(), lr=rate)                                                     #HYPERPARAM: optimizer choice

    #Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    #Eval
    model.eval()
    with torch.no_grad():
        logits = model(X_eval)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int().numpy()

    accuracy = accuracy_score(y_eval, preds)
    precision = precision_score(y_eval, preds)
    recall = recall_score(y_eval, preds)
    f1 = f1_score(y_eval, preds)
    conf_matrix = confusion_matrix(y_eval, preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(conf_matrix)

run_model(train_set=train_trans, eval_set=eval_trans, batch=32, rate=0.0002094937373001936, epochs=50, hidden_1=64, hidden_2=128, threshold=0.5)

#Hyperparams from random seach using 80/20 test split from training data and 20 iterations.

#ENG HYPERPARAMETERS: (train_set=train_eng, eval_set=eval_eng, batch=32, rate=0.0002094937373001936, epochs=50, hidden_1=64, hidden_2=128, threshold=0.5)
#batch_size = random.choice([16, 32, 64, 128])
#lr = 10**np.random.uniform(-4, -2)
#epochs = random.choice([50, 100, 150, 200])
#h1 = random.choice([32, 64, 128])
#h2 = random.choice([32, 64, 128])

#TRANSLATED HYPERPARAMETERS: (train_set=train_eng, eval_set=eval_eng, batch=128, rate=0.0008701372822602438, epochs=100, hidden_1=32, hidden_2=64, threshold=0.5)
#Same random search arguments