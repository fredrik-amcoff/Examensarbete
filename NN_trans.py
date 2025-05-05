import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#Load
df_train = pd.read_csv('text_statistics_trans.csv')
df_eval = pd.read_csv('text_statistics_swe.csv')

y_train = df_train["ai"].values
X_train = df_train.drop("ai", axis=1).values
y_eval = df_eval["ai"].values
X_eval = df_eval.drop("ai", axis=1).values

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  #HYPERPARAM: Batch size

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): #HYPERPARAM: number of layers
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  #First fully connected layer
        self.relu = nn.ReLU()                         #Fctivation function for above layer
        self.fc2 = nn.Linear(hidden_size, output_size)  #Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#Initialization
model = FeedforwardNN(input_size=X_train.shape[1], hidden_size=20, output_size=1) #HYPERPARAM: layer sizes
criterion = nn.BCEWithLogitsLoss() #HYPERPARAM: loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) #HYPERPARAM: learning rate, optimizer choice

#Training loop
model.train()
for epoch in range(100): #HYPERPARAM: Epochs
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
    preds = (probs > 0.5).int().numpy()

accuracy = accuracy_score(y_eval, preds)
precision = precision_score(y_eval, preds)
recall = recall_score(y_eval, preds)
f1 = f1_score(y_eval, preds)
conf_matrix = confusion_matrix(y_eval, preds)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

conf_matrix = confusion_matrix(y_eval, preds)
print(conf_matrix)