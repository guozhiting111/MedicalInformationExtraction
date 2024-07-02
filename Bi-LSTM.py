import numpy as np
import torch
import torch.nn as nn
import json
from my_utils import NERDataset,collate_fn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score


def eval(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = torch.tensor(batch[0])
            targets = torch.tensor(batch[1])
            len_list = batch[-1]
            predictions = model(inputs)
            predicted_labels = torch.argmax(predictions,dim=-1)
            predicted_labels = predicted_labels.permute([1,0])
            for l in range(len(len_list)):
                all_predictions.append(predicted_labels[l,0:len_list[l]-1].tolist())
                all_targets.append(targets[l,0:len_list[l]-1].tolist())
            # 收集预测标签和目标标签
            all_predictions = sum(all_predictions,[])
            all_targets=sum(all_targets,[])
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
    f1 = f1_score(all_targets, all_predictions)

    return f1
def train(model,epoch_num,loss_function,optimzer,dataloader):
    for epoch in range(epoch_num):
        model.train()
        for batch in dataloader:
            train_x = torch.tensor(batch[0])
            train_y = torch.tensor(batch[1])
            optimzer.zero_grad()
            pre_y = model(train_x)
            train_y = train_y.view(-1)
            loss = loss_function(pre_y.view(-1, pre_y.size(-1)), train_y)
            loss.backward()
            optimzer.step()
        f1 = eval(model,dataloader_eval)
        print('Epoch: {}, Loss: {:.5f}, F1 score: {:.5f}'.format(epoch + 1, loss.item(), f1))




class ModelLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,num_tolabel,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      )
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,bidirectional=True,
                            )
        self.fc = nn.Linear(2*hidden_size, num_tolabel)
        # [batch,len,emb]
        # [len,batch,emb]
    def forward(self,train_x):
        train_x = self.embedding(train_x)
        out,_ = self.lstm(train_x)
        out = self.fc(out.view(-1, out.size(-1)))
        return out



dataset_train = NERDataset('nlp2024-data/dataset/dev.json')
dataloader_train = DataLoader(dataset_train,batch_size=8,collate_fn=collate_fn)
dataset_eval = NERDataset('nlp2024-data/dataset/dev.json')
dataloader_eval = DataLoader(dataset_eval,batch_size=8,collate_fn=collate_fn)
with open('char2id.json','r',encoding='utf-8') as f:
    data = f.read()
    lenth = len(data)
with open('label2id.json','r',encoding='utf-8') as f:
    Data = f.read()
    num_tolabel = len(Data)
model = ModelLSTM(lenth,100,num_tolabel,100)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
epoch_num = 30
train(model,epoch_num,loss_function,optimizer,dataloader_train)