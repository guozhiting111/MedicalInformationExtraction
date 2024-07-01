import torch.utils
from my_utils import NERDatasetBert,collate_fn,collate_fn_bert
import torch
from torch.utils.data import DataLoader,Dataset
from model import NerModelBert
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics import F1Score
import numpy as np
from sklearn.metrics import f1_score
from transformers import BertModel,BertTokenizer
import json
from tqdm import tqdm
import torch.nn.functional as F

label2id = json.load(open('label2id.json',encoding='utf-8'))

def eval(model:NerModelBert,dev_dataset,batch_size):
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn_bert)
    model.eval()
    preds = []
    labels = []
    for batch in dev_loader:
        inputs = {"input_ids":batch[0],
                  "attention_mask":batch[1]}
        label = batch[2]
        len_list = batch[-1]
        with torch.no_grad():
            output = model(inputs) #[batch,length,num_labels]
            pred = model.decode(output)
        # pred = torch.argmax(output,dim=-1)
        for l in range(len(len_list)):
            preds.append(pred[l][1:len_list[l]-1])
            labels.append(label[l,1:len_list[l]-1].tolist())
    preds = sum(preds,[])
    labels = sum(labels,[])
    preds = np.array(preds)
    labels = np.array(labels)
    f1 = f1_score(labels,preds,average='micro')
    return f1
        
        


def train(model:NerModelBert,train_dataset,dev_dataset,epoch_num,batch_size,lr,num_labels):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr=lr)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=collate_fn_bert)
    
    loss_list = []
    for epoch in range(epoch_num):
        model.train()
        loss_total = 0
        for batch in tqdm(train_loader,desc="Training"):
            inputs = {"input_ids":batch[0],
                      "attention_mask":batch[1]}
            label = batch[2]
            optimizer.zero_grad()
            output = model(inputs)
            # loss = F.cross_entropy(output.view(-1,num_labels),label.view(-1),model.crf_layer.transitions)
            loss = -model.crf_layer.crf(output.permute([1,0,2]),label.permute([1,0]))
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            loss_total += loss.item()
        f1 = eval(model,dev_dataset,batch_size)
        print(f"epoch:{epoch},f1:{f1},loss:{loss_total}")
    
def main():
    num_labels = len(label2id)
    epoch_num = 10
    batch_size = 4
    lr = 1e-5
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    model = NerModelBert(bert_model=bert_model,num_labels=num_labels)
    train_dataset = NERDatasetBert('nlp2024-data/dataset/small_train.json',tokenizer)
    dev_dataset = NERDatasetBert('nlp2024-data/dataset/small_dev.json',tokenizer)
    train(model,train_dataset,dev_dataset,epoch_num,batch_size,lr,num_labels)

if __name__ == "__main__":
    main()
            
    