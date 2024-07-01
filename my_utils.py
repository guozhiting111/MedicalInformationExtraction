import torch
from torch.utils.data import DataLoader,Dataset
import json

label2id = json.load(open('label2id.json'))
char2id = json.load(open('char2id.json'))

class NERDataset(Dataset):
    def __init__(self,data_path) -> None:
        super().__init__()
        self.data_path = data_path
        data = json.load(open(data_path)) # 一个字典
        self.data = self.data_process(data)
    
    def data_process(self,data):
        sentences = []
        labels = []
        for key,val in data.items():
            dialogues = val['dialogue']
            for dialogue in dialogues:
                sentences.append(dialogue['sentence'])
                labels.append(dialogue['BIO_label'])
        return {'sentences':sentences,'labels':labels}
    
    def __len__(self):
        return len(self.data['sentences'])
    
    def __getitem__(self, index) -> list:
        sentence = self.data['sentences'][index]
        label = self.data['labels'][index]
        label = label.split()
        label = [label2id[c] for c in label]        
        return {"sentence":sentence,"label":label}
    
def collate_fn(batch):
    max_len = max([len(f['sentence']) for f in batch])
    sentence = []
    for f in batch:
        s = list(f['sentence']) + (max_len - len(f['sentence']))*['#']
        s = [char2id[c] for c in s]
        sentence.append(s)
    label = [s['label'] for s in batch]
    return sentence,label