import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
class NerModelBert(nn.Module):
    def __init__(self, bert_model:BertModel,num_labels,hidden_size=768) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.hidden_size = hidden_size
        self.crf_layer = CRFLayer(num_labels)
        self.cls = nn.Linear(self.hidden_size,num_tags)

    def forward(self,inputs):
        x = self.bert_model(inputs)
        
    

class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super(CRFLayer,self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags,num_tags))
        self.crf = CRF(num_tags=self.num_tags,batch_first=True)
        self.crf.transitions = self.transitions
        
    def forward(self, inputs, sequence_lengths):
        # inputs: [batch_size, seq_len, num_features]
        # sequence_lengths: [batch_size]
        return inputs, sequence_lengths
    
    def decode(self,inputs,lengths):
        return self.crf.decode(inputs,lengths)