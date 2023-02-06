
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import BertPreTrainedModel, BertModel

class BertForTextClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.hidden_dim = config.hidden_size
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.fc = nn.Linear(self.hidden_dim, 3)

	def forward(self, input_ids, attention_mask):
		'''Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)'''
		
		#Feed the input to Bert model to obtain outputs
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		#Obtain the representations of [CLS] heads
		cls_reps = outputs.last_hidden_state[:, 0]
		cls_reps = self.dropout(cls_reps)
		logits = self.fc(cls_reps)
		logits = softmax(logits)
		return logits
