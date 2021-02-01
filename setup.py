import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW, Adafactor 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
batch_size = 1
verbose = False
optim = AdamW(model.parameters(), lr=5e-5)

