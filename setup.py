import torch
from torch.utils.data import DataLoader
#from datasets.squad_v2 import squad
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW, Adafactor 
from transformers import Trainer, TrainingArguments
from SQUAD4L import obtain_dataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
batch_size = 2
verbose = True
writer = SummaryWriter()
model_save_path = f'models/{datetime.now().strftime("%b-%d-%Y-%H%M%S")}'
optim = AdamW(model.parameters(), lr=5e-5)

