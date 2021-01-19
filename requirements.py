# The SQUAD tutoiral did not work with Longformer. One possibility is that it is meant for
# DistilBert so lets try with that. https://huggingface.co/transformers/master/custom_datasets.html

import torch
from torch.utils.data import DataLoader
#from datasets.squad_v2 import squad
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering, AdamW
from transformers import Trainer, TrainingArguments
from SQUAD4L import obtain_dataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
train_dataset, val_dataset = obtain_dataset(path1, path2, num_samples_train=800, num_samples_val=200)
model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')
batch_size = 2
verbose = True
writer = SummaryWriter()
model_save_path = f'models/{datetime.now().strftime("%b-%d-%Y-%H%M%S")}'
optim = AdamW(model.parameters(), lr=5e-5)

