import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering, AdamW, Adafactor
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')
batch_size = 2
verbose = True
writer = SummaryWriter()
model_save_path = f'models/{datetime.now().strftime("%b-%d-%Y-%H%M%S")}'
optim = AdamW(model.parameters(), lr=5e-5)

