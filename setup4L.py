import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering, AdamW, Adafactor
from datetime import datetime

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')
batch_size = 1
verbose = True
optim = AdamW(model.parameters(), lr=5e-5)

