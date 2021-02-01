import torch
from transformers import ReformerTokenizer, ReformerForQuestionAnswering, AdamW
import torch
from datetime import datetime

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
model = ReformerForQuestionAnswering.from_pretrained('google/reformer-crime-and-punishment')
batch_size = 1
verbose = True
optim = AdamW(model.parameters(), lr=5e-5)

