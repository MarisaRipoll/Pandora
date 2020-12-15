# The SQUAD tutoiral did not work with Longformer. One possibility is that it is meant for
# DistilBert so lets try with that. https://huggingface.co/transformers/master/custom_datasets.html

import torch
from torch.utils.data import DataLoader
#from datasets.squad_v2 import squad
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW
from transformers import Trainer, TrainingArguments
from SQuAD import obtain_dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# SETUP
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path1 = 'data/squad/train-v2.0.json'
path2 = 'data/squad/dev-v2.0.json'
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_dataset, val_dataset = obtain_dataset(path1, path2, num_samples_train=800, num_samples_val=200)
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
batch_size = 2
verbose = True
writer = SummaryWriter()

print('dataset type: ', type(train_dataset))
print('len(dataset): ', len(train_dataset))

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

# TRAINING
if verbose == True:
    print(f'Training...')
    
#print('train_dataset[0]:', train_dataset[0])
#print('train_dataset[1]:', train_dataset[1])


for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        predictions = outputs[1].argmax(dim=1)
        #accuracy, precision, recall, f1 = self.performance_measures(predictions, label,)
        # print(f'label: label, predictions: {predictions}')
        #print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1: {f1}')
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        optim.step()
        
writer.flush()
writer.close()

model.eval()
    
    
@staticmethod
def performance_measures(predictions, labels, averaging_method='macro'):
    if predictions.device.type == 'cuda' or labels.device.type == 'cuda':
        predictions, labels = predictions.cpu(), labels.cpu()

    # averaging for multiclass targets, can be one of [‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    accuracy = accuracy_score(labels, predictions)
    zero_division = 0
    precision = precision_score(
        labels, predictions, average=averaging_method, zero_division=zero_division)
    recall = recall_score(
        labels, predictions, average=averaging_method, zero_division=zero_division)
    f1 = f1_score(labels, predictions, average=averaging_method,
                  zero_division=zero_division)

    return accuracy, precision, recall, f1

