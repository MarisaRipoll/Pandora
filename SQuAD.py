# https://huggingface.co/transformers/master/custom_datasets.html

import json, torch
from pathlib import Path
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def read_squad(path, num_samples=-1):

    '''This function reads the squad dataset
    
    Args:
            path (str): json file with data
            num_samples (int): number of samples to use from dataset. If set to -1 use all.
    '''
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    count = 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            #print('count: ', count)
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    if count >= num_samples and num_samples != -1:
                        break
                    count = count + 1
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

            
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
    
def obtain_dataset(path1, path2, num_samples_train=80, num_samples_val=20):
    train_contexts, train_questions, train_answers = read_squad(path1, num_samples=num_samples_train)
    val_contexts, val_questions, val_answers = read_squad(path2, num_samples=num_samples_val)
    
    print(f'len(train_questions): {len(train_questions)}')
    print(f'len(train_contexts): {len(train_contexts)}')
    print(f'len(train_answers): {len(train_answers)}\n')
    
    print(f'len(val_questions): {len(val_questions)}')
    print(f'len(val_contexts): {len(val_contexts)}')
    print(f'len(val_answers): {len(val_answers)}\n')
    
    print('now add_end_idx')
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)
    
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)
    
    print('obtaining data')
    train_data = SquadDataset(train_encodings)
    val_data = SquadDataset(val_encodings)
    train_percentage = num_samples_train/len(train_data)
    val_percentage = num_samples_val/len(val_data)
    
    #print('train_data[0]:', train_data[0])
    #print('train_data[1]:', train_data[1])
    
    print('split dataset')
    train_dataset = train_data
    val_dataset = val_data
    #train_dataset = train_test_split(train_data, test_size=train_percentage)
    #val_dataset = train_test_split(val_data, test_size=val_percentage)
    
    return train_dataset, val_dataset
    
""" 
train_contexts, train_questions, train_answers = read_squad('data/squad/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('data/squad/dev-v2.0.json')

add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

train_data = SquadDataset(train_encodings)
val_data = SquadDataset(val_encodings)
train_dataset = train_test_split(train_data, test_size=0.01)
val_dataset = train_test_split(val_data, test_size=0.01)


model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
"""
"""
batch_size = 1

training_args = TrainingArguments(
    output_dir='./results',                  # output directory
    num_train_epochs=3,                      # total number of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=10,                         # n° warmup steps for learning rate scheduler
    weight_decay=0.01,                       # strength of weight decay
    logging_dir='./logs',                    # directory for storing logs
    logging_steps=10,
)


trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
"""


