from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast
from sklearn.model_selection import train_test_split
from src.datasets import BaseDataset
from pathlib import Path
import json

def read_file(data):
    ids = []
    contexts = []
    questions = []
    answers = []
    
    for datapoint in data:
        ids.append(datapoint["_id"])
        questions.append(datapoint["question"])
        answers.append(datapoint["answer"])
        datapoint_context = []
        for title in datapoint["context"]:
            for paragraph in title[1]:
                datapoint_context.append(paragraph)
        contexts.append(datapoint_context)
        #print(datapoint["question"])
        #print(datapoint["answer"])
        #print('\n\n')
    return contexts, questions, answers

filename = 'data/partial_data_100samples.json'
data = json.load(open(filename, 'r'))
train_data, val_data = train_test_split(data, test_size = 0.2)
train_contexts, train_questions, train_answers = read_file(train_data)
val_contexts, val_questions, val_answers = read_file(val_data)

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
