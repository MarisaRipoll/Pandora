# https://huggingface.co/transformers/master/custom_datasets.html

import json, torch, pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pathlib import Path
from matplotlib import rc
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def data_reader(path, num_samples=-1, mode='train', verbose=False, verbose_limit=5):

    '''This function reads the dataset: SQUADv2
    
    Args:
            path (str): json file with data
            num_samples (int): number of samples to use from dataset. If set to -1 use all.
            mode (string): mode can be 'train', 'eval' or 'all'. And it controls the number  
                           and type of variables that the function returns.
            verbose (bool): gives extra information about the sample contents and structure.
            verbose_limit (int): max number of samples for which verbose remains True.
    '''
    # For readability's sake and to avoid wasting time and resources with excessive printing,
    # we set a maximum number of lines for which verbose remains true -> verbose_limit.
    if verbose == True and (num_samples > verbose_limit or num_samples==-1):
        print('WARNING: num_samples in data_reader exceeds the verbose_limit. Verbose will be reset to False')
         
    path = Path(path)
    with open(path, 'rb') as f: dataset = json.load(f)

    # For training we need contexts, questions and answers:
    contexts = []
    questions = []
    answers = []

    # For evaluation we need id, answers and answers[answer_starts]
    ids = []
    answer_starts = []
    answer_ends = []
    answer_texts = []

    # Extra information we could need (the rest of the metadata)
    titles = []
    negatives = []
    count = 0

    if verbose == True:
        print('Dataset version: ', dataset['version'])
        print('data length: ', len(dataset['data']))
        for i, dict_key in enumerate(dataset['data']):
            print(dict_key)
            break
        #pprint.PrettyPrinter(indent=1, compact=True).pprint(data)
        #pprint.PrettyPrinter(indent=1, compact=True).pprint(data['data'][0])

    for data in dataset['data']:
        document_title = data['title']
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                sample_id = qa['id']
                negative = qa['is_impossible']
                for answer in qa['answers']:
                    # One question may have different answers and each of them counts as a separate sample.
                    # Therefore the limit number of samples is checked here.
                    if count >= num_samples and num_samples != -1:
                        break
                    count = count + 1
                    answer_start = answer['answer_start']
                    text = answer['text']

                    titles.append(document_title)
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                    ids.append(sample_id)
                    answer_texts.append(text)
                    negatives.append(negative)
                    
                    # Before we add answer_starts lets make sure there are no errors.
                    # (Squad answers are sometimes off by a few characters.)
                    # In doing so we can also add answer_ends to the metadata :D
                    answer_end = answer_start + len(text)
                    if context[answer_start:answer_end] == text:
                        answer_starts.append(answer_start)
                        answer_ends.append(answer_end)
                    elif context[answer_start-1:answer_end-1] == text: 
                        answer_starts.append(answer_start-1)
                        answer_ends.append(answer_end-1)
                    elif context[answer_start-2:answer_end-2] == text: 
                        answer_starts.append(answer_start-2)
                        answer_ends.append(answer_end-2)
                    else:
                        print('ERROR: SQUAD Answer does not match idxs')
                        answer_starts.append(answer_start)
                        answer_ends.append(answer_start)


    if mode=='train':
        return contexts, questions, answers
    elif mode=='eval':
        return contexts, questions, ids, answer_starts, answer_ends, answer_texts
    elif mode=='all':
        return contexts, questions, answers, ids, answer_starts, answer_ends, answer_texts, negatives, titles
    else:
        print('ERROR: incorrect mode chosen!')


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

            
def add_token_positions(encodings, answers, tokenizer):
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
    
def obtain_dataset(path1, path2, num_samples_train=80, num_samples_val=20, verbose=False,
                   tokenizer=LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')):
    train_contexts, train_questions, train_answers = data_reader(path1, num_samples=num_samples_train)
    val_contexts, val_questions, val_answers = data_reader(path2, num_samples=num_samples_val)
    
    if verbose==True:
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
    
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)
    
    if verbose==True: print('obtaining data')
    train_data = SquadDataset(train_encodings)
    val_data = SquadDataset(val_encodings)
    train_percentage = num_samples_train/len(train_data)
    val_percentage = num_samples_val/len(val_data)
    
    if verbose==True: print('split dataset')
    train_dataset = train_data
    val_dataset = val_data
    
    return train_dataset, val_dataset


def get_input_length(path):

    '''This function computes the length in characters of all context+question input samples.
    It is a helper function for create_frequency_of_input_lengths_graph()
    
    Args:
            path (str): json file with data
    '''
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    positive_inputs = []
    # negative_inputs = []
    count = 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            len_context = len(context)
            for qa in passage['qas']:
                len_question = len(qa['question'])
                positive_inputs.append(len_context + len_question)
                ####### TODO: negative examples - stack them in a bar chart with a separate color. 
                #negative_inputs.append(len_context + len_question)
                #for answer in qa['answers']:
                #    inputs.append(len_context + len_question)
    return positive_inputs    # , negative_inputs


def create_frequency_of_input_lengths_graph(path, bar_region_size=200, fsize=(15, 8), show_max_length=False):
    '''This function is used to create a graph showing the length frequency (in characters) for
    all context + question input samples in the given dataset.
    
    Args:
            path (str): json file with data
    '''
    inputs = get_input_length(path)
    inputs = sorted(inputs)
    if show_max_length==True:
        print('The longest input character length is of: ', inputs[-1])
        print('This corresponds to an average of {0} tokens'.format(inputs[-1]/4))
    len_names = inputs[-2]//bar_region_size
    if inputs[-2]%bar_region_size != 0: len_names += 1
    names = [int(bar_region_size*i + bar_region_size/2) for i in range(len_names)]
    samples_per_length = [0] * len(names)
    y_pos = np.arange(len(names))

    for length in inputs:
        for i in range(len(names)):
            if bar_region_size*i < length <= bar_region_size*i + bar_region_size:
                samples_per_length[i] += 1

    plt.figure(figsize=fsize)
    plt.bar(y_pos, samples_per_length, align='center', alpha=0.5)
    plt.xticks(y_pos, names, rotation=90)
    plt.ylabel('Frequency')
    plt.xlabel('Length of Input Characters (Context + Question)')
    plt.title('Frequency of Input Lengths for SQUADv2')

    plt.show()


def show_squad_dataset_info():
    fig = go.Figure(data=[go.Table(columnwidth=[2, 1, 1, 1],
                                   header=dict(values=['', 'Train', 'Dev', 'Test']),
                                   cells=dict(values=[['Total examples',
                                                       'Negative examples',
                                                       'Total Articles',
                                                       'Articles with negatives'],
                                                      [130.319, 43.498, 442, 285], 
                                                      [11.873, 5.945, 35, 35], 
                                                      [8.862, 4.332, 28, 28]]))]) 
    fig.show()

