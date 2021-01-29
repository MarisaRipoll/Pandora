import json, torch, pprint
import string, collections, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pathlib import Path
from matplotlib import rc
from transformers import LongformerTokenizerFast
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def data_reader(path, num_samples=-1, mode='train', verbose=False, verbose_limit=5):

    '''This function reads the dataset: SQUADv2 and obtains the significant information from the
    dataset dict structure. It also corrects the errors present in the dataset by readjusting the start
    and end positions when necessary. Finally it returns the required metadata collected in lists.
    
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
                    ids.append(sample_id)
                    answer_texts.append(text)
                    negatives.append(negative)
                    
                    # Before we add answer_starts lets make sure there are no errors.
                    # (Squad answers are sometimes off by a few characters.)
                    # In doing so we can also add answer_ends to the metadata :D
                    answer_end = answer_start + len(text)
                    if context[answer_start:answer_end] == text:
                        answer['answer_end'] = answer_end
                    elif context[answer_start-1:answer_end-1] == text: 
                        answer['answer_start'] = answer_start - 1
                        answer['answer_end'] = answer_end - 1
                    elif context[answer_start-2:answer_end-2] == text: 
                        answer['answer_start'] = answer_start - 2
                        answer['answer_end'] = answer_end - 2
                    else:
                        print('ERROR: SQUAD Answer does not match idxs')
                        answer_starts.append(answer_start)
                        answer_ends.append(answer_start)

                    answers.append(answer)

    if mode=='train':
        return contexts, questions, answers
    elif mode=='eval':
        return contexts, questions, answers, answer_texts
    elif mode=='all':
        return contexts, questions, answers, ids, answer_texts, negatives, titles
    else:
        print('ERROR: incorrect mode chosen!')


class Dataset(torch.utils.data.Dataset):
    '''
    This class inherits from the PyTorch Dataset module and thus converts the incoming
    data lists of dictionaries into an abstract dataset class of torch.tensors. This class
    maps keys to data samples. :)
    '''
    def __init__(self, data, answers, tokenizer):
        self.tokenizer = tokenizer
        start = []
        end = []
        for i in range(len(answers)):
            start.append(data.char_to_token(i, answers[i]['answer_start']))
            end.append(data.char_to_token(i, answers[i]['answer_end'] - 1))
            self.none_checker(start)
            self.none_checker(end)
        data.update({'start_positions': start, 'end_positions': end})
        self.data = data

    def none_checker(self, token):
        if token[-1] is None: token[-1] = self.tokenizer.model_max_length

    def __getitem__(self, idx):
        data = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        return data

    def __len__(self):
        return len(self.data.input_ids)
    

def obtain_dataset(path1, path2, num_samples_train=80, num_samples_val=20, verbose=False,
                   tokenizer=LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')):
    '''
    This function takes in the path of the json files of the train and dev SQUAD files and does all necessary
    data_preprocessing steps. Using the data_reader function and the Dataset class defined above. 
    '''
    
    train_contexts, train_questions, train_answers = data_reader(path1, num_samples=num_samples_train)
    val_contexts, val_questions, val_answers, val_answer_texts = data_reader(path2, num_samples=num_samples_val, mode='eval')

    if verbose==True:
        print(f'len(train_questions): {len(train_questions)}')
        print(f'len(train_contexts): {len(train_contexts)}')
        print(f'len(train_answers): {len(train_answers)}\n')
    
        print(f'len(val_questions): {len(val_questions)}')
        print(f'len(val_contexts): {len(val_contexts)}')
        print(f'len(val_answers): {len(val_answers)}\n')
    
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    
    if verbose==True: print('Passing inputs to the Dataset class')
    train_data = Dataset(train_encodings, train_answers, tokenizer)
    val_data = Dataset(val_encodings, val_answers, tokenizer)
    
    if verbose==True: print('Using sklearn split to return desired amount of data')
    train_percentage = num_samples_train/len(train_data)
    val_percentage = num_samples_val/len(val_data)
    train_dataset = train_data
    val_dataset = val_data
    
    return train_dataset, val_dataset


def get_input_length(path):

    '''This function computes the length in characters of all context+question input samples.
    It is a helper function for create_frequency_of_input_lengths_graph(). It returns input 
    lengths for the positive and the negative examples separately.
    
    Args:
            path (str): json file with data
    '''

    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    positive_inputs = []
    negative_inputs = []
    count = 0
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            len_context = len(context)
            for qa in passage['qas']:
                len_question = len(qa['question'])
                if bool(qa['answers']) == True:
                  # Answer exists --> This is a positive example
                  positive_inputs.append(len_context + len_question)
                else:
                  # Answer does not exist --> This is a negative example
                  negative_inputs.append(len_context + len_question)
                
    return positive_inputs, negative_inputs


def create_frequency_of_input_lengths_graph(path, title, bar_region_size=200, fsize=(15, 8), show_max_length=False):
    '''This function is used to create a graph showing the length frequency (in characters) for
    all context + question input samples in the given dataset.
    
    Args:
            path (str): json file with data
    '''
    positive_inputs, negative_inputs = get_input_length(path)
    positive_inputs = sorted(positive_inputs)
    negative_inputs = sorted(negative_inputs)
    if show_max_length==True:
        print('\n\n#####{0}#####\n'.format(title))
        print('The longest input character length is of: ', max(positive_inputs[-1], negative_inputs[-1]))
        print('This corresponds to an average of {0} tokens'.format(max(positive_inputs[-1], negative_inputs[-1])/4))
    len_names = max(positive_inputs[-2], negative_inputs[-2])//bar_region_size
    if max(positive_inputs[-2], negative_inputs[-2])%bar_region_size != 0: len_names += 1
    names = [int(bar_region_size*i + bar_region_size/2) for i in range(len_names)]
    positive_samples_per_length = [0] * len(names)
    negative_samples_per_length = [0] * len(names)
    y_pos = np.arange(len(names))

    for length in positive_inputs:
        for i in range(len(names)):
            if bar_region_size*i < length <= bar_region_size*i + bar_region_size:
                positive_samples_per_length[i] += 1

    for length in negative_inputs:
        for i in range(len(names)):
            if bar_region_size*i < length <= bar_region_size*i + bar_region_size:
                negative_samples_per_length[i] += 1

    plt.figure(figsize=fsize)
    p1 = plt.bar(y_pos, positive_samples_per_length, align='center', alpha=0.5)
    p2 = plt.bar(y_pos, negative_samples_per_length, align='center', alpha=0.5)

    plt.xticks(y_pos, names, rotation=90)
    plt.ylabel('Frequency')
    plt.xlabel('Length of Input Characters (Context + Question)')
    plt.title(title)
    plt.legend((p1[0], p2[0]), ('Positive Samples', 'Negative_samples'))

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