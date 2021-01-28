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

    '''This function reads the dataset: SQUADv2 and obtains the significant information from
    sample structure. It returns 
    
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


# Inherits from pytorchs Dataset module: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset  
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, answers, tokenizer):
        self.answers = answers
        self.tokenizer = tokenizer
        start = []
        end = []
        for i in range(len(answers)):
            start.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            none_checker(start)
            none_checker(end)
        encodings.update({'start_positions': start, 'end_positions': end})
        self.encodings = encodings

    def none_checker(list):
        if list[-1] is None: list[-1] = self.tokenizer.model_max_length

    def __getitem__(self, idx):
        data = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return data

    def __len__(self):
        return len(self.encodings.input_ids)
    
def obtain_dataset(path1, path2, num_samples_train=80, num_samples_val=20, verbose=False,
                   tokenizer=LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')):
    train_contexts, train_questions, train_answers = data_reader(path1, num_samples=num_samples_train)
    val_contexts, val_questions, val_answers, val_answer_texts = data_reader(path2, num_samples=num_samples_val, mode='eval')

    if verbose==True:
        print(f'len(train_questions): {len(train_questions)}')
        print(f'len(train_contexts): {len(train_contexts)}')
        print(f'len(train_answers): {len(train_answers)}\n')
    
        print(f'len(val_questions): {len(val_questions)}')
        print(f'len(val_contexts): {len(val_contexts)}')
        print(f'len(val_answers): {len(val_answers)}\n')
    
        print('now add_end_idx')
    
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    
    if verbose==True: print('obtaining data')
    train_data = Dataset(train_encodings, train_answers, tokenizer)
    val_data = Dataset(val_encodings, val_answers, tokenizer)
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

# The eval and metric functions are heavily influenced by the code in: 
# https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html

def likeliest_predictions(start, end, input_ids, tokenizer, n=5):
    start  = start.detach().cpu().tolist()[0] # covert to one dimensional list
    end    = end.detach().cpu().tolist()[0]   # covert to one dimensional list
    inputs = input_ids.detach().cpu().tolist()[0]

    start_idx = [i for i, logit in sorted(enumerate(start), key=lambda x: x[1], reverse=True)[:n]]
    end_idx = [i for i, logit in sorted(enumerate(end), key=lambda x: x[1], reverse=True)[:n]]

    PrelimPrediction = collections.namedtuple("PrelimPrediction", ["start_idx", "end_idx", "start_logit", "end_logit"])
    BestPrediction = collections.namedtuple("BestPrediction", ["text", "start_logit", "end_logit"])
    prelim_preds = []
    nbest = []
    seen_preds = []
    for start_index in start_idx:
        for end_index in end_idx:
            if end_index < start_index: continue
            prelim_preds.append(PrelimPrediction(start_idx = start_index, end_idx = end_index,
                                                 start_logit = start[start_index], end_logit = end[end_index]))
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    for pred in prelim_preds:
        if len(nbest) >= n: break
        if pred.start_idx > 0: # non-null answers have start_idx > 0
            text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs[pred.start_idx:pred.end_idx+1]))
            text = text.strip()
            text = " ".join(text.split())
            if text in seen_preds:continue
            seen_preds.append(text)
            nbest.append(BestPrediction(text=text, start_logit=pred.start_logit, end_logit=pred.end_logit))
    nbest.append(BestPrediction(text="", start_logit=start[0], end_logit=end[0])) # Include null answer.
    # compute the difference between the null score and the best non-null score
    score_diff = start[0] + end[0] - nbest[0].start_logit - nbest[0].end_logit
    return score_diff, nbest


def em_metric(prediction, target):
    # Punctuation, case, space and article normalization
    prediction = prediction.lower()
    prediction = "".join(char for char in prediction if char not in set(string.punctuation))
    prediction = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", prediction)
    prediction = " ".join(prediction.split())

    # Punctuation, case, space and article normalization   
    target = target.lower() 
    target = "".join(char for char in target if char not in set(string.punctuation))
    target = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", target)
    target = " ".join(target.split())

    # Check if prediction and targets is the same:
    if prediction == target: return 1
    else: return 0

def f1_metric(prediction, target):
    # Punctuation, case, space and article normalization
    prediction = prediction.lower()
    prediction = "".join(char for char in prediction if char not in set(string.punctuation))
    prediction = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", prediction)
    prediction = " ".join(prediction.split())
    prediction_tokens = prediction.split()

    # Punctuation, case, space and article normalization    
    target = target.lower() 
    target = "".join(char for char in target if char not in set(string.punctuation))
    target = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", target)
    target = " ".join(target.split())
    target_tokens = target.split()

    if len(prediction_tokens) == 0 or len(target_tokens) == 0:
        if prediction_tokens == target_tokens: return 1
        else: return 0
    
    common_tokens = set(prediction_tokens) & set(target_tokens)
    if len(common_tokens) == 0: return 0  # None of the tokens are shared between target and prediction --> f1=0
     
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(target_tokens)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score