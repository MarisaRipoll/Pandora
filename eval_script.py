# The eval and metric functions are heavily influenced by the code in: 
# https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html

import collections, string, re

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
    question_indices = [i+1 for i, token in enumerate(inputs[1:inputs.index(2)])]
    for start_index in start_idx:
        for end_index in end_idx:
            if end_index < start_index: continue
            if start_index in question_indices: continue
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
    return nbest


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
    prediction_words = prediction.split()

    # Punctuation, case, space and article normalization    
    target = target.lower() 
    target = "".join(char for char in target if char not in set(string.punctuation))
    target = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", target)
    target = " ".join(target.split())
    target_words = target.split()

    if len(prediction_words) == 0 or len(target_words) == 0:
        if prediction_words == target_words: return 1
        else: return 0
    
    common_words = set(prediction_words) & set(target_words)
    if len(common_words) == 0: return 0  # None of the words are shared between target and prediction --> f1=0
     
    precision = len(common_words) / len(prediction_words)
    recall = len(common_words) / len(target_words)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score