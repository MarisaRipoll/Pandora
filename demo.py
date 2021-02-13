import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW, Adafactor 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

model_default = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

def one_dimensional_attention_representation(text, question, model=0):
    # If model is not specified default to longformer checkpoint
    if model == 0: model = model_default
    encoding = tokenizer(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    tokens = input_ids.detach().cpu().tolist()[0]
    attention_mask = encoding["attention_mask"]
    outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_logits):torch.argmax(end_logits)+1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    attentions = []
    for i, torch_tensor_of_word in enumerate(outputs.global_attentions[0][0][-1]):
        if torch.sum(torch_tensor_of_word) == 0: break
        else: attentions.append(torch.sum(torch_tensor_of_word).item())
    question_indexes = [token for i, token in enumerate(tokens[1:tokens.index(2)])]
    text_indexes = [token for i, token in enumerate(tokens[tokens.index(2)+2:-1])]
    question_attentions = attentions[1:len(question_indexes)+1]
    factor = 100/max(question_attentions)
    question_attentions = [int(factor*attention) for attention in question_attentions]
    text_attentions = attentions[len(question_indexes)+3:-1]
    factor = 100/max(text_attentions)
    text_attentions = [int(factor*attention) for attention in text_attentions]
    attentions = question_attentions + text_attentions
    tokens = question_indexes + text_indexes
    words = [tokenizer.decode(token) for token in tokens]

    return words, attentions
