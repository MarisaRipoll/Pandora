model.eval()


for i, sample in enumerate(val_loader):
    input_ids = sample['input_ids'].to(device)
    attention_mask = sample['attention_mask'].to(device)
    start_positions = sample['start_positions'].to(device)
    end_positions = sample['end_positions'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    score_diff, nbest = likeliest_predictions(outputs.start_logits, outputs.end_logits,
                                              sample['input_ids'], tokenizer, n=5)
    pprint.PrettyPrinter(indent=1, compact=True).pprint(nbest)
    pprint.PrettyPrinter(indent=1, compact=True).pprint(score_diff)
    for i in range(len(nbest)):
        print('i) ', nbest[i])
        print('nbest[i].text: ', nbest[i].text)
    print(val_answer_texts[i])
    break
    
    #print('id: ', val_dataset['val_ids'][i])
    #print('prediction_text: ', prediction_text)
    #### REFERENCE: ####
    #reference_text = {'text': [val_dataset['val_answers'][i]['text'], val_dataset['val_answers'][i]['text']]}
    #reference_text = {'text': [val_dataset['val_answers'][i]['text']]}
    #references = {'id': val_dataset['val_ids'][i], 'answers': reference_text}
    #references = [val_dataset['val_ids'][i], reference_text]
    #print('reference_text: ', reference_text)
    #score = squad_metric.add_batch(predictions=predictions, references=references)
    #score = squad_metric.add_batch(prediction_text, reference_text)
    #loss = outputs.loss
    #writer.add_scalar("Loss/train", loss, epoch)
    if i > 1:
        break


    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    prediction_tokens = all_tokens[torch.argmax(outputs.start_logits) :torch.argmax(outputs.end_logits)+1]
    prediction_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(prediction_tokens))
    