import pprint, torch
from eval_script import *
from datetime import datetime
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter



def train(train_loader, val_loader, model_type='longformer', optim=0, lr=5e-5, model=0, device=0, tokenizer=0, num_epochs=10, n=5):
    if model_type == 'longformer' and device == 0: from setup4L import device
    if model_type == 'longformer' and tokenizer == 0: from setup4L import tokenizer
    if model_type == 'longformer' and model == 0: from setup4L import model

    if model_type == 'distilbert' and device == 0: from setup import device
    if model_type == 'distilbert' and tokenizer == 0: from setup import tokenizer
    if model_type == 'distilbert' and model == 0: from setup import model

    if optim == 0: optim = AdamW(model.parameters(), lr=lr)
    summary_path = f'runs/{model_type}_{len(train_loader)}samples_{num_epochs}epochs_{lr}lr_{datetime.now().strftime("%b-%d-%Y-%H%M%S")}'
    writer = SummaryWriter(summary_path)

    em_scores_train = []
    f1_scores_train = []
    em_scores_val = []
    f1_scores_val = []
    for epoch in range(num_epochs):
        em_score_epoch_train = []
        f1_score_epoch_train = []
        em_score_epoch_val = []
        f1_score_epoch_val = []

        model.train()
        print(f'Commencing TRAINING for Epoch{epoch+1}/{num_epochs}')
        for i, batch in enumerate(train_loader):

            #########################
            ##### TRAINING STEP #####
            #########################

            # Uncomment the line below to use global attentions
            # global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, output_attentions=True, start_positions=start_positions, end_positions=end_positions)
            optim.zero_grad()
            loss, f1_score, em_score = get_scores(batch, em_score_epoch_val, f1_score_epoch_val,
                                                  device, tokenizer, model)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("f1_score/train", f1_score, epoch)
            writer.add_scalar("em_score/train", em_score, epoch)
            loss.backward()
            optim.step()
            if i%10 == 0: print(f'Step {i} - loss: {loss:.3} - f1_score: {f1_score:.3}')
            else: print(f'Step {i} - loss: {loss:.3}')

        model.eval()
        print(f'Commencing EVALUATION for Epoch {epoch+1}/{num_epochs}')
        for i, batch in enumerate(val_loader):

            ###########################
            ##### EVALUATION STEP #####
            ###########################

            loss, f1_score, em_score = get_scores(batch, em_score_epoch_val, f1_score_epoch_val,
                                                  device, tokenizer, model)
            writer.add_scalar("f1_score/val", f1_score, epoch)
            writer.add_scalar("em_score/val", em_score, epoch)
            print(f'Step {i} - f1_score: {f1_score:.3}')

        writer.add_scalar("Loss/val", loss, epoch)

        # Append the averages of the whole epoch to metric lists for train and val.
        em_scores_val.append(sum(em_score_epoch_val)/len(em_score_epoch_val))
        f1_scores_val.append(sum(f1_score_epoch_val)/len(f1_score_epoch_val))  
        em_scores_train.append(sum(em_score_epoch_train)/len(em_score_epoch_train))
        f1_scores_train.append(sum(f1_score_epoch_train)/len(f1_score_epoch_train))

    # Save model using variables as titles (including f1[-1])
    f1_score_train = f1_scores_train[-1]
    model_save_path = f'models/{len(train_loader)}samples_{num_epochs}epochs_{f1_score_train:.3}f1_{lr}lr_{datetime.now().strftime("%b-%d-%Y-%H%M%S")}'
    torch.save(model, model_save_path)
    print(f'TRAINING DONE. MODEL SAVED (lr:{lr})\n\n')

    return em_scores_train, f1_scores_train, em_scores_val, f1_scores_val


def get_scores(batch, em_score_epoch, f1_score_epoch, device, tokenizer, model):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True, start_positions=start_positions, end_positions=end_positions)
    nbest = likeliest_predictions(outputs.start_logits, outputs.end_logits, batch['input_ids'], tokenizer, n=5)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    target_tokens = all_tokens[start_positions:end_positions+1]
    target_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(target_tokens))
    em_score = max((em_metric(nbest[i].text, target_text)) for i in range(len(nbest)))
    f1_score = max((f1_metric(nbest[i].text, target_text)) for i in range(len(nbest)))
    em_score_epoch.append(em_score)
    f1_score_epoch.append(f1_score)
    loss = outputs.loss

    return loss, f1_score, em_score