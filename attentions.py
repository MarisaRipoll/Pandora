import torch
import torch.optim as optimizer_type
from train_script import train
from torch.utils.tensorboard import SummaryWriter
from transformers import LongformerForQuestionAnswering, LongformerTokenizerFast

model = LongformerForQuestionAnswering.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_different_windows_local(train_loader, val_loader, optim=0, writer=0, lr=0.005, eps=1e-08,
                                  model=model, attention_per_layer=0, attentions_list=0, device=device, num_epochs=3):
    if optim == 0: optim = optimizer_type.RMSprop(model.parameters(), lr=lr, eps=eps, centered=False)
    if attention_per_layer!=0 and attentions_list!=0: print('WARNING: YOU ARE USING attentions_list AND attention_per_layer. CHOOSE ONE.')
    if attentions_list!=0:
        for attention in attentions_list:
            if writer==0: writer=SummaryWriter('local_attention_of_window_{attention}')
            configuration = model.config
            configuration.attention_window = attention
            model.train()
            model = model.to(device)
            train(train_loader, val_loader, optim=optim, writer=writer, lr=lr, 
                  model=model, device=device, tokenizer=tokenizer, num_epochs=num_epochs)
    if attention_per_layer!=0:
        if writer==0: writer=SummaryWriter('local_attention_varied_by_layer')
        configuration = model.config
        configuration.attention_window = attention
        model.train()
        model = model.to(device)
        train(train_loader, val_loader, optim=optim, writer=writer, lr=lr, 
              model=model, device=device, tokenizer=tokenizer, num_epochs=num_epochs)
    
