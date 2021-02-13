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
            summary_path = f'runs/local_attention_of_window_{attention}'
            if writer==0: writer=SummaryWriter(summary_path)
            configuration = model.config
            configuration.attention_window = attention
            model.train()
            model = model.to(device)
            train(train_loader, val_loader, optim=optim, writer=writer, lr=lr, 
                  model=model, device=device, tokenizer=tokenizer, num_epochs=num_epochs, summary_path=summary_path)
    if attention_per_layer!=0:
        if writer==0: writer=SummaryWriter('runs/local_attention_varied_by_layer')
        configuration = model.config
        configuration.attention_window = attention_per_layer
        model.train()
        model = model.to(device)
        train(train_loader, val_loader, optim=optim, writer=writer, lr=lr, 
              model=model, device=device, tokenizer=tokenizer, num_epochs=num_epochs, summary_path='local_attention_varied_by_layer')
    

def train_local_vs_global(train_loader, val_loader, optim=0, writer=0, lr=0.005, eps=1e-08,
                          model=model, attention_value=512, device=device, num_epochs=3):
    if optim == 0: optim = optimizer_type.RMSprop(model.parameters(), lr=lr, eps=eps, centered=False)
    ## First Local attention:
    summary_path_local = f'runs/{num_epochs}epochs_{len(train_loader)}samples_LOCAL'
    if writer==0: writerlocal=SummaryWriter(summary_path_local)
    else: writerlocal = writer
    configuration = model.config
    configuration.attention_window = attention_value
    model.train()
    model = model.to(device)
    train(train_loader, val_loader, optim=optim, writer=writerlocal, lr=lr, model=model, device=device,
          tokenizer=tokenizer, num_epochs=num_epochs, summary_path=summary_path_local, local_only=True)

    ## Second Global attention:
    summary_path_global = f'runs/{num_epochs}epochs_{len(train_loader)}samples_GLOBAL'
    if writer==0: writerglobal=SummaryWriter(summary_path_global)
    else: writerglobal = writer
    model.train()
    model = model.to(device)
    train(train_loader, val_loader, optim=optim, writer=writerglobal, lr=lr, model=model, device=device,
          tokenizer=tokenizer, num_epochs=num_epochs, summary_path=summary_path_global, global_only=True)