print('Importing libraries...')

from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
from Model import BaseLineModel, CrossModalAttentionModel
from tools import compute_Chiara_Loss, contrastive_loss, negative_sampling_contrastive_loss, replace_by_negative
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizer
import torch
from torch import optim
import time
import os
from tqdm import tqdm
from datetime import datetime
import random

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print('Defining model...')

random.seed(42)
model_type = 'baseline'
print('Model type: {}'.format(model_type))
loss_type = 'contrastive'
print('Loss type: {}'.format(loss_type))
model_name = 'scibert'
print('Using model: {}'.format(model_name))
tokenizer_name = 'scibert'
print('Using tokenizer: {}'.format(tokenizer_name))

if tokenizer_name == 'distilbert-base-uncased':
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
elif tokenizer_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
elif tokenizer_name == 'scibert':
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
print('Loading data...')
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
print('Creating datasets...')
# val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
# train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

if_crossmodal=False
if model_type=='crossmodal':
    if_crossmodal=True

val_dataset = GraphTextDataset(root='data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='data/', gt=gt, split='train', tokenizer=tokenizer)

print('Setting device...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

nb_epochs = 40
batch_size = 32
learning_rate =5e-6
dropout=0.3
print('The learning rate is:',learning_rate)
print('The dropout is:',dropout)
print('Creating dataloaders...')

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

print('Creating model...')

if model_type == 'baseline':    model = BaseLineModel(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=768,dropout=dropout) # nout = bert model hidden dim
if model_type == 'crossmodal':  model = CrossModalAttentionModel(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=768)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

print('Starting training...')

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    print('Training...')
    model.train()
    for data in tqdm(train_loader):
        batch, cids = data
        input_ids = batch.input_ids
        _ = batch.pop('input_ids')
        attention_mask = batch.attention_mask
        _ = batch.pop('attention_mask')
        graph_batch = batch
        if model_type == 'baseline':
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            if loss_type == 'contrastive':  
                current_loss = contrastive_loss(x_graph, x_text)   
            elif loss_type == 'Chiara':
                #1 :current_loss, _ = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//2)
                #2 :current_loss, _ = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//3)
                #3 :current_loss, _ = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//4)
                #4 :current_loss, _ = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//5)
                current_loss, _ = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=1)
        if model_type == 'crossmodal':
            #Let's randomly choose half of the batch
            #we keep one half as it is, and we put the 
            #graph of the first one as the graphs of the second one.
            #Then we apply a negative sampling contrastive loss.
            #We do this to prevent the cross modal from training without
            #taking into account the text input.
            new_batch, new_graph_x, new_edge_index, new_ptr, labels = replace_by_negative(graph_batch.batch, graph_batch.x, graph_batch.edge_index, len(input_ids), len(input_ids)//2)     
            graph_batch.batch = new_batch
            graph_batch.x = new_graph_x
            graph_batch.edge_index = new_edge_index
            graph_batch.ptr = new_ptr
            x_graph, x_text = model(graph_batch.to(device),input_ids.to(device),attention_mask.to(device))
            current_loss, pred = negative_sampling_contrastive_loss(x_graph, x_text, labels)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 
    print('Evaluating...')
    model.eval()       
    val_loss = 0   
    val_acc = 0     
    for data in tqdm(val_loader):
        batch, cids = data
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        if loss_type == 'contrastive': 
            current_loss = contrastive_loss(x_graph, x_text)   
        elif loss_type == 'Chiara':
            #1 :current_loss, current_acc = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//2)
            #2 :current_loss, current_acc = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//3)
            #3 :current_loss, current_acc = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//4)
            #4 :current_loss, current_acc = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids)//5)
            #5: current_loss, current_acc = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=len(input_ids))
            current_loss, current_acc = compute_Chiara_Loss(x_graph, x_text, CL_neg_samples=1)
            val_acc += current_acc
        val_loss += current_loss.item()       
    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    if best_validation_loss==val_loss:
        print('validation loss improoved saving checkpoint...')
        c = datetime.now()
        current_time = c.strftime('%H_%M_%S')
        save_path = os.path.join('./ensemble_models/', 'baseline_scibert_dropout_' + str(dropout) + '_lr_' + str(learning_rate) + current_time + '_epoch_' + str(i+1) + '.pt')
        torch.save({
        'epoch': i,
        'model':model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer,
        'validation_accuracy': val_loss,
        'loss': loss,
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))


print("Training finished!")