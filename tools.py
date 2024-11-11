import torch.nn.functional as F
import torch
import numpy as np
import random
import os

#Losses

CE = torch.nn.CrossEntropyLoss()
BCEL = torch.nn.BCEWithLogitsLoss()

def contrastive_loss(v1, v2):
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def Chiara_Loss(X, Y, CL_neg_samples,T=0.01,SSL_loss='EBM_NCE',normalize=True):
    if normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if SSL_loss == 'EBM_NCE':
        criterion = BCEL
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(CL_neg_samples)], dim=0)
        neg_X = X.repeat((CL_neg_samples, 1))
        pred_pos = torch.sum(X * Y, dim=1) / T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / T
        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + CL_neg_samples * loss_neg) / (1 + CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif SSL_loss == 'InfoNCE':
        criterion = CE
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        # CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception
    # return CL_loss
    return CL_loss, CL_acc

def compute_Chiara_Loss(X, Y, CL_neg_samples):
    CL_loss, CL_acc = Chiara_Loss(X, Y, CL_neg_samples)
    CL_loss_opp, CL_acc_opp = Chiara_Loss(Y, X, CL_neg_samples)
    return ((CL_loss + CL_loss_opp) / 2,(CL_acc + CL_acc_opp) / 2)

def negative_sampling_contrastive_loss(v1, v2, labels):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye), logits.diag() > 0

# Other functions

def replace_by_negative(batch, graph_x, edge_index, total_amount, positive_amount):

    all_indices = list(np.arange(total_amount))
    positive_indices = list(np.random.choice(total_amount, positive_amount, replace=False))
    
    new_batch = []
    new_graph_x = []
    new_edge_index = [[],[]]
    new_ptr = [0]
    labels = torch.zeros(total_amount)
    k = 0
    l = 0
    for i,index in enumerate(all_indices):
        if index in positive_indices:
            temp = graph_x[torch.where(batch==index)]
            new_graph_x.append(temp)
            try:
                indices_for_batchid = [id for id, value in enumerate(batch) if value == index]
                edges_indices = [torch.where(edge_index[0]==j)[0] for j in indices_for_batchid]
                offset = new_ptr[-1] - batch.tolist().index(index)
                new_edge_index[0].extend(torch.cat([edge_index[0][id]+offset for id in edges_indices]).tolist())
                new_edge_index[1].extend(torch.cat([edge_index[1][id]+offset for id in edges_indices]).tolist())
            except:
                True  

            new_batch.append(i+0*batch[torch.where(batch==index)])
            labels[index] = 1
            new_ptr.append(new_ptr[-1]+len(temp))
            l += 1
        else:
            replace_index = positive_indices[k]
            temp = graph_x[torch.where(batch==replace_index)]
            new_batch.append(i+0*batch[torch.where(batch==replace_index)])
            new_graph_x.append(temp)
            
            try:
                indices_for_batchid = [id for id, value in enumerate(batch) if value == replace_index]
                edges_indices = [torch.where(edge_index[0]==j)[0] for j in indices_for_batchid]
                offset = new_ptr[-1] - batch.tolist().index(replace_index)
                new_edge_index[0].extend(torch.cat([edge_index[0][id]+offset for id in edges_indices]).tolist())
                new_edge_index[1].extend(torch.cat([edge_index[1][id]+offset for id in edges_indices]).tolist())
            except:
                True
            
            new_ptr.append(new_ptr[-1]+len(temp))
            k += 1
    return torch.cat(new_batch), torch.cat(new_graph_x), torch.LongTensor(new_edge_index), torch.LongTensor(new_ptr), labels

def select_random_file(processed_dir):
    files = os.listdir(processed_dir)
    return os.path.join(processed_dir, files[np.random.randint(0,len(files))])
