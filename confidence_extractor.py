print("Loading libraries...")
import torch
import numpy as np
from dataloader import GraphTextDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
from os import path as osp
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "true"

print("Loading model...")
confindences_path = "./confidences/crossmodal_contrastive_roberta-base03_11_18_34_epoch_19.pkl"
model_path = "./models/crossmodal_contrastive_roberta-base03_11_18_34_epoch_19.pt"
checkpoint = torch.load(model_path)

print("Loading data...")
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
tokenizer = checkpoint['tokenizer']
batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

model=checkpoint['model']
model.eval()
print("done")

def patch_attention(m):
    forward_orig = m.forward
    def wrap(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return forward_orig(*args, **kwargs)
    m.forward = wrap

class SaveWeights:
    def __init__(self):
        self.mha_weights = {}
    def __call__(self, module, module_in, module_out):
        for i,cid in enumerate(cids):
            self.mha_weights[cid.item()] = module_out[1][i] #takes only the first sample from the batch, so it gives only 256x256
    def clear(self):
        self.outputs = []

save_weights = SaveWeights()
layer = model.text_transformer_decoder.layers[-1].multihead_attn
patch_attention(layer)
handle = layer.register_forward_hook(save_weights)

association_rules = {}
confidences = {}
print("Going through data to get weights...")
#Go through data to get the weights

for k, data in tqdm(enumerate(train_loader)):
    batch, cids = data
    cid = cids[0]
    input_ids = batch.input_ids
    _ = batch.pop('input_ids')
    attention_mask = batch.attention_mask
    _ = batch.pop('attention_mask')
    graph_batch = batch
    if graph_batch.edge_index.numel() == 0: continue
    #Collecting attention weights through the hook
    x_graph, x_text = model(graph_batch.to(device),input_ids.to(device),attention_mask.to(device))
    text_length = attention_mask.sum(dim=1).item()
    graph_length = graph_batch.x.shape[0]
    weights = save_weights.mha_weights[cid.item()]
    weights = weights.detach().cpu().numpy()
    save_weights.mha_weights[cid.item()] = weights[:text_length, :graph_length]
    max_i, max_j = save_weights.mha_weights[cid.item()].shape
    #Creating association rules' supports
    for i,t in enumerate(input_ids[0]):
        t = str(t.item())
        if t in ['0','1','2']: continue #Special tokens
        _ = association_rules.setdefault(t,{})
        for j,m in enumerate(graph_batch.tokens[0]):
            _ = association_rules[t].setdefault(m,0)
            association_rules[t][m] += save_weights.mha_weights[cid.item()][i][j].item()
            if j>= max_j-1: break
        if i>= max_i-1: break

#Computing confidence
for t in association_rules.keys():
    weights_sum = sum(association_rules[t].values())
    _ = confidences.setdefault(t,{})
    for m in association_rules[t].keys():
        if weights_sum == 0: confidences[t][m] = 0
        else: confidences[t][m] = association_rules[t][m]/weights_sum

print("Saving confidences...")
with open(confindences_path, 'a') as fp:
    json.dump(confidences, fp)
