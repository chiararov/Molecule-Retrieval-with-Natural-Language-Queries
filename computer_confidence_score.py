import json
from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

confindences_path = "./confidences/crossmodal_contrastive_roberta-base03_11_18_34_epoch_19.pkl"
with open(confindences_path, 'r') as json_file:
    conf_dict = json.load(json_file)

print("Loading model...")

model_path = 'crossmodal_contrastive_roberta-base02_18_21_49_epoch_1.pt'
checkpoint = torch.load('./models/' + model_path)

print("Loading data...")
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
tokenizer = checkpoint['tokenizer']
batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=checkpoint['model']
model.eval()

print("Creating test datasets...")
test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

print("Creating test dataloaders...")
test_graph_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

num_text_batches = len(test_text_loader)
num_graph_batches = len(test_graph_loader)
probas = np.zeros((num_text_batches, num_graph_batches))

########### Prepare rule creation ###########
print("Preparing rule creation...")
print("Creating train dataset...")
train_dataset = GraphTextDataset(root='data/', gt=gt, split='train', tokenizer=tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

print("Loading all learnt graph token ids...")
all_graph_tokens = np.concatenate([batch.tokens[0] for batch,_ in train_loader])
print("Loading all learnt text token ids...")
all_text_tokens = np.concatenate([batch.input_ids[0] for batch,_ in train_loader])

graph_token_ids = defaultdict(lambda : -1)
text_token_ids = defaultdict(lambda : -1) 

graph_token_ids_rev = {}
text_token_ids_rev = {}
for i, k in enumerate(all_graph_tokens):
    graph_token_ids[k] = i
    graph_token_ids_rev[i] = k
for i, k in enumerate(all_text_tokens):
    text_token_ids[k] = i
    text_token_ids_rev[i] = k

def generate_rules(text_tokens, graph_tokens):
    text_subs = [frozenset([text_token_ids[j] for j in i]) for i in combinations(text_tokens, 1)]
    mol_subs = [frozenset([graph_token_ids[j] for j in i]) for i in combinations(graph_tokens, 1)]
    rules = []
    for t in text_subs:
        for m in mol_subs:
            rules.append((t, m))
    return rules

def ar_score(text_tokens, mol_tokens, top_num=10):
    rules = generate_rules(text_tokens, mol_tokens)
    tmp = np.array([conf_dict[list(r[0])[0], list(r[1])[0]] for r in rules])
    mx = np.min((top_num, len(tmp)))
    top_confs = -np.partition(-tmp, mx-1)[:mx]
    return np.mean(top_confs)
########### End rule creation ###########


# Loop over text batches
for i, text_batch in tqdm(enumerate(test_text_loader)):
    text_tokens = text_batch['input_ids'][0]
    # Extract corresponding values from data_dict and accumulate in probas
    for j, graph_batch in enumerate(test_graph_loader):
        probas[i, j] = ar_score(text_tokens, graph_batch.tokens[0])

print("Saving submission...")
solution = pd.DataFrame(probas)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submissions/submission_' + model_path + '.csv', index=False)

print("Testing done!")        

