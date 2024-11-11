print("Loading imports...")
from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

print("Loading model...")
# model_path = "./models/model01_29_23_15_epoch_20.pt"
# model_path = './models/model01_31_10_52_epoch_5_div3.pt'
# model_path='./models/baseline_contrastive_roberta-base03_16_50_11_epoch_15.pt' 
# model_path='./models/baseline_contrastive_roberta-base03_16_13_42_epoch_10.pt'
# model_path='./models/baseline_contrastive_roberta-base03_17_25_09_epoch_20.pt'  
model_path='./ensemble_models/baseline_scibert_sub_dropout_0.3_lr_5e-0617_08_47_epoch_29.pt' 
checkpoint = torch.load(model_path)

print("Loading data...")
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
tokenizer = checkpoint['tokenizer']
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=checkpoint['model']
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

print("Creating datasets...")
test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

print("Creating dataloaders...")
test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

print("Embedding graphs...")
graph_embeddings = []
for batch in tqdm(test_loader):
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

print("Embedding texts...")
test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in tqdm(test_text_loader):
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())

print("Loading sklearn.metrics...")
from sklearn.metrics.pairwise import cosine_similarity

print("Calculating similarity...")
similarity = cosine_similarity(text_embeddings, graph_embeddings)

print("Saving submission...")
solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv('submissions/submission_scibert_' + model_path[-22:-3] + '.csv', index=False)

print("Testing done!")