from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel, RobertaModel, BertModel

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels,dropout):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch):

        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        if edge_index.numel() > 0:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.dropout(x)
            x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.dropout(x)
        x = self.mol_hidden2(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name,dropout):
        super(TextEncoder, self).__init__()
        if model_name == 'roberta-base':
            self.bert = RobertaModel.from_pretrained(model_name)
        elif model_name == 'distilbert-base-uncased':
            self.bert = AutoModel.from_pretrained(model_name)
        elif model_name == 'scibert':
            self.bert = BertModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.dropout = nn.Dropout(dropout)
        self.model_name = model_name
          
    def forward(self, input_ids, attention_mask):
        
        if self.model_name == 'scibert':
            encoded_text = self.bert(input_ids, attention_mask=attention_mask)['pooler_output']
        else:
            encoded_text = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        return self.dropout(encoded_text)
    
class BaseLineModel(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels,dropout=0):
        super(BaseLineModel, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels,dropout=dropout)
        self.text_encoder = TextEncoder(model_name,dropout)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder


class CrossModalAttentionModel(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, mol_trunc_length = 256, nhead=8, nlayers=3, temp=0.07, dropout=0.5):
        super(CrossModalAttentionModel, self).__init__()

        if model_name == 'roberta-base':
            self.text_transformer_model = RobertaModel.from_pretrained(model_name)
            ninp = 768
        elif model_name == 'distilbert-base-uncased':
            self.text_transformer_model = AutoModel.from_pretrained(model_name)
        
        self.text_hidden1 = nn.Linear(ninp, nhid)
        self.text_hidden2 = nn.Linear(nhid, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.num_node_features = num_node_features
        self.graph_hidden_channels = graph_hidden_channels
        self.mol_trunc_length = mol_trunc_length

        self.drop = nn.Dropout(p=dropout)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.text_transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        
        self.temp = nn.Parameter(torch.Tensor([temp]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #For GCN:
        self.conv1 = GCNConv(self.num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

        self.other_params = list(self.parameters()) #get all but bert params

        
    def forward(self, graph_batch, input_ids, attention_mask):
        text_encoder_output = self.text_transformer_model(input_ids, attention_mask = attention_mask)

        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        if edge_index.numel() > 0:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.drop(x)
            try:
                x = self.conv2(x, edge_index)
            except:
                print("x.size:",x.size)
                print("edge_index.shape:",edge_index.shape)
                print(":",x)
                print("edge_index:",edge_index)
            x = x.relu()
            x = self.drop(x)
            x = self.conv3(x, edge_index)
        mol_x = x

        #turn pytorch geometric output into the correct format for transformer
        #requires recovering the nodes from each graph into a separate dimension
        node_features = torch.zeros((graph_batch.num_graphs, self.mol_trunc_length, self.graph_hidden_channels)).to('cuda')
        for i, p in enumerate(graph_batch.ptr): #graph_batch.ptr is a row tensor containing 0, the last index corresponding to graph1, the last index corresponding to graph2, etc...
            if p == 0: 
                old_p = p
                continue
            node_features[i - 1, :p-old_p, :] = mol_x[old_p:torch.min(p, old_p + self.mol_trunc_length), :]
            old_p = p
        node_features = torch.transpose(node_features, 0, 1)

        text_output = self.text_transformer_decoder(text_encoder_output['last_hidden_state'].transpose(0,1), node_features,tgt_key_padding_mask = attention_mask == 0) 

        #Readout layer
        x = global_mean_pool(mol_x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x)
        x = x.relu()
        x = self.drop(x)
        x = self.mol_hidden2(x)

        text_x = torch.tanh(self.text_hidden1(text_output[0,:,:])) #[CLS] pooler
        text_x = self.text_hidden2(text_x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return x, text_x
    