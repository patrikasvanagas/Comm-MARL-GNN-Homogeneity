import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def _init_layer(m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

class Encoder(nn.Module):
    def __init__(self, observation_size, hidden_size=64, use_embedding=False, embedding_dim=20, n_token=1000):
        super(Encoder, self).__init__()
        self.observation_size = observation_size
        self.use_embedding = use_embedding

        if use_embedding:
            self.observation_size = self.observation_size * embedding_dim
            self.embedding = nn.Embedding(n_token, embedding_dim, dtype=torch.float32)
        
        self.net = nn.Sequential(
            _init_layer(nn.Linear(self.observation_size, hidden_size)),
        )
        self.train()

    def forward(self, observation):
        if self.use_embedding:
            observation += 1
            observation = self.embedding(observation.long()).reshape(-1, self.observation_size)
        
        return self.net(observation)

class GATNetwork(nn.Module):
    def __init__(self, encoder_dim, gnn_updates=10):
        super(GATNetwork, self).__init__()
        self.iterations = gnn_updates
        self.encoder_dim = encoder_dim

        self.net = _init_layer(nn.Linear(encoder_dim, encoder_dim))
        self.attention =  _init_layer(nn.Linear(2 * encoder_dim, 1)) 
        self.train()


    def forward(self, embeddings):
        agents_nr = embeddings.shape[0]

        mask = torch.zeros((agents_nr, agents_nr, 1, 1, 1), device=embeddings.device)
        mask = torch.diagonal_scatter(mask, torch.ones(1, 1, 1, agents_nr), 0).bool()

        for _ in range(self.iterations):
            all_embeddings_rolled = torch.stack([torch.roll(embeddings, shifts=-i, dims=0) for i in range(len(embeddings))])
            all_embeddings = embeddings.unsqueeze(0).expand(agents_nr, -1, -1, -1, -1)

            concat_values = torch.cat((all_embeddings, all_embeddings_rolled), dim=-1)
            attentions_all = F.leaky_relu(self.attention(concat_values)) 
            attentions_all.masked_fill_(mask, -1e10)
            
            attentions = F.softmax(attentions_all, dim=1)
            next_embeddings = (all_embeddings_rolled * attentions).sum(dim=1)            
            embeddings = next_embeddings

        return embeddings