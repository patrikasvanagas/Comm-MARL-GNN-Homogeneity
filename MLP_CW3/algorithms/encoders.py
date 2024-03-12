import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


def _init_layer(m):
    nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
    nn.init.constant_(m.bias.data, 0)
    return m


def fully_connected_edge_index(num_nodes):
    edge_indexes = [[], []]
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_indexes[0].append(i)
            edge_indexes[1].append(j)
    return torch.tensor(edge_indexes)


class Encoder(nn.Module):
    def __init__(
        self,
        observation_size,
        hidden_size=64,
        use_embedding=False,
        embedding_dim=20,
        n_token=1000,
    ):
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
            observation = self.embedding(observation.long()).reshape(
                -1, self.observation_size
            )

        return self.net(observation)


class GATNetwork_(nn.Module):
    def __init__(self, encoder_dim, gnn_updates=1):
        super(GATNetwork_, self).__init__()
        self.iterations = gnn_updates
        self.encoder_dim = encoder_dim

        self.net = _init_layer(nn.Linear(encoder_dim, encoder_dim))
        self.attention = _init_layer(nn.Linear(2 * encoder_dim, 1))
        self.train()

    def forward(self, embeddings):
        agents_nr = embeddings.shape[0]

        mask = torch.zeros((agents_nr, agents_nr, 1, 1, 1), device=embeddings.device)
        mask = torch.diagonal_scatter(mask, torch.ones(1, 1, 1, agents_nr), 0).bool()

        for _ in range(self.iterations):
            all_embeddings_rolled = torch.stack(
                [
                    torch.roll(embeddings, shifts=-i, dims=0)
                    for i in range(len(embeddings))
                ]
            )
            all_embeddings = embeddings.unsqueeze(0).expand(agents_nr, -1, -1, -1, -1)

            concat_values = torch.cat((all_embeddings, all_embeddings_rolled), dim=-1)
            attentions_all = F.leaky_relu(self.attention(concat_values))
            attentions_all.masked_fill_(mask, -1e10)

            attentions = F.softmax(attentions_all, dim=1)
            next_embeddings = (all_embeddings_rolled * attentions).sum(dim=1)
            embeddings = next_embeddings + embeddings

        return embeddings


class GNNFactory(nn.Module):
    def __init__(self, gnn_class, in_dim, out_dim, layer_nr, skip_connections=True):
        super(GNNFactory, self).__init__()
        self.layers = nn.ModuleList(
            [gnn_class(in_dim, out_dim) for _ in range(layer_nr)]
        )
        self.skip_connections = skip_connections

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index) + (x if self.skip_connections else 0)
        return x


class GATNetwork(nn.Module):
    def __init__(self, encoder_dim, layer_nr=1):
        super(GATNetwork, self).__init__()
        self.encoder_dim = encoder_dim
        self.gnn = GNNFactory(
            GATConv,
            in_dim=encoder_dim,
            out_dim=encoder_dim,
            layer_nr=layer_nr,
            skip_connections=True,
        )

    def forward(self, embeddings):
        """
        embeddings: The embedded observations of each agents of shape [num_agents, num_samples, num_processes, embedding_dim]
        """
        agents_nr, samples_nr, process_nr, _ = embeddings.shape
        edge_index = fully_connected_edge_index(agents_nr)
        futures = [
            torch.jit.fork(self.gnn, x, edge_index=edge_index)
            for x in embeddings.transpose(2, 0).reshape(-1, agents_nr, self.encoder_dim)
        ]
        outputs = torch.stack([torch.jit.wait(fut) for fut in futures], dim=0)
        return outputs.reshape(samples_nr, process_nr, agents_nr, -1).permute(
            2, 0, 1, 3
        )


class GATv2Network(nn.Module):
    def __init__(self, encoder_dim, layer_nr=1):
        super(GATv2Network, self).__init__()
        self.encoder_dim = encoder_dim
        self.gnn = GNNFactory(
            GATv2Conv,
            in_dim=encoder_dim,
            out_dim=encoder_dim,
            layer_nr=layer_nr,
            skip_connections=True,
        )

    def forward(self, embeddings):
        """
        embeddings: The embedded observations of each agents of shape [num_agents, num_samples, num_processes, embedding_dim]
        """
        agents_nr, samples_nr, process_nr, _ = embeddings.shape
        edge_index = fully_connected_edge_index(agents_nr)
        futures = [
            torch.jit.fork(self.gnn, x, edge_index=edge_index)
            for x in embeddings.transpose(2, 0).reshape(-1, agents_nr, self.encoder_dim)
        ]
        outputs = torch.stack([torch.jit.wait(fut) for fut in futures], dim=0)
        return outputs.reshape(samples_nr, process_nr, agents_nr, -1).permute(
            2, 0, 1, 3
        )


class AttentionMechanism(nn.Module):
    def __init__(self, encoding_dim, d_K):
        """
        Initializes the attention mechanism module from
        https://arxiv.org/pdf/1906.01202.pdf (p. 4)

        Parameters:
        - encoding_dim: The dimension of the agent state encoding.
        - d_K: The dimensionality of the key and query vectors.
        """
        super(AttentionMechanism, self).__init__()
        self.encoding_dim = encoding_dim
        self.d_K = d_K

        self.W_K = nn.Linear(encoding_dim, d_K)
        self.W_Q = nn.Linear(encoding_dim, d_K)
        self.W_V = nn.Linear(encoding_dim, d_K)
        self.W_out = nn.Linear(d_K, encoding_dim)

    def forward(self, encodings):
        """
        Forward pass of the attention mechanism.

        Parameters:
        - encodings: Tensor of shape (batch_size, num_agents, encoding_dim), the encodings h_i for each agent.

        Returns:
        - V_f: The aggregated and transformed value for each agent
          of shape (batch_size, num_agents, encoding_dim).
        - attention_weights: The attention weights for each agent
          of shape (batch_size, num_agents, num_agents).
        """
        # encodings shape expected: (batch_size, num_agents, encoding_dim)
        # keys.shape = (batch_size, num_agents, d_K)
        Keys = self.W_K(encodings)
        Queries = self.W_Q(encodings)
        Values = self.W_V(encodings)

        # attention_scores and weights shape: (batch_size, num_agents, num_agents)
        attention_scores = torch.matmul(Queries, Keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_K, dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        # aggregated_values shape: (batch_size, num_agents, encoding_dim)
        aggregated_values = torch.matmul(attention_weights, Values)
        # V_f shape: (batch_size, num_agents, encoding_dim)
        V_f = self.W_out(aggregated_values)

        return V_f, attention_weights
