import torch
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv


def fully_connected_edge_index(num_nodes):
    edge_indexes = [[], []]
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_indexes[0].append(i)
            edge_indexes[1].append(j)
    return torch.tensor(edge_indexes)


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
