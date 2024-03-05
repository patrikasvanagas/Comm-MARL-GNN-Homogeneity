import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_layer(m):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.constant_(m.bias.data, 0)
        return m

class FCLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layernorm=None, activation=None, device=None):
        super(FCLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = _init_layer(nn.Linear(input_dim, hidden_dim, device=device))
        self.norm = nn.LayerNorm(hidden_dim, device=device) if layernorm else None
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, layernorm=None, device=None):
        super(GRULayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim, device=device)
        self.norm = nn.LayerNorm(hidden_dim, device=device) if layernorm else None
     
    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, x, hiddens):
        # flatten hiddens if needed
        if hiddens is not None and len(hiddens.shape) > 2:
            hiddens_shape = hiddens.shape
            hiddens = hiddens.reshape(-1, self.hidden_dim)
            x = x.reshape(-1, x.shape[-1])
        else:
            hiddens_shape = None
        hiddens = self.gru(x, hiddens)
        # if flattened before, unflatten again
        if hiddens_shape is not None:
            hiddens = hiddens.view(hiddens_shape)
        if self.norm is not None:
            hiddens = self.norm(hiddens)
        return hiddens


class RNNBase(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens, layernorm, device):
        super(RNNBase, self).__init__()
        assert len(hiddens) > 0; "Provide valid hidden layer configuration"

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.norm = nn.LayerNorm(input_dim, device=device) if layernorm else None
        # RNN block
        self.pre_rnn = FCLayer(input_dim, hiddens[0], layernorm, nn.ReLU(), device)
        self.rnn = GRULayer(hiddens[0], hiddens[0], layernorm, device=device)
        # Extra hidden layers
        fc_network_layers = [FCLayer(hidden_dim, hidden_dim, layernorm, nn.ReLU(), device) for hidden_dim in hiddens[1:]]
        # Out layer
        fc_network_layers.append(FCLayer(hiddens[-1], output_dim, layernorm, None, device))
        
        self.post_rnn = nn.Sequential(*fc_network_layers)
        self.rnn_hidden_dim = self.rnn.hidden_dim
     
    def init_hidden(self, batch_size=1):
        return self.rnn.init_hidden(batch_size)
    
    def forward(self, inputs, hiddens=None):
        if self.norm is not None:
            inputs = self.norm(inputs)
        x = self.pre_rnn(inputs)
        hiddens = self.rnn(x, hiddens)
        out = self.post_rnn(hiddens)
        return out, hiddens

class MultiAgentNetworks(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            hiddens,
            layernorm,
            device=DEVICE,
    ):
        super(MultiAgentNetworks, self).__init__()
        self.rnn_hidden_dims = [hiddens[0] for _ in range(len(input_dims))]
        self.networks = nn.ModuleList([
            RNNBase(
                input_dim, output_dim, hiddens, layernorm, device
            )
            for input_dim, output_dim in zip(input_dims, output_dims)
        ])
    
    def forward(self, inputs, hiddens):
        if hiddens is None:
            hiddens = [None] * len(inputs)
        futures = [
            torch.jit.fork(model, x, h) for model, x, h in zip(self.networks, inputs, hiddens)
        ]
        outputs = [torch.jit.wait(fut) for fut in futures]
        outs = torch.stack([out for out, _ in outputs], dim=0)
        hiddens = torch.stack([h for _, h in outputs], dim=0)
        return outs, hiddens


class MultiAgentSharedNetworks(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            hiddens,
            layernorm,
            device=DEVICE,
    ):
        super(MultiAgentSharedNetworks, self).__init__()
        input_dim = input_dims[0] 
        output_dim = output_dims[0]
        assert all([i == input_dim for i in input_dims]), "Input dimensions must be equal for shared networks!"
        assert all([o == output_dim for o in output_dims]), "Output dimensions must be equal for shared networks!"

        self.rnn_hidden_dims = [hiddens[0] for _ in range(len(input_dims))]
        self.shared_network = RNNBase(input_dim, output_dim, hiddens, layernorm, device)
    
    def forward(self, inputs, hiddens):
        if hiddens is None:
            hiddens = [None] * len(inputs)
        futures = [
            torch.jit.fork(self.shared_network, x, h) for x, h in zip(inputs, hiddens)
        ]
        outputs = [torch.jit.wait(fut) for fut in futures]
        outs = torch.stack([out for out, _ in outputs], dim=0)
        hiddens = torch.stack([h for _, h in outputs], dim=0)
        return outs, hiddens