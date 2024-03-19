import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math


def _init_layer(m):
    nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
    nn.init.constant_(m.bias.data, 0)
    return m


class Encoder(nn.Module):
    def __init__(
        self,
        observation_size,
        hidden_size=64,
        use_embedding=False,
        embedding_dim=20,
        n_token=1000,
        device="cpu",
    ):
        super(Encoder, self).__init__()
        self.observation_size = observation_size
        self.use_embedding = use_embedding
        self.device = device
        if use_embedding:
            self.observation_size = self.observation_size * embedding_dim
            self.embedding = nn.Embedding(n_token, embedding_dim, dtype=torch.float32)

        self.net = nn.Sequential(
            _init_layer(nn.Linear(self.observation_size, hidden_size, device=device)),
        )

    def forward(self, observation):
        if self.use_embedding:
            observation += 1
            observation = self.embedding(observation.long()).reshape(
                -1, self.observation_size
            )

        return self.net(observation)


class GATNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = False,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        share_weights: bool = False,
    ):
        super(GATNetwork, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        # Layer used for the source transformation
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Layer used for the target transformation; If share weight we use the same transformation
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(2 * self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h has shape (agents_nr, samples_nr, process_nr, encoder_dim)
        agents_nr, samples_nr, process_nr, encoder_dim = h.shape
        # h change shape to (samples_nr, process_nr, agents_nr, encoder_dim)
        h = h.permute(1, 2, 0, 3)
        # Left and right transformations (W_l*h and W_r*h)
        g_l = self.linear_l(h).view(
            samples_nr, process_nr, agents_nr, self.n_heads, self.n_hidden
        )
        g_r = self.linear_r(h).view(
            samples_nr, process_nr, agents_nr, self.n_heads, self.n_hidden
        )
        # Chance g_l dim to (samples_nr, process_nr, agents_nr*agents_nr, encoder_dim)
        g_l_repeat = g_l.repeat(1, 1, agents_nr, 1, 1)
        # Chance g_r dim to (samples_nr, process_nr, agents_nr*agents_nr, encoder_dim) BUT interleaved
        g_r_repeat_interleave = g_r.repeat_interleave(agents_nr, dim=2)
        g_concat = torch.cat((g_l_repeat, g_r_repeat_interleave), dim=-1)
        g_concat = g_concat.view(
            samples_nr,
            process_nr,
            agents_nr,
            agents_nr,
            self.n_heads,
            2 * self.n_hidden,
        )
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        # Remove self loops
        mask = torch.zeros((1, 1, agents_nr, agents_nr, 1), device=h.device)
        mask = torch.diagonal_scatter(
            mask, torch.ones(1, 1, 1, agents_nr), dim1=2, dim2=3
        ).bool()
        e.masked_fill_(mask, float("-inf"))

        a = self.softmax(e)
        # a = self.dropout(a)
        attn_res = torch.einsum("abijh,abjhf->abihf", a, g_r)

        if self.is_concat:
            attn_res = attn_res.reshape(
                samples_nr, process_nr, agents_nr, self.n_heads * self.n_hidden
            )
        else:
            attn_res = attn_res.mean(dim=3)

        attn_res = attn_res.permute(2, 0, 1, 3)
        return attn_res, a.squeeze(-1)


class GATv2Network(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = False,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        share_weights: bool = False,
    ):
        super(GATv2Network, self).__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        # Layer used for the source transformation
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Layer used for the target transformation; If share weight we use the same transformation
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(2 * self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=3)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        # h has shape (agents_nr, samples_nr, process_nr, encoder_dim)
        agents_nr, samples_nr, process_nr, encoder_dim = h.shape
        # h change shape to (samples_nr, process_nr, agents_nr, encoder_dim)
        h_ = h.permute(1, 2, 0, 3)
        # Left and right transformations (W_l*h and W_r*h)
        g_l = self.linear_l(h_).view(
            samples_nr, process_nr, agents_nr, self.n_heads, self.n_hidden
        )
        g_r = self.linear_r(h_).view(
            samples_nr, process_nr, agents_nr, self.n_heads, self.n_hidden
        )
        # Chance g_l dim to (samples_nr, process_nr, agents_nr*agents_nr, encoder_dim)
        g_l_repeat = g_l.repeat(1, 1, agents_nr, 1, 1)
        # Chance g_r dim to (samples_nr, process_nr, agents_nr*agents_nr, encoder_dim) BUT interleaved
        g_r_repeat_interleave = g_r.repeat_interleave(agents_nr, dim=2)
        g_concat = torch.cat((g_l_repeat, g_r_repeat_interleave), dim=-1)
        g_concat = g_concat.view(
            samples_nr,
            process_nr,
            agents_nr,
            agents_nr,
            self.n_heads,
            2 * self.n_hidden,
        )
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        # Remove self loops
        mask = torch.zeros((1, 1, agents_nr, agents_nr, 1), device=h_.device)
        mask = torch.diagonal_scatter(
            mask, torch.ones(1, 1, 1, agents_nr), dim1=2, dim2=3
        ).bool()
        e.masked_fill_(mask, float("-inf"))

        a = self.softmax(e)
        # a = self.dropout(a)
        attn_res = torch.einsum("abijh,abjhf->abihf", a, g_r)

        if self.is_concat:
            attn_res = attn_res.reshape(
                samples_nr, process_nr, agents_nr, self.n_heads * self.n_hidden
            )
        else:
            attn_res = attn_res.mean(dim=3)

        attn_res = attn_res.permute(2, 0, 1, 3)
        return attn_res, a.squeeze(-1)


class AttentionMechanism_v1(nn.Module):
    def __init__(self, encoding_dim, d_K, no_attention_to_self=True):
        """
        Initializes the attention mechanism module from
        https://arxiv.org/pdf/1906.01202.pdf (p. 4)

        Parameters:
        - encoding_dim: The dimension of the agent state encoding.
        - d_K: The dimensionality of the key and query vectors.
        - no_attention_to_self: If True, the attention mechanism will not
        allow agents to attend to themselves.
        """
        super(AttentionMechanism_v1, self).__init__()
        self.encoding_dim = encoding_dim
        self.d_K = d_K
        self.no_attention_to_self = no_attention_to_self

        self.W_K = nn.Linear(encoding_dim, d_K, bias=True)
        self.W_Q = nn.Linear(encoding_dim, d_K, bias=True)
        self.W_V = nn.Linear(encoding_dim, d_K, bias=True)
        self.W_out = nn.Linear(d_K, encoding_dim, bias=True)

        nn.init.xavier_uniform_(self.W_K.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_Q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_V.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_out.weight, gain=1 / math.sqrt(2))

        nn.init.constant_(self.W_K.bias, 0.0)
        nn.init.constant_(self.W_Q.bias, 0.0)
        nn.init.constant_(self.W_V.bias, 0.0)
        nn.init.constant_(self.W_out.bias, 0.0)

    def forward(self, encodings):
        """
        Forward pass of the attention mechanism.

        Parameters:
        - encodings: Tensor of shape (n_agents, batch_size, number of envs,
        encoding dim), the encodings h_i for each agent.

        Returns:
        - V_f: The aggregated and transformed value for each agent
          of shape (n_agents, batch_size, number of envs, encoding dim)
        - attention_weights: The attention weights for each agent
          of shape (n_agents, batch_size, number of envs, number of agents)
        """
        # Permute to shape (batch_size, num_envs, num_agents, encoding_dim)
        # to allow for matrix multiplication
        encodings = encodings.permute(1, 2, 0, 3)

        # keys.shape = (batch_size, num_envs, num_agents, d_K)
        Keys = self.W_K(encodings)
        Queries = self.W_Q(encodings)
        Values = self.W_V(encodings)

        # attention_scores and weights shape: (batch_size, num_envs, num_agents, num_agents)
        attention_scores = torch.matmul(Queries, Keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_K, dtype=torch.float32)
        )

        if self.no_attention_to_self:
            batch_size, num_envs, num_agents, _ = attention_scores.shape
            mask = torch.eye(num_agents, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(batch_size, num_envs, 1, 1)
            attention_scores.masked_fill_(mask, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)

        # aggregated_values shape: (batch_size, num_envs, num_agents, encoding_dim)
        aggregated_values = torch.matmul(attention_weights, Values)
        # V_f shape: (batch_size, num_envs, num_agents, encoding_dim)
        V_f = self.W_out(aggregated_values)

        # Permute back to original shape
        V_f = V_f.permute(2, 0, 1, 3)
        return V_f, attention_weights


class AttentionMechanism_v2(nn.Module):
    def __init__(
        self,
        encoding_dim=128,
        out_features: int = 128,
        n_heads: int = 8,
        is_concat: bool = False,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        share_weights: bool = False,
    ):
        super(AttentionMechanism_v2, self).__init__()
        self.encoding_dim = encoding_dim

        self.W_K = nn.Linear(encoding_dim, encoding_dim, bias=True)
        self.W_Q = nn.Linear(encoding_dim, encoding_dim, bias=True)
        self.W_V = nn.Linear(encoding_dim, encoding_dim, bias=True)
        self.W_out = nn.Linear(encoding_dim, encoding_dim, bias=True)

        self.input_layer_norm = nn.LayerNorm(encoding_dim)
        self.output_layer_norm = nn.LayerNorm(encoding_dim)

        nn.init.xavier_uniform_(self.W_K.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_Q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_V.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_out.weight, gain=1 / math.sqrt(2))

        nn.init.constant_(self.W_K.bias, 0.0)
        nn.init.constant_(self.W_Q.bias, 0.0)
        nn.init.constant_(self.W_V.bias, 0.0)
        nn.init.constant_(self.W_out.bias, 0.0)

    def forward(self, encodings):
        """
        Forward pass of the attention mechanism.

        Parameters:
        - encodings: Tensor of shape (n_agents, batch_size, number of envs,
        encoding dim), the encodings h_i for each agent.

        Returns:
        - attention: The aggregated and transformed value for each agent
          of shape (n_agents, batch_size, number of envs, encoding dim)
        # - attention_weights: The attention weights for each agent
        #   of shape (n_agents, batch_size, number of envs, number of agents)
        """

        num_agents, batch_size, num_envs, encoding_dim = encodings.shape

        # PRIDEJAU DABAR
        encodings = self.input_layer_norm(encodings)

        # (num_agents, batch_size, num_envs, encoding_dim)
        # -> (batch_size, num_envs, num_agents, encoding_dim)
        encodings = encodings.permute(1, 2, 0, 3).contiguous()

        Keys = self.W_K(encodings).view(batch_size * num_envs, num_agents, encoding_dim)
        Queries = self.W_Q(encodings).view(
            batch_size * num_envs, num_agents, encoding_dim
        )
        Values = self.W_V(encodings).view(
            batch_size * num_envs, num_agents, encoding_dim
        )

        # (batch_size * num_envs, num_agents, encoding_dim)
        # x (batch_size * num_envs, encoding_dim, num_agents)
        # -> (batch_size * num_envs, num_agents, num_agents)
        q_dot_k = torch.bmm(Queries, Keys.transpose(1, 2)) / math.sqrt(encoding_dim)

        q_dot_k = F.softmax(q_dot_k, dim=2)

        # (batch_size * num_envs, num_agents, num_agents)
        # x (batch_size * num_envs, num_agents, encoding_dim)
        # -> (batch_size * num_envs, num_agents, encoding_dim)
        attention = torch.bmm(q_dot_k, Values)

        # PRIDEJAU DBR
        attention = self.output_layer_norm(attention)

        # (batch_size * num_envs, num_agents, encoding_dim)
        # x (encoding_dim, encoding_dim)
        # -> (batch_size * num_envs, num_agents, encoding_dim)
        attention = self.W_out(attention)

        # (batch_size * num_envs, num_agents, encoding_dim)
        # -> (num_envs, batch_size, num_agents, encoding_dim)
        attention = attention.view(num_envs, batch_size, num_agents, encoding_dim)

        # (num_envs, batch_size, num_agents, encoding_dim)
        # -> (num_agents, batch_size, num_envs, encoding_dim)
        attention = attention.permute(2, 1, 0, 3).contiguous()

        # (batch_size * num_envs, num_agents, num_agents)
        # -> (batch_size, num_envs, num_agents, num_agents)
        attention_weights = q_dot_k.view(batch_size, num_envs, num_agents, num_agents)
        

        return attention, attention_weights
