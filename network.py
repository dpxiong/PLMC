import torch
import torch.nn.functional as F
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size/num_attention_heads)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_scores(mixed_query_layer)
        key_layer = self.transpose_scores(mixed_key_layer)
        value_layer = self.transpose_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores
        
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_head_size * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TransEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(TransEncoder, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size)
        self.output = SelfOutput(hidden_size, num_attention_heads, hidden_dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self_attention(input_tensor)
        self_output = self.LayerNorm(self_output + input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class feat2Embed(nn.Module):
    def __init__(self):
        super(feat2Embed, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9139, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class PLMC(nn.Module):
    def __init__(self, protein_hf_dim, hid_embed_dim, num_heads, dropout, max_length=1000):
        super(PLMC, self).__init__()
        self.dropout = dropout
        self.pemb = nn.Linear(1280, hid_embed_dim)
        self.embed_dim = hid_embed_dim
        self.protein_hf_dim = protein_hf_dim
        self.num_attention = num_heads

        self.protein_plm_layer = nn.Linear(self.embed_dim * max_length, self.embed_dim)
        self.protein_plm_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.protein_plm_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        
        self.feat2_embed = feat2Embed()

        self.protein_hf_layer = nn.Linear(self.protein_hf_dim, self.embed_dim)
        self.protein_hf_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.protein_hf_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        
        self.total_layer = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.total_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.total_layer_1 = nn.Linear(self.embed_dim, 2)
        self.softmax = nn.Softmax(dim=1)
        self.trans_encoder = TransEncoder(self.embed_dim, self.num_attention, self.dropout, self.dropout)

    def forward(self, protein_plm, protein_hf, device):
        p_feature1 = self.trans_encoder(self.pemb(protein_plm.to(torch.float32)))
        p_feature1 = p_feature1.view(p_feature1.shape[0], -1)
        p_feature1 = F.relu(self.protein_plm_bn(self.protein_plm_layer(p_feature1)), inplace=True)
        p_feature1 = F.dropout(p_feature1, training=self.training, p=self.dropout)
        p_feature1 = self.protein_plm_layer_1(p_feature1)
        
        p_feature2 = self.feat2_embed(protein_hf.to(torch.float32))
        p_feature2 = F.relu(self.protein_hf_bn(self.protein_hf_layer(p_feature2)), inplace=True)
        p_feature2 = F.dropout(p_feature2, training=self.training, p=self.dropout)
        p_feature2 = self.protein_hf_layer_1(p_feature2)

        p_feature = torch.cat([p_feature1, p_feature2], dim=1)
        p_feature = F.relu(self.total_bn(self.total_layer(p_feature)), inplace=True)
        p_feature = F.dropout(p_feature, training=self.training, p=self.dropout)
        p_feature = self.total_layer_1(p_feature)
        
        probs = self.softmax(p_feature)
        
        return probs