import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


def doy_to_month(doy):
    # Approximate mapping of day of the year to month
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = torch.zeros_like(doy)
    cumulative_days = 0
    for i, days in enumerate(month_days):
        cumulative_days += days
        month += (doy <= cumulative_days).int() * (month == 0).int() * (i + 1)
    return month - 1 # adjust to 0-based indexing

def normalize_doy(x_mark):
    while x_mark.max() > 365:
        x_mark = x_mark - 365 * (x_mark > 365).int()
    return x_mark

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1462):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

### reconsider the TokenEmbedding class: kernel_size = 3 for embeddings of the satellite channels?
### does that mean only three neighboring channels can be considered?
class TokenEmbeddingCNN(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=10):
        super(TokenEmbeddingCNN, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        ### note that we have changed the default kernel_size from 3 to 10
        ### because we have 10 satellite channels and a kernel of size 3 
        ### would only consider the three neighboring channels
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=kernel_size, padding='same', padding_mode='circular', bias=False)
        ### since we changed the kernel_size to always be equal to the number of input features, 
        ### we don't need any padding anymore
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                            kernel_size=1, padding=0, bias=False)
        ### (comment added on 2024-12-16) explanation of the above: 
        ### the kernel_size parameter essentially defines the number of timesteps
        ### of the moving window of the Conv1d layer, 
        ### and as we only want to tokenize satellite bands and not timesteps at all, 
        ### we should set it to 1 (padding=0 is no problem then)
        ### kernel_size is used as a hyperparameter here because my understanding was not entirely correct
        ### for the final code version, we can probably remove that and keep kernel_size=1 all the time
        ### that is also why Benny said "a Conv1d layer with kernel_size of 1 is essentially 
        ### a token embedding using a MLP, but more efficient"
        ### here's the explanation that really helped me: 
        ### https://pub.aimind.so/lets-understand-the-1d-convolution-operation-in-pytorch-541426f01448
        ### note that in_channels is the relevant parameter here to operate across all input channels
        ### that's why it is always set to the number of channels in the input data
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

### this is the alternative class which uses a fully connected layer for token embeddings
### in case we are not happy with the Conv1d kernel_size=3 "neighborhood" problem
class TokenEmbeddingMLP(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbeddingMLP, self).__init__()
        self.tokenLinear = nn.Linear(in_features=c_in, out_features=d_model, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenLinear(x)
        return x

### using self-attention insteaf of MLP or CNN for token embeddings
class TokenEmbeddingAttention(nn.Module):
    def __init__(self, c_in, d_model, nhead):
        super(TokenEmbeddingAttention, self).__init__()
        self.embedding = nn.Linear(c_in, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Linear embedding
        x = self.embedding(x)
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Add & Norm
        x = self.norm(x + attn_output)
        
        return x

### this class provides TRAINABLE (not fixed) sine and cosine embeddings
### this is the class that we should use for our DOY embeddings
### it accounts for the cyclical nature of our data (forest time series)
### and should be able to capture the yearly seasonality
### it is not implemented yet, but we should take it into account later
class TrainableSinCosEmbedding(nn.Module):
    # num_embeddings = size of vocabulary
    def __init__(self, num_embeddings, embedding_dim):
        super(TrainableSinCosEmbedding, self).__init__()

        # Initialize the embeddings using sine and cosine functions
        position = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        embeddings = torch.zeros(num_embeddings, embedding_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)

        # Create an nn.Embedding layer and set its weights to the initialized embeddings
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed.weight = nn.Parameter(embeddings)
        self.embed.weight.requires_grad = True  # Make the embeddings trainable

    def forward(self, x):
        return self.embed(x)

### FIXED embeddings are not updated during training
### these are the embeddings of the original transformer
### and likely not useful for our use case
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

### make sure that embed_type is NOT "fixed"
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='learnedsincos', freq='d'):
        super(TemporalEmbedding, self).__init__()

        ### adapt this to my DOY approach
        # minute_size = 4
        # hour_size = 24
        # weekday_size = 7
        # day_size = 32
        # month_size = 13
        # doy_size = 1827 # 4*365 + 1*366 (leap year) + 1 (padding)
        doy_size = 366 # each of the four years get he normal doy values

        ### three options: 
        ### TrainableSinCosEmbedding: trainable + sin/cos embeddings account for cyclical nature of data
        ### FixedEmbedding: sin/cos but fixed (not updated during training)
        ### nn.Embedding: not fixed, but not sin/cos embeddings (not accounting for cyclical nature)
        Embed = FixedEmbedding if embed_type == 'fixed' else TrainableSinCosEmbedding if embed_type == 'learnedsincos' else nn.Embedding
        # if freq == 't':
        #     self.minute_embed = Embed(minute_size, d_model)
        # self.hour_embed = Embed(hour_size, d_model)
        # self.weekday_embed = Embed(weekday_size, d_model)
        # self.day_embed = Embed(day_size, d_model)
        # self.month_embed = Embed(month_size, d_model)
        ### note that this might not be the most sophisticated way to embed the day of year
        ### since this uses the normal embedding layer of pytorch
        ### later we should add learnable embeddings with sine and cosine functions
        ### at the moment we pass embed_type = "learned" to the model
        ### that's the simple nn.Embedding
        ### now we want to make it sin/cos embeddings but NOT fixed -> embed_type = "learnedsincos"
        ### (this combination does not seem to exist in the TimesNet script as of now)
        ### we call it embed_type = "learnedsincos"
        self.doy_embed = Embed(doy_size, d_model)

        ### initialize month embedding
        self.month_embed = Embed(12, d_model)

    def forward(self, x):
        x = x.long()
        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
        #     self, 'minute_embed') else 0.
        # hour_x = self.hour_embed(x[:, :, 3])
        # weekday_x = self.weekday_embed(x[:, :, 2])
        # day_x = self.day_embed(x[:, :, 1])
        # month_x = self.month_embed(x[:, :, 0])
        doy_x = self.doy_embed(x)

        ### prepare monthly temporal embeddings
        ### normalize day of the year values
        normalized_doy = normalize_doy(x)
        ## translate to month of the year
        month = doy_to_month(normalized_doy)
        ### get monthly embedding
        month_embedding = self.month_embed(month)

        ### this would be the place where we can switch to concatenation of the embeddings
        ### instead of summing them up
        # return hour_x + weekday_x + day_x + month_x + minute_x
        return doy_x # + month_embedding

### never used (there is no use case for us)
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='learnedsincos', freq='h', dropout=0.5, kernel_size=10):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbeddingCNN(c_in=c_in, d_model=d_model, kernel_size=kernel_size) # c_in = input channels
        # self.value_embedding = TokenEmbeddingMLP(c_in=c_in, d_model=d_model) # c_in = input channels
        # self.value_embedding = TokenEmbeddingAttention(c_in=c_in, d_model=d_model, nhead=8)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        ### since we do NOT pass "timeF", we never use TimeFeatureEmbedding (which is not useful for our use case)
        
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    ### x_mark is definitely the timestamp information!
    ### that means that we can use this to embed the day of year
    def forward(self, x, x_mark):
        if x_mark is None:
            ### not used in our case, cause we always have timestamps
            ### but always used in the "normal" anomaly detection tasks in this code
            x = self.value_embedding(x) + self.position_embedding(x)  # output: [batch_size, seq_len, d_model]
        else:
            ### this might also be a suitable location for concatenation instead of summing up
            ### note that we added the month embedding here already (in the TemporalEmbedding class)
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) # + self.position_embedding(x)
            ### we dropped the position embedding
            ### the location within the timeseries is ALWAYS given as x_mark
            ### and contains all of the information needed (day of the year)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbeddingCNN(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
