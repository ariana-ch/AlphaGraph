import math
import numpy as np
import torch
import torch.nn as nn
from math import sqrt
from torch.nn import functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# See the code: https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, dropout: float = 0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        # if self.activation is not None:
        #     dims = x.shape
        #     x = self.flatten(x)
        #     x = x.reshape(*dims)
        x = self.dropout(x)
        return x


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = nn.Flatten(-2)(x)  # Flatten the last two dimensions
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TimeSeriesNormalisation(nn.Module):
    def __init__(self):
        super().__init__()
        self.means = None
        self.stdevs = None
        self.transpose = Transpose(1, 2)

    def forward(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        self.means = means
        self.stdevs = stdev

        #    We permute once: [B, seq_len, enc_in] -> [B, enc_in, seq_len]
        x_enc = self.transpose(x_enc)  # => [batch_size, enc_in=1, seq_len=24 (example)]
        return x_enc

    def denormalise(self, dec_out):
        dec_out = self.transpose(dec_out) # same as dec_out.permute(0, 2, 1)  # => [B, pred_len, n_vars]
        dec_out = dec_out * self.stdevs[:, 0, :].unsqueeze(1) + self.means[:, 0, :].unsqueeze(1)
        return dec_out  # => final shape: [B, pred_len, enc_in]


class PatchTSTModel(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(
        self, lookback_len=30, pred_len=1, d_model=128, n_heads=4, d_ff=256, dropout=0.1,
        e_layers=2, factor=5, activation='gelu', features = 45,
        enc_in=20, patch_len=10, stride=5, output_logits=True,
    ):
        super().__init__()

        self.lookback_len = lookback_len
        self.pred_len = pred_len
        padding = stride
        self.entry_layer = nn.Linear(features, enc_in)  # Linear layer to map original features to enc_in
        # normalisation per time series
        self.time_series_normalisation = TimeSeriesNormalisation()

        # patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        )

        # patch_num = how many patches from seq_len,
        # e.g. patch_num = (seq_len - patch_len)//stride + 1 + 1
        patch_num = int((lookback_len - patch_len) // stride + 1 + 1)
        self.head_nf = d_model * patch_num

        # prediction head
        self.head1 = FlattenHead(enc_in, self.head_nf, pred_len, dropout=0.0 if output_logits else dropout)
        if output_logits:
            self.head2 = Head(enc_in, pred_len, dropout=dropout)
        else:
            self.head2 = nn.Flatten(start_dim=-2)
        # self.fc = nn.Sequential(Transpose(2, 1),
        #                         nn.Linear(enc_in, 1),
        #                         Transpose(2, 1))  # Final linear layer to map to enc_in features

    def forward(self, x):
        """
        x: [batch_size, seq_len, enc_in]
        """
        # 1) Normalization
        # Non-stationary Transformer paper link: https://arxiv.org/abs/2205.14415
        bs = x.shape[0] # batch size
        x = self.entry_layer(x)  # => [batch_size, enc_in, seq_len]
        x = self.time_series_normalisation(x) # x_enc: [batch_size, enc_in, seq_len]

        # 2) Patch + Embedding
        x, n_vars = self.patch_embedding(x) # enc_out
        # shape of enc_out: [B * n_vars, patch_num, d_model]

        # 3) Encoder
        x, attns = self.encoder(x)
        # enc_out: [B*n_vars, patch_num, d_model]

        # 4) Reshape back to [B, n_vars, patch_num, d_model]
        x = x.reshape(bs, n_vars, x.shape[1], x.shape[2])
        # => [B, 1, patch_num, d_model]

        # 5) Permute for head => [B, n_vars, d_model, patch_num]
        x = x.permute(0, 1, 3, 2)

        # 6) Decoder Head => Flatten last two dims & apply linear
        x = self.head1(x)  # dec_out => [B, n_vars, pred_len]

        # 7) De-Normalization
        x = self.time_series_normalisation.denormalise(x)
        # => final shape: [B, pred_len, enc_in]

        # 8) Final linear layer to map to enc_in features
        x = self.head2(x)  # => [B, pred_len]
        return x

