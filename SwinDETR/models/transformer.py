""" Transformer Module for DETR """


import torch
from torch import nn, Tensor

from copy import deepcopy
from typing import Optional


def with_pos_embed(tensor, pos: Optional[Tensor] = None):
    return tensor + pos if pos is not None else tensor


def get_clones(module: nn.Module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            intermediate.append(output)

        return torch.stack(intermediate)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, dim_feedforward, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, num_head, dropout=dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = with_pos_embed(src, pos)

        src2 = self.attention(q, k, value=src, attn_mask=mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_head, dim_feedforward, dropout):
        super().__init__()

        self.attention1 = nn.MultiheadAttention(hidden_dim, num_head, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(hidden_dim, num_head, dropout=dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        q = k = with_pos_embed(tgt, query_pos)

        tgt2 = self.attention1(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.attention2(query=with_pos_embed(tgt, query_pos), key=with_pos_embed(memory, pos),
                               value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class Transformer(nn.Module):
    def __init__(self, hidden_dim, num_head, num_encoder_layer, num_decoder_layer, dim_feedforward, dropout):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layer)

        decoder_layer = TransformerDecoderLayer(hidden_dim, num_head, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layer)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query, pos):
        """
        Params:
            - src: tensor of shape [batch_size, hidden_dim, image_height // 32, image_width // 32]
            - mask: tensor of shape [batch_size, image_height // 32, image_width // 32]
            - query: object queries, tensor of shape [num_query, hidden_dim]
            - pos: positional encoding, the same shape as src
            
        Returns:
            tensor of shape [batch_size, num_query * num_decoder_layer, hidden_dim]
        """
        N = src.shape[0]

        # flatten NxCxHxW to HWxNxC
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)
        query = query.unsqueeze(1).repeat(1, N, 1) # [num_query, N, hidden_dim]
        tgt = torch.zeros_like(query)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        output = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos, query_pos=query).transpose(1, 2)

        return output