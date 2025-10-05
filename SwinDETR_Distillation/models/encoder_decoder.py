""" DETR Transformer Encoder-Decoder with Distillation Query """



import torch
import torch.nn as nn
from torch import Tensor

from copy import deepcopy
from typing import Optional


def with_pos_embed(tensor, pos: Optional[Tensor] = None):
    return tensor + pos if pos is not None else tensor


def get_clones(module: nn.Module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


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
        """
        Params:
            - src: image features, [batch_size, hidden_dim, image_height // 32, image_width // 32]
            - mask: attention mask, [batch_size, image_height // 32, image_width // 32]
            - src_key_padding_mask: key padding mask, [batch_size, image_height // 32 * image_width // 32]
            - pos: positional encoding (same shape as src)
        """
        # add positional encoding to src (k, q)
        q = k = with_pos_embed(src, pos)

        # 1) Multi-Head Self-Attention
        src2 = self.attention(q, k, value=src, attn_mask=mask, key_padding_mask=src_key_padding_mask)[0]
        # 2) Add & Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 3) Feed Forward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 4) Add & Norm
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
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        Params:
            - tgt: object queries, [num_query, batch_size, hidden_dim]
            - memory: features from encoder, [batch_size, hidden_dim, image_height // 32 * image_width // 32]
            - tgt_mask: attention mask for tgt, [num_query, num_query]
            - tgt_key_padding_mask: key padding mask for tgt, [batch_size, num_query]
            - memory_mask: attention mask for memory(feature from encoder), [num_query, image_height // 32 * image_width // 32]
            - memory_key_padding_mask: key padding mask for memory(feature from encoder), [batch_size, image_height // 32 * image_width // 32]
            - pos: positional encoding (same shape as src)
            - query_pos: positional encoding for query (same shape as tgt)
        """
        # add positional encoding(query_pos) to tgt(object queries) (k, q)
        q = k = with_pos_embed(tgt, query_pos)

        # 1) Multi-Head Self-Attention
        tgt2 = self.attention1(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # 2) Add & Norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # 3) Multi-Head Cross-Attention
        tgt2 = self.attention2(query=with_pos_embed(tgt, query_pos), key=with_pos_embed(memory, pos), value=memory,
                               attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        # 4) Add & Norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # 5) Feed Forward Network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 6) Add & Norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


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
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
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


# DETR Transformer Encoder-Decoder with Distillation Query
class TransformerEncoderDecoder(nn.Module):
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

    def forward(self, src, mask, query, distill_query, pos):
        """
        Params:
            - src: image features, [batch_size, hidden_dim, image_height // 32, image_width // 32]
            - mask: attention mask, [batch_size, image_height // 32, image_width // 32]
            - query: object queries, [num_query, hidden_dim]
            - distill_query: distillation queries, [num_distill_query, hidden_dim]
            - pos: positional encoding (the same shape as src)
            
        Returns:
            - output: [num_decoder_layer, batch_size, num_query + num_distill_query, hidden_dim]
        """
        N = src.shape[0]

        # flatten src from NxCxHxW to HWxNxC
        src = src.flatten(2).permute(2, 0, 1)
        # flatten mask from NxHxW to NxHW
        mask = mask.flatten(1)
        # flatten pos from NxCxHxW to HWxNxC
        pos = pos.flatten(2).permute(2, 0, 1)

        # repeat query & distill query for each batch, [num_query/num_distill_query, N(batch_size), hidden_dim]
        query = query.unsqueeze(1).repeat(1, N, 1)
        distill_query = distill_query.unsqueeze(1).repeat(1, N, 1)
        # create combined query (object queries + distillation queries)
        combined_query = torch.cat([query, distill_query], dim=0)

        # create tgt for decoder
        combined_tgt = torch.zeros_like(combined_query)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        output = self.decoder(combined_tgt, memory, memory_key_padding_mask=mask, pos=pos, query_pos=combined_query).transpose(1, 2)

        return output