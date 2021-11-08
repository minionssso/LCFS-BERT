# -*- coding: utf-8 -*-
# file: lcfs_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention

# from layers.point_wise_feed_forward import PositionwiseFeedForward
# from transformers.models.bert.modeling_bert import BertPooler
# from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig


class MyPooler(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.dense = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PointwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_hid, d_inner_hid=None,d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class LCFS_GLOVE(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LCFS_GLOVE, self).__init__()
        # sa_config = BertConfig(hidden_size=self.hidden,output_attentions=True)

        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        # self.mhsa_global = Attention(embed_dim=opt.embed_dim, n_head=8, score_function='mlp')
        self.mhsa_global = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.pct_global = PointwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.mhsa_local = Attention(embed_dim=opt.embed_dim, n_head=8, score_function='mlp')
        self.pct_local = PointwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.dropout = nn.Dropout(opt.dropout)

        # self.mean_pooling_double = nn.Linear(hidden * 2, hidden)
        self.mean_pooling_double = PointwiseFeedForward(opt.hidden_dim * 2, opt.hidden_dim)
        # self.final_sa = Attention(embed_dim=opt.embed_dim, n_head=8, score_function='scaled_dot_product')
        self.final_sa = NoQueryAttention(embed_dim=opt.embed_dim, n_head=8)
        self.final_pooler = MyPooler(opt)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices,distances_input=None):
        texts = text_local_indices.cpu().numpy() # batch_size x seq_len
        asps = aspect_indices.cpu().numpy() # batch_size x aspect_len
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.hidden_dim),
                                          dtype=np.float32) # batch_size x seq_len x hidden size
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))): # For each sample
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) # Calculate aspect length
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][0])[0][0]
                except:
                    continue
                # Mask begin -> Relative position of an aspect vs the mask
                if asp_begin >= mask_len:
                    mask_begin = asp_begin - mask_len
                else:
                    mask_begin = 0
                for i in range(mask_begin): # Masking to the left
                    masked_text_raw_indices[text_i][i] = np.zeros((self.opt.hidden_dim), dtype=np.float)
                for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len): # Masking to the right
                    masked_text_raw_indices[text_i][j] = np.zeros((self.opt.hidden_dim), dtype=np.float)
            else:
                distances_i = distances_input[text_i]
                for i,dist in enumerate(distances_i):
                    if dist > mask_len:
                        masked_text_raw_indices[text_i][i] = np.zeros((self.opt.hidden_dim), dtype=np.float)

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices,distances_input=None):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
                                          dtype=np.float32) # batch x seq x dim
        mask_len = self.opt.SRD
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) - 2
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][2])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2 # central position
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
                for i in range(1, np.count_nonzero(texts[text_i])-1):
                    srd = abs(i - asp_avg_index) + asp_len / 2
                    if srd > self.opt.SRD:
                        distances[i] = 1 - (srd - self.opt.SRD)/np.count_nonzero(texts[text_i])
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
            else:
                distances_i = distances_input[text_i] # distances of batch i-th
                for i,dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(texts[text_i])
                    else:
                        distances_i[i] = 1

                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances_i[i]

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]  # Raw text without adding aspect term
        aspect_indices = inputs[3] # Raw text of aspect
        distances = inputs[4]
        #distances = None
        
        # Embedding Layer: Glove,也可以加squeeze_embedding
        text_global_indices = self.embed(text_bert_indices)
        text_local_indices = self.embed(text_local_indices)
        aspect_indices = self.embed(aspect_indices)

        # MHSA + PCT
        text_mhsa_global, _ = self.mhsa_global(text_global_indices, text_global_indices)  # 看下这个atten的输出
        text_pct_global = self.pct_global(text_mhsa_global)
        text_mhsa_local, _ = self.mhsa_local(text_local_indices, aspect_indices)
        text_pct_local = self.pct_local(text_mhsa_local)

        if self.opt.local_context_focus == 'cdm':
            masked_local_text_vec = self.feature_dynamic_mask(text_local_indices, aspect_indices, distances)
            text_local_out = torch.mul(text_pct_local, masked_local_text_vec)

        elif self.opt.local_context_focus == 'cdw':
            weighted_text_local_features = self.feature_dynamic_weighted(text_local_indices, aspect_indices, distances)
            text_local_out = torch.mul(text_pct_local, weighted_text_local_features)

        out_cat = torch.cat((text_local_out, text_pct_global), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.final_sa(mean_pool)
        pooled_out = self.final_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        # if output_attentions:
        #     return (dense_out,spc_att,local_att)  # spc_att=最后一个头的attention参数
        return dense_out, local_att
