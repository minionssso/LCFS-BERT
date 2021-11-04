# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx
import spacy
# from transformers import BertTokenizer,XLNetTokenizer
from pytorch_transformers import BertTokenizer,XLNetTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


# def _load_word_vec(path, word2idx=None):
#     fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     word_vec = {}
#     for line in fin:
#         tokens = line.rstrip().split()
#         if word2idx is None or tokens[0] in word2idx.keys():
#             word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
#     return word_vec

def _load_word_vec(path, word2idx=None, embed_dim=300):
    # path=glove文件
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()  # 删除字符串末尾空格
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove/glove.6B.300d.txt'
            # if embed_dim != 300 else './glove/glove.840B.300d.txt'
        # word_vec = _load_word_vec(fname, word2idx=word2idx)
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


# TODO 这里用int64，0.5的值算完变为 0了
def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


# 非BERT模型用此Tokenizer
class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        # self.tokenizer = tokenizer
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):  # 构建 word2idx, idx2word（没有CLS、SEP
        if self.lower:
            text = text.lower()
        words = text.split()
        # unknownidx = len(self.word2idx)+1
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
                # self.word2idx[word] = unknownidx


    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):  # 把句子str的word用id代替，且句子
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1  # 4584
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def tokenize(self, text, dep_dist, reverse=False, padding='post', truncating='post'):
        sequence, distances = [], []
        for word, dist in zip(text, dep_dist):  # 句子中单词遍历
            sequence.append(word)
            distances.append(dist)
        # sequence = self.tokenizer.convert_tokens_to_ids(sequence)
        for ix, seq in enumerate(sequence):
            if seq in self.word2idx.keys():
                sequence[ix] = self.word2idx[seq]
                # seq_id.append(self.word2idx[seq])
        # sequence = self.text_to_sequence(sequence)  # sequence是list,要先变为str

        if len(sequence) == 0:
            sequence = [0]
            dep_dist = [0]
        if reverse:
            sequence = sequence[::-1]
            dep_dist = dep_dist[::-1]
        sequence = pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        dep_dist = pad_and_truncate(dep_dist, self.max_seq_len, padding=padding, truncating=truncating,value=self.max_seq_len)

        return sequence, dep_dist


# class Tokenizer4Pretrain:
#     def __init__(self, tokenizer, max_seq_len):
#         self.tokenizer = tokenizer
#         self.cls_token = tokenizer.cls_token
#         self.sep_token = tokenizer.sep_token
#         self.max_seq_len = max_seq_len

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    # Group distance to aspect of an original word to its corresponding subword token
    def tokenize(self, text, dep_dist, reverse=False, padding='post', truncating='post'):
        sequence, distances = [], []
        for word, dist in zip(text, dep_dist):
            tokens = self.tokenizer.tokenize(word)  # 看看这里是啥,有些词在Bert里面是可以继续切分的，比如arafat切分成'ara'和'##fat',所以最后sequence比text长
            for jx, token in enumerate(tokens):  # type(tokens)
                sequence.append(token)
                distances.append(dist)
        sequence = self.tokenizer.convert_tokens_to_ids(sequence)  # sequence是str吗

        if len(sequence) == 0:
            sequence = [0]
            dep_dist = [0]
        if reverse:
            sequence = sequence[::-1]
            dep_dist = dep_dist[::-1]
        sequence = pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        dep_dist = pad_and_truncate(dep_dist, self.max_seq_len, padding=padding, truncating=truncating,value=self.max_seq_len)

        return sequence, dep_dist


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            auxiliary_aspect = 'What is the polarity of {}'.format(aspect)  # TODO
            polarity = lines[i + 2].strip()

            raw_text = text_left + " " + aspect + " " + text_right
            text_raw_indices = tokenizer.text_to_sequence(raw_text)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            auxiliary_aspect_indices = tokenizer.text_to_sequence(auxiliary_aspect)
            auxiliary_aspect_len = np.sum(auxiliary_aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1
            sent = text_left + " " + aspect + " " + text_right
            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + sent + ' [SEP] ' + aspect + ' [SEP]')

            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            # if 'Roberta' in type(tokenizer.tokenizer).__name__:
            #     bert_segments_ids = np.zeros(np.sum(text_raw_indices != 0) + 2 + aspect_len + 1)
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)  # padding置零

            text_raw_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP]')

            # Find distance in dependency parsing tree
            raw_tokens, dist = calculate_dep_dist(sent,aspect)  # 返回asp与上下文词的距离
            raw_tokens.insert(0,'[CLS]')
            dist.insert(0,0)
            raw_tokens.append('[SEP]')
            dist.append(0)

            _, distance_to_aspect = tokenizer.tokenize(raw_tokens, dist)  # distance_to_aspect就是依赖树距离的输入，shape(80,)
            aspect_bert_indices = tokenizer.text_to_sequence('[CLS] ' + aspect + ' [SEP]')

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
                'dep_distance_to_aspect':distance_to_aspect,
                'raw_text':raw_text,
                'aspect':aspect
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

nlp = spacy.load("en_core_web_sm")  # 得到句法依赖树的工具
def calculate_dep_dist(sentence,aspect):
    terms = [a.lower() for a in aspect.split()]  # aspect多词组成的继续划分
    doc = nlp(sentence)
    # Load spacy's dependency tree into a networkx graph
    edges = []
    cnt = 0
    term_ids = [0] * len(terms)
    for token in doc:  # 遍历句子中的token
        # Record the position of aspect terms
        if cnt < len(terms) and token.lower_ == terms[cnt]: 
            term_ids[cnt] = token.i  # 把aspect在句子中的位置存于term_ids中
            cnt += 1

        for child in token.children:  # token和边child的关系
            edges.append(('{}_{}'.format(token.lower_,token.i),
                          '{}_{}'.format(child.lower_,child.i)))

    graph = nx.Graph(edges)

    dist = [0.0]*len(doc)
    text = [0]*len(doc)
    for i,word in enumerate(doc):
        source = '{}_{}'.format(word.lower_,word.i)
        sum = 0
        for term_id,term in zip(term_ids,terms):
            target = '{}_{}'.format(term, term_id)
            try:
                sum += nx.shortest_path_length(graph,source=source,target=target)  # 求最短路径长度
            except:
                sum += len(doc) # No connection between source and target
        dist[i] = sum/len(terms) # 多个asp token分别和句子token之间的距离，再除以asp token的数量
        text[i] = word.text
    return text,dist