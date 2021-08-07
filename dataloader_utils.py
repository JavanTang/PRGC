# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain

from utils import Label2IdxSub, Label2IdxObj


class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag


def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples  这里就是获取数据
    """
    examples = []

    # read src data
    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])    # 实体对
                re_list.append(rel2idx[triple[1]])  # 关系列表
                rel2ens[rel2idx[triple[1]]].append(
                    (triple[0], triple[-1]))  # 关系对应的实体对
            # 👇🏻这个只是把一个text的放入了一个样本中
            example = InputExample(
                text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def find_head_idx(source, target):
    """在list中搜索第一个出现的子集,这个代码写的还不错

    :param source: 子集
    :type source: list
    :param target: 需要查找的list
    :type target: list
    :return: 第一个出现子集的位置
    :rtype: int, 没有搜索到就返回-1
    """
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def _get_so_head(en_pair, tokenizer, text_tokens):
    """获取实体对中subject和object在text_tokens这个list中出现的第一个位置

    :param en_pair: 实体对,len为2,一个subject,一个object
    :type en_pair: list
    :param tokenizer: Transformer tokenizer
    :type tokenizer: Transformer tokenizer
    :param text_tokens: 这个是将之前的text转换成了一个list
    :type text_tokens: list
    :return: 分别是 subject在text_tokens中第一次出现的位置,object在text_tokens中第一次出现的位置,以及sub的list和obj的list
    :rtype: set
    """
    sub = tokenizer.tokenize(en_pair[0])
    obj = tokenizer.tokenize(en_pair[1])
    # 找到第一个subject的位置,这个位置对应的是text_tokens 这个list的index
    sub_head = find_head_idx(source=text_tokens, target=sub)
    if sub == obj:  # 这里考虑了两个实体一样
        obj_head = find_head_idx(
            source=text_tokens[sub_head + len(sub):], target=obj)
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        else:
            obj_head = sub_head
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj


def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params):
    """这里就是一个example,一个example的处理了,这里才是最重要的地方

    :param example: 一条训练数据,也就是一句话,里面有实体列表 关系列表 以及关系对应的实体对
    :type example: InputeExample
    :param max_text_len: 文字的最大长度,太长了就会直接截取
    :type max_text_len: int
    :param tokenizer: Transformer的tokenizer
    :type tokenizer: Transformer的tokenizer
    :param rel2idx: 离散关系对应的数值
    :type rel2idx: dict
    :param data_sign: 所属的data类别(test,train,val)
    :type data_sign: str
    :param ex_params: TODO
    :type ex_params: dict
    :return: [description] TODO
    :rtype: list or InputFeatures,InputFeatures(input_tokens,   # 就是这句话的token,这里的token就是英文分词后的英文,这里注意咯,这里不是向量!!!
                                                input_ids,      # 这里都是0-1的值,这里主要是位置的mask
                                                attention_mask, # 这里是attention mask
                                                seq_tag=None,   # 这个里面装了两个list,一个是sub,一个是obj的,里面的取值有三种012 0就是不是实体 1就是实体的开始 2就是实体的延续
                                                corres_tag=None,
                                                relation=None,
                                                triples=None,
                                                rel_tag=None)
    """
    text_tokens = tokenizer.tokenize(
        example.text)  # 这个方法不太知道是干嘛的,只是知道他这里可能是分词吧...  这里就是把str=>list
    # cut off
    if len(text_tokens) > max_text_len:
        text_tokens = text_tokens[:max_text_len]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    # 这里其实就是做了一个位置的mask, 这里需要注意这个mask和后面的masked attention的mask用法可不是同一种搞法,
    # 下面其实就是如果什么位置不够就补成0
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation  这里很关键呀,这个就是那个correspondence and relation的矩阵
        # 之后会用它去做一个剔除的工作!!!
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]    # 这里就是论文里面预测关系的list
        # 这个地方的工作只是为了将那个construct tags of correspondence and relation的值给赋值好
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            # get sub and obj head
            sub_head, obj_head, _, _ = _get_so_head(
                en_pair, tokenizer, text_tokens)
            # construct relation tag
            rel_tag[rel] = 1
            if sub_head != -1 and obj_head != -1:
                # 这里它就把有关系的都放进去了,并赋值为1 TODO 这里我有一个问题,那这里多重关系,这里始终都是1呀...
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        # positive samples
        for rel, en_ll in example.rel2ens.items():
            # init
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            for en in en_ll:
                # get sub and obj head
                sub_head, obj_head, sub, obj = _get_so_head(
                    en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head +
                                 len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head +
                                 len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            # 上面把sub和obj分成了两个部分,不知道这里的sub和obj的识别是不是做了两个分类,而且这个超级像Cas那篇文章的思想
            seq_tag = [tags_sub, tags_obj]

            # sanity check TODO 这个可以学习一下
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            # 这里需要注意,这里的sub_feats是以一个关系为一个特征的,然后这个关系中可能带着很多实体!
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,
                seq_tag=seq_tag,
                relation=rel,
                rel_tag=rel_tag
            ))
        # relation judgement ablation 关系判断消融  这个消融的概念是什么??? TODO
        if not ex_params['ensure_rel']:
            # negative samples
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            for neg_rel in neg_rels:
                # init
                seq_tag = max_text_len * [Label2IdxSub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(
                    attention_mask) == max_text_len, f'length is not equal!!'
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,
                    rel_tag=rel_tag
                ))
    # val and test data
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):  # 这个zip的方法要吸收下来,挺好用的 TODO
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(
                en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))  # 这可能就是一个三元组吧
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats


def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params) -> list:
    """将example的数据转换成为features放入模型,这里用了多线程去convert数据

    :param params: 这个都是自定义好的,都是写到了./utils.py里面
    :type params: dict
    :param examples: 这里将example转成了对象,InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
    :type examples: InputExample
    :param tokenizer: 使用预训练模型直接出来的tokenizer
    :type tokenizer: Transformer Tokenizer
    :param rel2idx: 关系对应的关系,例如 "国家主席"=>1, 可以理解成是将离散的值转换成为数值
    :type rel2idx: dict
    :param data_sign: 是train,test,val哪一种,例如"train"
    :type data_sign: str
    :param ex_params: 默认是NULL
    :type ex_params: [type]
    :return: 具体特征
    :rtype: list
    """
    max_text_len = params.max_seq_length
    # multi-process  这里使用多进程的方式值的学习 TODO
    with Pool(10) as p:
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign, ex_params=ex_params)
        features = p.map(func=convert_func, iterable=examples)

    return list(chain(*features))
