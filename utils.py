#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List
import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re 
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
from multiprocessing import cpu_count
from contractions import fix


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)

    max_length=max([len(x) for x in sents])
    diffs=[max_length-len(x) for x in sents]



    for i in range(len(diffs)):
        sents[i].extend([pad_token]*diffs[i])

    sents_padded=sents[:]
    ### END YOUR CODE

    return sents_padded

def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """



    if source=="tgt":
        nlp=spacy.load("en_core_web_sm")
    else:
        nlp=spacy.load("es_core_news_sm")

    with open(file_path,"r") as file:
        input_data=file.readlines()
        file.close()


    if source=="tgt":
        for x in tqdm(range(len(input_data))):
            input_data[x]=fix(input_data[x])


    for x in range(len(input_data)):
        input_data[x]=input_data[x].lower()


    output=nlp.pipe(input_data,n_threads=cpu_count(),batch_size=5000,disable=["tagger", "parser","ner"])

    data=[]
    counter=1
    for doc in output:
        print(counter)
        sent=[]
        
        for tok in doc:
            tok=str(tok)

            if tok=="'s" and source=='tgt':
                sent[-1]=sent[-1]+tok
                continue

            if re.search("^[^A-Za-z0-9]+$",tok) is not None:
                continue
            else:
                sent.append(tok)
        
        if source=="tgt":
            sent = ['<s>'] + sent + ['</s>']

        if len(sent)==0:
            sent=list(doc)
        data.append(sent)
        counter+=1

    # data = []
    # for line in open(file_path):
    #     sent = line.strip().split(' ')
    #     # only append <s> and </s> to the target sentence
    #     if source == 'tgt':
    #         sent = ['<s>'] + sent + ['</s>']
    #     data.append(sent)

    return data



def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents