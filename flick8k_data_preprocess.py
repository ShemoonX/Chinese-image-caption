# This file processes the original data description into a glossary index
# This file is used to index the data processing to Numbers to facilitate the neural network model to learn the mapping between the data.

import tqdm
import random
import re
from PIL import Image
import matplotlib.pyplot as plt
import jieba
import torch as t
import os

def process():
    id_l = []
    cap_l_0 = []
    ifn_0 = r"flick8k_caption_0.txt"
    infile_0 = open(ifn_0, 'r', encoding='utf-8')
    for eachline in tqdm.tqdm(infile_0.readlines()):
        url = eachline
        id = url[0:url.rfind('#zhc#')]
        id_l.append(id)
        cap_r = url[url.rfind('#zhc#'):-2]
        cap = cap_r.replace('#zhc#0 ', '')
        cap_l_0.append(cap)
    cap_l_1 = []
    ifn_1 = r"flick8k_caption_1.txt"
    infile_1 = open(ifn_1, 'r', encoding='utf-8')
    for eachline in tqdm.tqdm(infile_1.readlines()):
        url = eachline
        cap_r = url[url.rfind('#zhc#'):-2]
        cap = cap_r.replace('#zhc#1 ', '')
        cap_l_1.append(cap)
    cap_l_2 = []
    ifn_2 = r"flick8k_caption_2.txt"
    infile_2 = open(ifn_2, 'r', encoding='utf-8')
    for eachline in tqdm.tqdm(infile_2.readlines()):
        url = eachline
        cap_r = url[url.rfind('#zhc#'):-2]
        cap = cap_r.replace('#zhc#2 ', '')
        cap_l_2.append(cap)
    cap_l_3 = []
    ifn_3 = r"flick8k_caption_3.txt"
    infile_3 = open(ifn_3, 'r', encoding='utf-8')
    for eachline in tqdm.tqdm(infile_3.readlines()):
        url = eachline
        cap_r = url[url.rfind('#zhc#'):-2]
        cap = cap_r.replace('#zhc#3 ', '')
        cap_l_3.append(cap)
    cap_l_4 = []
    ifn_4 = r"flick8k_caption_4.txt"
    infile_4 = open(ifn_4, 'r', encoding='utf-8')
    for eachline in tqdm.tqdm(infile_4.readlines()):
        url = eachline
        cap_r = url[url.rfind('#zhc#'):-2]
        cap = cap_r.replace('#zhc#4 ', '')
        cap_l_4.append(cap)
    id_l[0] = '667626_18933d713e.jpg'
    id = id_l[0:6000]
    id2ix = {item: ix for ix, item in enumerate(id)}
    ix2id = {ix: id for id, ix in (id2ix.items())}
    id_v = id_l[6000:7000]
    id2ix_v = {item: ix for ix, item in enumerate(id_v)}
    ix2id_v = {ix: id for id, ix in (id2ix_v.items())}
    assert id2ix[ix2id[10]] == 10
    assert id2ix_v[ix2id_v[10]] == 10
    cap_all_l = []
    for i in range(7000):
        cap_all_one = []
        cap_all_one.append(cap_l_0[i])
        cap_all_one.append(cap_l_1[i])
        cap_all_one.append(cap_l_2[i])
        cap_all_one.append(cap_l_3[i])
        cap_all_one.append(cap_l_4[i])
        cap_all_l.append(cap_all_one)
    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(cap_all_l)]
    word_nums = {}
    for sentences in cut_captions:
        for sentence in sentences:
            for word in sentence:
                word_nums[word] = word_nums.get(word, 0) + 1
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)
    words = [word[1] for word in word_nums_list if word[0] >= 1]
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    words = [unknown, padding, end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    print(len(word2ix))
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123
    cut_captions_l = cut_captions[0:6000]
    ix_captions = [[[word2ix.get(word, word2ix.get(unknown)) for word in sentence] for sentence in item] for item in
                   cut_captions_l]
    cut_captions_v = cut_captions[6000:7000]
    ix_captions_v = [[[word2ix.get(word, word2ix.get(unknown)) for word in sentence] for sentence in item] for item in
                   cut_captions_v]
    readme = u"""id:图片名 caption: 分词之后的描述，通过ix2word可以获得原始中文词 """
    results = {
        'caption': ix_captions,
        'caption_v': ix_captions_v,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id': ix2id,
        'id2ix': id2ix,
        'ix2id_v': ix2id_v,
        'id2ix_v': id2ix_v,
        'padding': '</PAD>',
        'end': '</EOS>',
        'readme': readme
    }
    t.save(results, 'flick8k_t_v_caption.pth')

process()
