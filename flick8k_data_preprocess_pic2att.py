# This file is used to extract multi-label information for images from the Flick8k-CN data set

import tqdm
import random
import re
from PIL import Image
import matplotlib.pyplot as plt
import jieba
import torch as t
import jieba.posseg as psg
import os

def pic2att_process():
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
    id2ix = {item: ix for ix, item in enumerate(id_l)}
    ix2id = {ix: id for id, ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10
    cap_all_l = []
    for i in range(6000):
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
    words = [word[1] for word in word_nums_list if word[0] >= 2]
    words_r = []
    for item in words:
        cuts = psg.cut(item)
        for w in cuts:
            if w.flag == 'n' or w.flag == 'v':
                words_r.append(w.word)
    new_words = list(set(words_r))
    word_att_n = new_words[:2048]
    from random import shuffle
    shuffle(word_att_n)
    word_att_l = word_att_n.copy()
    word_att = [(ix, item) for ix, item in enumerate(word_att_l)]
    cut_captions_n = []
    for sentences in tqdm.tqdm(cut_captions):
        cut_captions_m = []
        for sentence in sentences:
            cut_captions_m += sentence
        cut_captions_n.append(cut_captions_m)
    pic2att = []
    for pic_cation in tqdm.tqdm(cut_captions_n):
        k = [0] * 2048
        for att_word in word_att:
            for pic_cation_w in pic_cation:
                if att_word[1] == pic_cation_w:
                    k[att_word[0]] = 1
        pic2att.append(k)
    results = {
        'ix2id': ix2id,
        'id2ix': id2ix,
        'cut_captions': cut_captions,
        'word_att': word_att,
        'pic2att': pic2att
    }
    t.save(results, 'flick8k_pic2att.pth')

pic2att_process()
