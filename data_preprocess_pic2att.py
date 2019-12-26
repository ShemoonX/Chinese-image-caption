# coding:utf-8
# This file is used to extract multi-label information for images from the Ai challenge 2017 training set

import json
import jieba

import tqdm
import torch as t
import os
from PIL import Image

class Config:
    annotation_file = 'ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
    save_path = 'pic2att.pth'

def pic2att_process():
    opt = Config()
    with open(opt.annotation_file) as f:
        data = json.load(f)
    id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
    ix2id = {ix: id for id, ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10
    captions = [item['caption'] for item in data]
    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(captions)]
    word_nums = {}
    for sentences in cut_captions:
        for sentence in sentences:
            for word in sentence:
                word_nums[word] = word_nums.get(word, 0) + 1
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)
    print(len(word_nums_list))
    words = [word[1] for word in word_nums_list if word[0] >= 52]
    print(len(words))
    words_cx = []
    import jieba.posseg as psg
    for item in words:
        cuts = psg.cut(item)
        for w in cuts:
            if w.flag == 'n' or w.flag == 'v' or w.flag == 'a' or w.flag == 'm' or w.flag == 'i':
                words_cx.append(w.word)
    # print('词性筛选后词表长度:{}'.format(len(words_cx)))
    words_i = list(set(words_cx))
    # print('重复词筛选后词表长度:{}'.format(len(words_i)))
    word_att_n = words_i[:2048]
    # print('最终属性词表长度:{}'.format(len(word_att_n)))
    from random import shuffle
    shuffle(word_att_n)
    word_att_l = word_att_n.copy()
    word_att = [(ix,item) for ix, item in enumerate(word_att_l)]
    cut_captions_n = []
    for sentences in tqdm.tqdm(cut_captions):
        cut_captions_m = []
        for sentence in sentences:
                cut_captions_m += sentence
        cut_captions_n.append(cut_captions_m)
    pic2att = []
    for pic_cation in tqdm.tqdm(cut_captions_n):
        k = [0]*2048
        for att_word in word_att:
            for pic_cation_w in pic_cation:
                if att_word[1] == pic_cation_w:
                    k[att_word[0]] = 1
        pic2att.append(k)
    results = {
            'ix2id': ix2id,
            'id2ix': id2ix,
            'word_att':word_att,
            'pic2att':pic2att
        }
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)

# run
pic2att_process()

# test
def text(index):
    opt = Config()
    results = t.load(opt.save_path)
    pic2att = results['pic2att']
    the_pic_att = pic2att[index]
    the_pic_att_s = [(ix, item) for ix, item in enumerate(the_pic_att)]
    word_att = results['word_att']
    for item in the_pic_att_s:
        if item[1] == 1:
            print(word_att[item[0]])
    ix2id = results['ix2id']
    img_data_path = 'ai_challenger_caption_train_20170902/caption_train_images_20170902/'
    img_path = os.path.join(img_data_path, ix2id[index])
    img = Image.open(img_path).convert('RGB')
    img.show()

text(68)

