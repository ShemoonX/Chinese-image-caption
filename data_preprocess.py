# coding:utf-8
# This file processes the original data description into a glossary index
# This file is used to index the data processing to Numbers to facilitate the neural network model to learn the mapping between the data.

import json
import jieba
import tqdm
import torch as t

class Config:
    annotation_file = 'ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
    unknown = '</UNKNOWN>'
    end = '</EOS>'
    padding = '</PAD>'
    max_words = 5000
    min_appear = 2
    save_path = 'caption.pth'

def process(**kwargs):
    opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    with open(opt.annotation_file) as f:
        data = json.load(f)
    id2ix = {item['image_id']: ix for ix, item in enumerate(data)}
    ix2id = {ix: id for id, ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10
    captions = [item['caption'] for item in data]
    # Using jieba0.38
    cut_captions = [[list(jieba.cut(ii, cut_all=False)) for ii in item] for item in tqdm.tqdm(captions)]
    word_nums = {}
    for sentences in cut_captions:
        for sentence in sentences:
            for word in sentence:
                word_nums[word] = word_nums.get(word, 0) + 1
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()], reverse=True)
    words = [word[1] for word in word_nums_list[:opt.max_words] if word[0] >= opt.min_appear]
    words = [opt.unknown, opt.padding, opt.end] + words
    word2ix = {word: ix for ix, word in enumerate(words)}
    ix2word = {ix: word for word, ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123
    ix_captions = [[[word2ix.get(word, word2ix.get(opt.unknown)) for word in sentence] for sentence in item] for item in cut_captions]
    readme = u"""word：词 ix:index id:图片名 caption: 分词之后的描述，通过ix2word可以获得原始中文词 """
    results = {
        'caption': ix_captions,
        'word2ix': word2ix,
        'ix2word': ix2word,
        'ix2id': ix2id,
        'id2ix': id2ix,
        'padding': '</PAD>',
        'end': '</EOS>',
        'readme': readme
    }
    t.save(results, opt.save_path)
    print('save file in %s' % opt.save_path)

    # test
    def test(ix, ix2=4):
        results = t.load(opt.save_path)
        ix2word = results['ix2word']
        examples = results['caption'][ix][ix2]
        sentences_p = (''.join([ix2word[ii] for ii in examples]))
        sentences_r = data[ix]['caption'][ix2]
        assert sentences_p == sentences_r, 'test failed'
    test(1000)
    print('test success')

# Run the code
process(annotation_file='ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json',max_words=5000)

# Execute methods on the command line

'''
if __name__ == '__main__':
    import fire
    fire.Fire()
    # python data_preprocess.py process --annotation-file=ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json --max-words=5000
'''