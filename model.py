# Image Chinese Description Generation Based on Multi-level Selective Visual Semantic Attributes

import torch as t
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch import nn
import tqdm
import time
import win_unicode_console
win_unicode_console.enable()
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from torchnet import meter
from utils import Visualizer
import torchvision as tv
from PIL import Image

class CaptionDataset():
    def __init__(self):
        data = t.load('caption.pth')
        word2ix = data['word2ix']
        self.ix2word = data['ix2word']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        all_low = t.load('results.pth')
        self.all_low = all_low
        all_pic_r = t.load('data_save/multi_label_extract_pic.pth')
        all_pic = all_pic_r['multi_label_extract_pic']
        self.all_pic = all_pic
        all_block_r = t.load('data_save/imageai_multi_label_extract_block.pth')
        all_block = all_block_r['multi_label_extract_block']
        self.all_block = all_block
    def __getitem__(self, index):
        img_low = self.all_low[index]
        img_one = self.all_pic[index]
        img_block = self.all_block[index]
        img_high = t.cat((img_one.unsqueeze(0), img_block), 0)
        caption = self.captions[index]
        rdn_index = np.random.choice(len(caption), 1)[0]
        caption = caption[rdn_index]
        caption = t.LongTensor(caption)
        return img_low, img_high, caption, index
    def __len__(self):
        return len(self.ix2id)

def create_collate_fn(padding, eos, max_length=50):
    def collate_fn(img_cap):
        img_cap.sort(key=lambda x: len(x[2]), reverse=True)
        img_lows, img_highs, captions, indexs = zip(*img_cap)
        img_lows = t.cat([img_low.unsqueeze(0) for img_low in img_lows], 0)
        img_highs = t.cat([img_high.unsqueeze(0) for img_high in img_highs], 0)
        lengths = [min(len(c) + 1, max_length) for c in captions]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(len(captions), batch_length).fill_(padding)
        for i, c in enumerate(captions):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[i, end_cap] = eos
            cap_tensor[i, :end_cap].copy_(c[:end_cap])
        return (img_lows, img_highs, cap_tensor, lengths, indexs)
    return collate_fn

def get_dataloader():
    dataset = CaptionDataset()
    dataloader = data.DataLoader(dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 drop_last =True,
                                 collate_fn=create_collate_fn(padding=1, eos=2),
                                 )
    return dataloader

class IMAGE_AI_MODEL(nn.Module):
    def __init__(self, hidden_size=512, output_size=5003, dropout_p=0.1, max_length=50):
        super(IMAGE_AI_MODEL, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.fc = nn.Linear(2048, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, 3)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, cell_hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # encoder
        if hidden.size()[2] == 2048:
            hidden = self.fc(hidden)
            hidden = F.relu(hidden)
            cell_hidden = self.fc(cell_hidden)
            cell_hidden = F.relu(cell_hidden)
        encoder_outputs = self.fc(encoder_outputs)
        encoder_outputs = F.relu(encoder_outputs)
        # Attention mechanism
        attn_weights = F.softmax(self.attn(t.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = t.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = t.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        # decoder
        output, (hidden, cell_hidden) = self.lstm(output, (hidden, cell_hidden))
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, cell_hidden, attn_weights
    def initHidden(self):
        return t.zeros(1, 1, self.hidden_size)

def train():
    model = IMAGE_AI_MODEL()
    model.train()
    model.cuda()
    criterion = t.nn.NLLLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = get_dataloader()
    data_set = dataloader.dataset
    print(len(data_set))
    ix2word = dataloader.dataset.ix2word
    _data = dataloader.dataset._data
    vis = Visualizer(env='word_embedding_caption')
    loss_meter = meter.AverageValueMeter()
    for epoch in range(10):
        loss_meter.reset()
        for ii, (img_lows, img_highs, cap_tensor, lengths, indexs) in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            loss = 0
            bitch_target_length = 0
            for i in range(8):
                decoder_hidden = img_lows[[i]].unsqueeze(0)
                cell_hidden = decoder_hidden.clone()
                encoder_outputs = img_highs[i]
                target_tensor = cap_tensor[i]
                target_length = lengths[i]
                bitch_target_length += target_length
                decoder_input = t.tensor([0])
                decoder_hidden = decoder_hidden.cuda()
                cell_hidden = cell_hidden.cuda()
                encoder_outputs = encoder_outputs.cuda()
                target_tensor = target_tensor.cuda()
                decoder_input = decoder_input.cuda()
                raw_img = _data['ix2id'][indexs[i]]
                img_path_q = 'ai_challenger_caption_train_20170902/caption_train_images_20170902/'
                img_path = img_path_q + raw_img
                ture_words = []
                for w in range(target_length):
                    ture_words.append(ix2word[target_tensor[w].item()])
                    ture_words.append('|')
                decoded_words = []
                for di in range(target_length):
                    decoder_output, decoder_hidden, cell_hidden, decoder_attention = model(decoder_input,
                                                                                           decoder_hidden, cell_hidden,
                                                                                           encoder_outputs)
                    loss += criterion(decoder_output, target_tensor[[di]])
                    decoder_input = target_tensor[[di]]
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == 2:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(ix2word[topi.item()])
            loss.backward()
            loss_batch = loss.item() / bitch_target_length
            loss_meter.add(loss_batch)
            optimizer.step()
            plot_every = 10
            if (ii + 1) % plot_every == 0:
                vis.plot('loss', loss_meter.value()[0])
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = tv.transforms.ToTensor()(raw_img)
                vis.img('raw', raw_img)
                raw_caption = ''.join(decoded_words)
                vis.text(raw_caption, win='raw_caption')
                ture_caption = ''.join(ture_words)
                vis.text(ture_caption, win='ture_caption')
        # save
        prefix = 'IMAGE_AI_MODEL'
        path = '{prefix}_{time}'.format(prefix=prefix, time=time.strftime('%m%d_%H%M'))
        t.save(model.state_dict(), path)
train()

'''
if __name__ == '__main__':
    import fire
    fire.Fire()

'''