
# This file is used to train image multi-label classifier

import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torchnet import meter

from torch.utils import data
import os
from PIL import Image
import numpy as np
import time


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class CaptionDataset():
    def __init__(self):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            normalize
        ])
        data = t.load('pic2att.pth')
        self.ix2id = data['ix2id']
        img_path = 'ai_challenger_caption_train_20170902/caption_train_images_20170902/'
        self.imgs = [os.path.join(img_path, self.ix2id[_]) \
                     for _ in range(len(self.ix2id))]
        self.atts = data['pic2att']
        self.word_att = data['word_att']
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        att_list = self.atts[index]
        att = t.Tensor(att_list)
        return img, att, index
    def __len__(self):
        return len(self.ix2id)

def create_collate_fn():
    def collate_fn(img_cap):
        imgs, caps, indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        caps = t.cat([cap.unsqueeze(0) for cap in caps], 0)
        return (imgs, caps, indexs)
    return collate_fn

def get_dataloader():
    dataset = CaptionDataset()
    dataloader = data.DataLoader(dataset,
                                batch_size=16,
                                shuffle=True,
                                collate_fn=create_collate_fn()
                                )
    return dataloader



def train():

    model = tv.models.resnet101(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = t.nn.Linear(2048,2048)
    for name, param in model.named_parameters():
        if name == 'layer4.2.conv2.weight':
            param.requires_grad = True
        if name == 'layer4.2.bn2.weight':
            param.requires_grad = True
        if name == 'layer4.2.bn2.bias':
            param.requires_grad = True
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)
    model.cuda()
    criterion = t.nn.BCEWithLogitsLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = get_dataloader()
    word_att = dataloader.dataset.word_att
    loss_meter = meter.AverageValueMeter()
    epoch = 32

    for epoch in range(epoch):
        loss_meter.reset()
        for ii, (imgs, caps, indexes) in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            imgs = imgs.cuda()
            caps = caps.cuda()
            labels = model(imgs)
            loss = criterion(labels, caps)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            if (ii + 1) % 50 == 0:
                print('epoch:',epoch,'loss:',loss_meter.value()[0])
            if (ii + 1) % 1000 == 0:
                ture_words = []
                print('真实属性词：')
                ture_pic_att = [(ix, item) for ix, item in enumerate(caps[6])]
                for item in ture_pic_att:
                    if item[1] == 1:
                        ture_words.append(word_att[item[0]])
                print(ture_words)
                gen_words = []
                print('预测属性词：')
                m = t.nn.Sigmoid()
                labels_sigmoid = m(labels)
                result_pic_att = [(ix, item) for ix, item in enumerate(labels_sigmoid[6])]
                for item in result_pic_att:
                    if item[1] >= 0.5:
                        gen_words.append(word_att[item[0]])
                print(gen_words)

        prefix = 'muti_labei_classification'
        path = '{prefix}_{time}'.format(prefix=prefix,time=time.strftime('%m%d_%H%M'))

        t.save(model, path)

train()

'''

if __name__ == '__main__':
    import fire

    fire.Fire()

'''