# coding:utf-8
# The high level feature extractor was used to extract the high level semantic information of the whole picture and save it to multi_label_extract_pic.pth

from config import Config
import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image


opt = Config()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
bitch_size = 16

class CaptionDataset():
    def __init__(self):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(300),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            normalize
        ])
        data = t.load('caption.pth')
        self.ix2id = data['ix2id']
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                       for _ in range(len(self.ix2id))]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        return img, index
    def __len__(self):
        return len(self.imgs)

def create_collate_fn():
    def collate_fn(img_cap):
        imgs,indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        return (imgs, indexs)

    return collate_fn

def get_dataloader():
    dataset = CaptionDataset()
    print(len(dataset))
    dataloader = data.DataLoader(dataset,
                                batch_size=bitch_size,
                                shuffle=False,
                                collate_fn=create_collate_fn()
                                )
    return dataloader

def extract():
    model_ckpt = 'muti_labei_classification'
    use_gpu = True
    model = t.load(model_ckpt)
    model.eval()
    if use_gpu:
        model.cuda()
    dataloader = get_dataloader()
    multi_label_extract = []
    with t.no_grad():
        for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
            imgs = imgs.cuda()
            try:
                features = model(imgs)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(t.cuda, 'empty_cache'):
                        t.cuda.empty_cache()
                else:
                    raise exception
            m = t.nn.Sigmoid()
            final_features = t.cat([m(feature).unsqueeze(0) for feature in features], 0)
            multi_label_extract[ii * bitch_size:(ii + 1) * bitch_size] = final_features.data.cpu()

    results = {
        'multi_label_extract_pic': multi_label_extract
    }

    t.save(results, 'multi_label_extract_pic.pth')

extract()
