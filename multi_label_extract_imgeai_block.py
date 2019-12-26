# This file uses yolov3 in image-ai to block images, and inputs subgraphs into the trained multi-label classifier to obtain the semantic features of image segmentation at the high level

import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image
from imageai.Detection import ObjectDetection
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class CaptionDataset():
    def __init__(self):

        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsYOLOv3()
        self.detector.setModelPath(os.path.join(self.execution_path, "yolo.h5"))
        self.detector.loadModel(detection_speed="flash")

        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            normalize
        ])

        data = t.load('caption.pth')
        self.ix2id = data['ix2id']
        img_path = 'ai_challenger_caption_train_20170902/caption_train_images_20170902/'
        imgs = [os.path.join(img_path, self.ix2id[_]) \
                     for _ in range(len(self.ix2id))]

        self.imgs = imgs


    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img_area = img.size[0] * img.size[1]
        detections = self.detector.detectObjectsFromImage(input_image=self.imgs[index])
        block_num = len(detections)
        cut_imgs = []
        if block_num == 2:
            for eachObject in detections:
                cropped = img.crop(eachObject["box_points"])
                # print(eachObject["box_points"])
                m = eachObject["box_points"][2] - eachObject["box_points"][0]
                n = eachObject["box_points"][3] - eachObject["box_points"][1]
                # print('截图宽：%d,截图高：%d' % (m, n))
                jie = (m * n) / img_area * 100
                # print('截图占比：%.2f%%' % jie)
                if jie < 10:
                    cut_imgs.append(self.transforms(img))
                else:
                    cut_imgs.append(self.transforms(cropped))
        if block_num == 1:
            for eachObject in detections:
                cropped = img.crop(eachObject["box_points"])
                # print(eachObject["box_points"])
                m = eachObject["box_points"][2] - eachObject["box_points"][0]
                n = eachObject["box_points"][3] - eachObject["box_points"][1]
                # print('截图宽：%d,截图高：%d' % (m, n))
                jie = (m * n) / img_area * 100
                # print('截图占比：%.2f%%' % jie)
                if jie < 10:
                    cut_imgs.append(self.transforms(img))
                else:
                    cut_imgs.append(self.transforms(cropped))
            cut_imgs.append(self.transforms(img))
        if block_num == 0:
            for i in range(2):
                cut_imgs.append(self.transforms(img))
        if block_num > 2:
            jie_d = {}
            for eachObject in detections:
                cropped = img.crop(eachObject["box_points"])
                # print(eachObject["box_points"])
                m = eachObject["box_points"][2] - eachObject["box_points"][0]
                n = eachObject["box_points"][3] - eachObject["box_points"][1]
                # print('截图宽：%d,截图高：%d' % (m, n))
                jie = (m * n) / img_area * 100
                # print('截图占比：%.2f%%' % jie)
                jie_d[jie] = cropped
            jie_l = sorted(jie_d.keys(), reverse=True)
            for l in range(2):
                if jie_l[l] < 10:
                    cut_imgs.append(self.transforms(img))
                else:
                    cut_imgs.append(self.transforms(jie_d[jie_l[l]]))
        return cut_imgs, index

    def __len__(self):
        return len(self.imgs)


def create_collate_fn():
    def collate_fn(img_cap):
        final_imgs,indexs = zip(*img_cap)
        return (final_imgs,indexs)
    return collate_fn


def get_dataloader():
    dataset = CaptionDataset()
    print(len(dataset))
    dataloader = data.DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=create_collate_fn(),
                                 )

    return dataloader


def extract():
    model_ckpt = 'muti_labei_classification'
    data = t.load('word_att.pth')
    word_att = data['word_att']
    use_gpu = True
    model = t.load(model_ckpt)
    model.eval()
    if use_gpu:
        model.cuda()
    dataloader = get_dataloader()

    multi_label_extract = []
    with t.no_grad():
        for ii, (final_bitch_imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
            final_features = t.Tensor(2, 2048).fill_(0)
            for i in range(2):
                cut_img = final_bitch_imgs[0][i].unsqueeze(0)
                cut_img = cut_img.cuda()
                try:
                    feature = model(cut_img)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(t.cuda, 'empty_cache'):
                            t.cuda.empty_cache()
                    else:
                        raise exception

                m = t.nn.Sigmoid()
                final_feature = m(feature.data.cpu())

                result_pic_att = [(ix, item) for ix, item in enumerate(final_feature[0])]
                decoded_words = []
                for item in result_pic_att:
                    if item[1] >= 0.5:
                        decoded_words.append(word_att[item[0]][1])

                batch_size = 1
                final_features[i * batch_size:(i + 1) * batch_size] = final_feature
            multi_label_extract.append(final_features)

    results = {
        'multi_label_extract_block': multi_label_extract
    }

    t.save(results, 'imageai_multi_label_extract_block.pth')

extract()