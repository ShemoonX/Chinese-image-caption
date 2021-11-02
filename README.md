# Chinese-image-caption

# Image Chinese Description Generation Based on Multi-level Selective Visual Semantic Attributes

The model framework of Chinese image description generation algorithm based on multilevel selective visual semantic attributes is shown in the following figure.
  
![Image text](https://raw.githubusercontent.com/ShemoonX/Chinese-image-caption/master/img-folder/model.PNG)

# Paper
__Journal of Chinese information__ 《Chinese Image Captioning Based on Middle-Level Visual-Semantic Composite Attributes》

[paper](http://jcip.cipsc.org.cn/CN/Y2021/V35/I4/129)

# requirement
Python 3.5 
PyTorch 1.0 
visdom
jieba 0.38 
imageai

evaluation tools:
[AI_Challenger offical eval code](https://github.com/AIChallenger/AI_Challenger_2017)

# Download the data

We tested the algorithm on the largest AI challenges 2017 image Chinese description data set and Flick8k-cn image Chinese description data set.Please download the corresponding training data from the following link：

[AI challenges 2017 image Chinese description data set](https://challenger.ai/?lan=zh)

[Flick8k-CN image Chinese description data set](http://lixirong.net/datasets/flickr8kcn)

# Data preprocessing
You can preprocess the AI challenge data set with __data_preprocess.py__ and the __flick8k_data_preprocess.py__ with the flick8k-cn data set.

# Image multi-label extraction
All the image descriptions in the training set were segmented, irrelevant function words were excluded, and only the nouns, verbs, adjectives, numeral words with clear meanings and the idioms that were agreed to be familiar were retained. Then, 2048 Chinese words with the highest frequency and the most representative image visual elements are selected as the attribute word list.

After the given attribute word list, we associate the image with a set of attribute vectors according to the description of each image in the training set to form a new data set, that is, the data set from the image to the attribute vector.

The multi-label function of image extraction is completed by __data_preprocess_pic2att.py__ corresponding to the training set of AI challenge and __flick8k_data_preprocess_pic2att.py__ corresponding to the data set of flick8k-cn.

# Multi-label detector training
The image high-level attribute feature detector is essentially a multi-label classifier. We use a pre-trained resnet101 network, and this part of the functionality is accomplished by __multi_label_classified.py__.

# Extract the underlying features of the image
We used resnet50, which was pre-trained and removed the final classification layer, as the low-level visual detector of the image, and the output 2048 dimension vector as the low-level visual feature of the image. This is done by __low_feature_extract.py__.

# Image multi-granularity high-level visual feature extraction
First, YOLOv3 was used for object detection of the image, and further processing was conducted according to the detection results to complete the preliminary semantic segmentation of the original image. Finally, the segmented subgraph and the original image were respectively input into the well-trained image high-level attribute feature detector. This part of the function by __multi_label_extract_imgeai_block.py__ and __multi_label_extract_pic.py__.

# Overall model architecture
The visual-semantic representation of the image extracted by the low-level visual feature detector and the high-level attribute feature detector of the multi-granularity image is input into this model, and the multi-level and multi-granularity attribute context representation is formed by combining the attention mechanism. The result is accurate, diverse and relatively vivid descriptions in Chinese.The details of the model are defined in __model.py__.

# Results show
We provide an example of the actual generation of this model using AI challenges 2017 image Chinese to describe the training set training to visually show the actual effect of the model, as shown in the following figure.

![Image text](https://raw.githubusercontent.com/ShemoonX/Chinese-image-caption/master/img-folder/result.PNG)
