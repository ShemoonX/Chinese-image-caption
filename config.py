# coding:utf-8

class Config:
    caption_data_path = 'caption.pth'
    img_path = 'ai_challenger_caption_train_20170902/caption_train_images_20170902/'
    img_feature_path = 'results.pth'
    img_pic_path = 'multi_label_extract_pic.pth'
    img_block_path = 'multi_label_extract_block.pth'
    scale_size = 300
    img_size = 224
    batch_size = 8
    shuffle = True
    num_workers = 4
    rnn_hidden = 512
    embedding_dim = 512
    num_layers = 1
    share_embedding_weights = False
    prefix = 'caption'
    env = 'caption'
    plot_every = 10
    debug_file = '/tmp/debugc'
    model_ckpt = None
    lr = 1e-3
    use_gpu = True
    epoch = 25
    test_img = 'img/example.jpeg'
