import argparse
import torch
import numpy as np
import random
import os

import run_batches
from my_io import get_tweet_img_loader, Vocabulary
from model import MultimodalMixture
import time


def prepare_data_loader(split_tag, vocab, trg_class_vocab, opt, is_shuffle=True):
    assert split_tag in ['train', 'valid', 'test']
    data_path = os.path.join(opt.data_path, '{}.pt'.format(split_tag))
    data = torch.load(data_path, 'rb')
    print('Load %d instances from %s' % (len(data), data_path))
    feat_tag = 'vgg' if 'vgg' in opt.img_ext_model else opt.img_ext_model
    img_feats_fn = os.path.join(opt.raw_data_path, '{}_img_{}.pt'.format(split_tag, feat_tag))
    attr_feats_fn = os.path.join(opt.raw_data_path, '{}_attribute.pt'.format(split_tag))
    url_map_fn = os.path.join(opt.raw_data_path, '{}_url_map.pt'.format(split_tag))
    bert_feats_fn = os.path.join(opt.raw_data_path, '{}_bert.pt'.format(split_tag))
    src_str_map_fn = os.path.join(opt.raw_data_path, '{}_src_str_map.pt'.format(split_tag))

    data_loader = get_tweet_img_loader(data, vocab, trg_class_vocab,
                                       opt.use_text, opt.use_img, opt.use_attr, opt.use_bert_src,
                                       img_feats_fn, attr_feats_fn, url_map_fn,
                                       bert_feats_fn, src_str_map_fn,
                                       is_test=opt.is_test, only_classifier=opt.only_classifier, debug=opt.debug,
                                       batch_size=opt.batch_size, shuffle=is_shuffle, num_workers=4)
    return data_loader


def main(opt):
    # Create model directory
    vocab_path = os.path.join(opt.data_path, 'vocab.pt')
    vocab, trg_class_vocab = torch.load(vocab_path, 'rb')
    opt.pad_idx = vocab('<pad>')
    opt.bos_idx = vocab('<bos>')
    opt.eos_idx = vocab('<eos>')
    opt.vocab_size = len(vocab)
    opt.trg_class_vocab_size = len(trg_class_vocab)
    print('\nLoad vocab from %s: token vocab size: %d, trg label vocab size: %d' %
          (vocab_path, opt.vocab_size, opt.trg_class_vocab_size))

    # for converting classifier predictions into ids
    opt.combine_pred = 'combine' in opt.cur_model
    opt.combine_pred_type = 'direct' if 'direct' in opt.cur_model else 'embed'
    opt.vocab = vocab
    opt.trg_class_vocab = trg_class_vocab
    if opt.combine_pred:
        if len(opt.model_path) != 0:
            opt.num_epochs = 5
            opt.epochs_to_save = 1
        opt.learning_rate = 0.0001
        print('\n=====================================================================')
        print('For Combine mode: fix the classifier [%d], set learning rate into %.5f, save model after epoch %d' %
              (opt.fix_classifier, opt.learning_rate, opt.epochs_to_save))
        print('=====================================================================\n')

    # prepare the data loader for train and valid split
    opt.only_classifier = False
    opt.is_test = False
    train_data_loader = prepare_data_loader('train', vocab, trg_class_vocab, opt)
    valid_data_loader = prepare_data_loader('valid', vocab, trg_class_vocab, opt)

    print('Finish preparing data load for train (%d batches) and valid (%d batches) with batch size: %d\n' %
          (len(train_data_loader), len(valid_data_loader), opt.batch_size))

    # Build the models
    t0 = time.time()
    model = MultimodalMixture(opt)
    model = model.to(opt.device)
    if opt.fix_classifier:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.linear_classifer_final.parameters():
            param.requires_grad = False
        print('\n\n=============================Fix the classifier===========================')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params_size = sum([np.prod(p.size()) for p in model_parameters])
    print('Finish building model with %d parameters (%.3fM)' % (params_size, params_size / 1000000))

    opt.model_dir = opt.model_dir + '_%.3fM' % (params_size / 1000000) + '-' + opt.data_tag
    print('\nThe trained models after %d epochs will be saved into %s' % (opt.epochs_to_save, opt.model_dir))

    if len(opt.model_path) != 0:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict)

        # model.load_state_dict(pretrained_model)
        # start_epoch = int(opt.model_path.split('/')[-1].split('_')[0].lstrip('e'))
        start_epoch = 0
        print('Load saved model from %s and continue to train from %d' % (opt.model_path, start_epoch))
    else:
        start_epoch = 0

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    run_batches.train_valid_mixture(model, optimizer, train_data_loader, valid_data_loader, start_epoch, opt)
    print('Finish the whole training and validating, takes %.2f seconds' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument('-data_tag', type=str, default='tw_mm_s1')
    parser.add_argument('-raw_data_path', type=str, default='../data/{}')
    parser.add_argument('-data_path', type=str, default='../processed_data/{}/')
    parser.add_argument('-model_path', type=str, default='')

    parser.add_argument('-emb_path', type=str, default='')
    parser.add_argument('-res_fn', type=str, default='results/auto_results.csv', help="")
    parser.add_argument('-debug', type=int, default=0, help="")
    parser.add_argument('-continue_to_predict', type=bool, default=True, help='')

    # for combine
    parser.add_argument('-fix_classifier', type=int, default=0, help="")

    # Training parameters
    parser.add_argument('-log_step', type=int, default=100, help='step size for printing log info')
    parser.add_argument('-num_epochs', type=int, default=15)
    parser.add_argument('-epochs_to_save', type=int, default=3, help='Empirically set 3 for seq2seq, 5 for classifier')
    parser.add_argument('-early_stop_tolerance', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-learning_rate_decay', type=float, default=0.5)
    parser.add_argument('-max_grad_norm', type=float, default=5)

    # Model parameters
    parser.add_argument('-seed', type=int, default=23)
    parser.add_argument('-is_test', type=bool, default=False, help='')
    parser.add_argument('-use_ocr', type=bool, default=False, help='')
    parser.add_argument('-copy_attn', type=int, default=1, help='')

    parser.add_argument('-cur_model', type=str, default='mixture_img_text_multi_head_att_h4_d128_combine_direct')
    parser.add_argument('-head_diff_weight', type=int)
    parser.add_argument('-img_ext_model', type=str, choices=['resnet', 'complex_resnet', 'vgg', 'simple_vgg', 'butd'],
                        default='vgg')
    parser.add_argument('-emb_type', type=str, choices=['random', 'glove', 'glove200d', 'glove300d', 'fasttext300d'],
                        default='glove', help='')
    parser.add_argument('-emb_size', type=int, default=200, help='dimension of word embedding vectors')

    parser.add_argument('-hidden_size', type=int, default=150, help='dimension of lstm hidden states')
    parser.add_argument('-num_layers', type=int, default=2, help='number of layers in GRU')
    parser.add_argument('-bidirectional', type=bool, default=True, help='')
    parser.add_argument('-dropout', type=float, default=0.1)

    opt = parser.parse_args()

    if opt.use_ocr:
        opt.data_tag = opt.data_tag + '_ocr'

    opt.raw_data_path = opt.raw_data_path.format(opt.data_tag)
    opt.data_path = opt.data_path.format(opt.data_tag)

    if 'bert' in opt.cur_model:
        opt.use_bert_src = True
    else:
        opt.use_bert_src = False

    # Device configuration
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if opt.emb_type != 'random':
        opt.emb_path = os.path.join(opt.data_path, '{}_emb.pkl'.format(opt.emb_type))

    if '300' in opt.emb_type:
        opt.emb_size = 300
    else:
        opt.emb_size = 200


    def make_model_tag(opt):
        model_tag = []

        if opt.use_ocr:
            model_tag.append('ocr')
        if ('seq2seq' in opt.cur_model or 'mixture' in opt.cur_model) and opt.copy_attn:
            model_tag.append('copy')

        opt.use_text = True

        if 'img' in opt.cur_model:
            opt.use_img = True
            model_tag.append(opt.img_ext_model)
        else:
            opt.use_img = False
        if 'attr' in opt.cur_model:
            opt.use_attr = True
        else:
            opt.use_attr = False

        model_tag.append(opt.emb_type)
        model_tag.append('s{}'.format(opt.seed))

        timemark = time.strftime('%m%d-%H%M', time.localtime(time.time()))
        model_tag.append(timemark)

        return '_'.join(model_tag)


    opt.model_tag = make_model_tag(opt)
    opt.model_dir = 'models/{}-{}'.format(opt.cur_model, opt.model_tag)

    if opt.debug:
        opt.model_dir = opt.model_dir + '_debug'
        opt.log_step = 10
        opt.num_epochs = 1
        opt.epochs_to_save = 1
        opt.res_fn = opt.res_fn.replace('.csv', '_debug.csv')

    print(opt)
    main(opt)
