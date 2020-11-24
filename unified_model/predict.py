import argparse
import torch
import os
import time
from my_io import Vocabulary
from model import MultimodalMixture
from sequence_generator import SequenceGenerator, preprocess_beam_search_result
from train import prepare_data_loader
from pred_evaluate_func import pred_evaluate
import numpy as np


def process_opt(opt):
    # get model name
    opt.cur_model = opt.model_path.split('/')[-2].split('-')[0].strip()

    # get dataset setting
    if len(opt.model_path.split('/')[-2].split('-')) == 4:
        # e.g.,  only_text-advanced_emb_glove_s24_0510-1551_10.66M-tw_mm_imagenet_s2
        opt.data_tag = opt.model_path.split('/')[-2].split('-')[-1].strip()
        if opt.data_tag.endswith('_cpred'):
            opt.data_tag = opt.data_tag.rstrip('_cpred')
    opt.raw_data_path = opt.raw_data_path.format(opt.data_tag)
    opt.data_path = opt.data_path.format(opt.data_tag)

    # get modalities used in the model
    # in seq2seq, text is necessary
    opt.use_text = True

    if 'img' in opt.model_path:
        opt.use_img = True
    else:
        opt.use_img = False

    if 'attr' in opt.model_path:
        opt.use_attr = True
    else:
        opt.use_attr = False

    if 'bert' in opt.cur_model:
        opt.use_bert_src = True
    else:
        opt.use_bert_src = False

    # decide the type of visual features
    for img_ext_type in ['complex_resnet', 'resnet', 'simple_vgg', 'vgg', 'butd']:
        if img_ext_type in opt.model_path:
            opt.img_ext_model = img_ext_type

    # decide the emb size and embedding method
    if '300' in opt.emb_type:
        opt.emb_size = 300
    else:
        opt.emb_size = 200

    if 'glove200d' in opt.model_path:
        opt.emb_type = 'glove200d'
    elif 'glove300d' in opt.model_path:
        opt.emb_type = 'glove300d'
    elif 'glove' in opt.model_path:  # glove refers to glove_twitter
        opt.emb_type = 'glove'
    elif 'fasttext300d' in opt.model_path:
        opt.emb_type = 'fasttext300d'
    else:
        opt.emb_type = 'random'

    if opt.emb_type != 'random':
        opt.emb_path = os.path.join(opt.data_path, '{}_emb.pkl'.format(opt.emb_type))

    if 'copy' in opt.model_path:
        opt.copy_attn = True
    else:
        opt.copy_attn = False

    return opt


def run_beam_search(generator, data_loader, opt):
    t0 = time.time()
    pred_output_file = open(opt.pred_path, "w")
    line_cnt = 0
    total_batch_step = len(data_loader)
    with torch.no_grad():
        start_time = time.time()
        print("Receiving %d batches with batch_size=%d" % (len(data_loader), opt.batch_size))
        for batch_i, batch in enumerate(data_loader):

            src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, \
            src_str_list, trg_str_2dlist, original_idx_list, img, attr = batch

            src = src.to(opt.device)
            src_mask = src_mask.to(opt.device)
            src_oov = src_oov.to(opt.device)

            if opt.use_img:
                img = img.to(opt.device)

            if opt.use_attr:
                attr = attr.to(opt.device)

            beam_search_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, img, attr)
            pred_list = preprocess_beam_search_result(beam_search_result, opt.idx2word, opt.vocab_size, oov_lists,
                                                      opt.eos_idx, opt.unk_idx, opt.replace_unk, src_str_list)

            # recover the original order in the dataset
            seq_pairs = sorted(zip(original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists),
                               key=lambda p: p[0])
            original_idx_list, src_str_list, trg_str_2dlist, pred_list, oov_lists = zip(*seq_pairs)

            for pred in pred_list:
                pred_str_list = pred['sentences']
                pred_print_out = ';'.join([' '.join(word_list) for word_list in pred_str_list])
                pred_output_file.write(pred_print_out + '\n')
                line_cnt += 1

            if batch_i % max((total_batch_step // 5), 1) == 0:
                print("Batch [%d/%d], Time: %.2f" % (batch_i + 1, total_batch_step, time.time() - start_time))

    pred_output_file.close()
    print("\nFinish writing %d lines to %s, takes %.2f" % (line_cnt, opt.pred_path, time.time() - t0))


def predict_seq2seq(opt, model=None):
    print('\n==================================Predict Seq2seq===================================')
    opt.pred_path = 'pred/' + '_'.join(opt.model_path.split('/')[-2:]).replace('.ckpt', '_seq2seq.txt')
    opt.test_src_fn = '../data/{}/test_src.txt'.format(opt.data_tag)

    # Create model directory
    vocab_path = os.path.join(opt.data_path, 'vocab.pt')
    vocab, trg_class_vocab = torch.load(vocab_path, 'rb')
    opt.pad_idx = vocab('<pad>')
    opt.bos_idx = vocab('<bos>')
    opt.eos_idx = vocab('<eos>')
    opt.unk_idx = vocab('<unk>')
    opt.vocab_size = len(vocab)
    opt.idx2word = vocab.idx2word
    opt.trg_class_vocab_size = len(trg_class_vocab)
    print('\nLoad vocab from %s: token vocab size: %d, trg label vocab size: %d' %
          (vocab_path, opt.vocab_size, opt.trg_class_vocab_size))

    # default para:
    if model is not None:
        opt.max_length = 6
        opt.beam_size = 10
        opt.n_best = 5
        opt.replace_unk = True
    else:
        opt.combine_pred = 'combine' in opt.model_path
        opt.vocab = vocab
        opt.trg_class_vocab = trg_class_vocab

    # Create data loader
    opt.is_test = True
    opt.only_classifier = False
    opt.debug = 'debug' in opt.model_path
    test_data_loader = prepare_data_loader('test', vocab, trg_class_vocab, opt, is_shuffle=False)

    # Restore the models
    if model is None:
        model = MultimodalMixture(opt).to(opt.device)
        model.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))

    model.eval()
    # Construct the sequence generator based on the pretrained models
    generator = SequenceGenerator(model, beam_size=opt.beam_size, max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attn, cuda=torch.cuda.is_available(), n_best=opt.n_best)

    # run beam search to obtain a ranked list of predictions
    run_beam_search(generator, test_data_loader, opt)

    # Evaluate the predictions
    try:
        return pred_evaluate(opt.pred_path, opt.test_src_fn, opt.res_fn)
    except ZeroDivisionError:
        print('ZeroDivisionError due to the poor performance')
        return [0]


def run_classifier(model, data_loader, opt):
    t0 = time.time()
    line_cnt = 0
    total_test_step = len(data_loader)
    with open(opt.pred_path, 'w') as fw:
        for batch_i, batch in enumerate(data_loader):
            src, src_lens, src_mask, src_oov, oov_lists, trg, trg_class, trg_lens, trg_mask, trg_oov, \
            src_str_list, trg_str_2dlist, original_idx_list, img, attr = batch

            max_num_oov = max([len(oov) for oov in oov_lists])
            if opt.use_text:
                src = src.to(opt.device)
                src_mask = src_mask.to(opt.device)
            if opt.use_img:
                img = img.to(opt.device)
            if opt.use_attr:
                attr = attr.to(opt.device)

            output = model(src, src_lens, src_mask, src_oov, trg, max_num_oov, img, attr, only_classifier=True)

            pred_batch = []
            for b_id in range(output.shape[0]):
                arr = np.array(output[b_id].tolist())
                top_indices = arr.argsort()[-opt.n_best:][::-1]
                preds = [opt.class_idx2word[top_indices[i]] for i in range(opt.n_best)]
                pred_batch.append(preds)

            seq_pairs = sorted(zip(original_idx_list, pred_batch), key=lambda p: p[0])
            sorted_original_idx_list, sorted_pred_batch = zip(*seq_pairs)

            for preds in sorted_pred_batch:
                fw.write(';'.join(preds) + '\n')
                line_cnt += 1

            if batch_i % max((total_test_step // 5), 1) == 0:
                print('Test Step [{}/{}], Time: {:.2f} seconds '
                      .format(batch_i + 1, total_test_step, time.time() - t0))

    print("\nFinish writing %d lines to %s, takes %.2f" % (line_cnt, opt.pred_path, time.time() - t0))


def predict_classifier(opt, model=None):
    print('\n==================================Predict Classifier===================================')
    opt.pred_path = 'pred/' + '_'.join(opt.model_path.split('/')[-2:]).replace('.ckpt', '_classifier.txt')
    opt.test_src_fn = '../data/{}/test_src.txt'.format(opt.data_tag)

    # default para:
    if model is not None:
        opt.n_best = 5

    # Create model directory
    vocab_path = os.path.join(opt.data_path, 'vocab.pt')
    vocab, trg_class_vocab = torch.load(vocab_path, 'rb')

    opt.pad_idx = vocab('<pad>')
    opt.bos_idx = vocab('<bos>')
    opt.eos_idx = vocab('<eos>')
    opt.unk_idx = vocab('<unk>')
    
    opt.vocab_size = len(vocab)
    opt.trg_class_vocab_size = len(trg_class_vocab)
    opt.class_idx2word = trg_class_vocab.idx2word
    print('\nLoad vocab from %s: token vocab size: %d, trg label vocab size: %d' %
          (vocab_path, opt.vocab_size, opt.trg_class_vocab_size))

    opt.is_test = True
    opt.only_classifier = False
    opt.debug = 'debug' in opt.model_path
    test_data_loader = prepare_data_loader('test', vocab, trg_class_vocab, opt, is_shuffle=False)

    # Restore the models
    if model is None:
        model = MultimodalMixture(opt).to(opt.device)
        model.load_state_dict(torch.load(opt.model_path, map_location=lambda storage, loc: storage))
    model.eval()

    # run classifier to obtain a ranked list of predictions
    run_classifier(model, test_data_loader, opt)

    # evaluate
    try:
        return pred_evaluate(opt.pred_path, opt.test_src_fn, opt.res_fn)
    except ZeroDivisionError:
        print('ZeroDivisionError due to the poor performance')
        return [0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and path parameters
    parser.add_argument('-data_tag', type=str, default='tw_mm_s1')
    parser.add_argument('-raw_data_path', type=str, default='../data/{}')
    parser.add_argument('-data_path', type=str, default='processed_data/{}/')
    parser.add_argument('-model_path', type=str, required=True)
    parser.add_argument('-res_fn', type=str, default='results/auto_results.csv', help="")

    # Decode parameters
    parser.add_argument('-run_evaluate', type=bool, default=True)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-beam_size', type=int, default=10)
    parser.add_argument('-n_best', type=int, default=5, help='')
    parser.add_argument('-max_length', type=int, default=6, help='')
    parser.add_argument('-replace_unk', type=bool, default=True, help='')
    parser.add_argument('-log_step', type=int, default=20, help='step size for printing log info')

    # Model parameters
    parser.add_argument('-is_test', type=bool, default=True, help='')
    parser.add_argument('-use_ocr', type=bool, default=False, help='')
    parser.add_argument('-copy_attn', type=bool, default=False, help='')
    parser.add_argument('-cur_model', type=str, default='only_text')
    parser.add_argument('-img_ext_model', type=str, default='vgg')
    parser.add_argument('-emb_type', type=str, default='glove', help='')
    parser.add_argument('-emb_size', type=int, default=200, help='dimension of word embedding vectors')
    parser.add_argument('-hidden_size', type=int, default=150, help='dimension of lstm hidden states')
    parser.add_argument('-num_layers', type=int, default=2, help='number of layers in GRU')
    parser.add_argument('-bidirectional', type=bool, default=True, help='')
    parser.add_argument('-dropout', type=float, default=0.1)

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = process_opt(opt)
    predict_seq2seq(opt)
    # predict_classifier(opt)
