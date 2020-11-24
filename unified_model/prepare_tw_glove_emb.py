import numpy as np
import pickle
import time
import torch
from my_io import Vocabulary
import io
import argparse


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def load_fasttext_emb_for_vocab(fasttext_fn, vocab_pt_fn, trg_emb_fn, emb_size=300):
    src_vocab = torch.load(vocab_pt_fn, 'rb')[0]  # (src_vocab, trg_vocab)
    vocab_size = len(src_vocab)
    weights = np.random.normal(0, scale=0.1, size=[vocab_size, emb_size]).astype(np.float32)
    hit_cnt = 0
    t0 = time.time()
    # data = load_vectors(fasttext_fn)
    fin = io.open(fasttext_fn, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    print('The size of embedding is %d * %d' % (n, d))
    for idx, line in enumerate(fin):
        line = line.rstrip().split(' ')
        word = line[0]
        if word in src_vocab.word2idx:
            id = src_vocab(word)
            try:
                weights[id] = np.array(line[1:], dtype=np.float32)
            except:
                print('word: %s (id: %d) has errors\n' % (word, id))
            hit_cnt += 1
        if idx % 100000 == 0:
            print('Processing %d lines' % idx)
    print('Find %d / %d from glove embedding, takes %.2f seconds' % (hit_cnt, vocab_size, time.time() - t0))
    with open(trg_emb_fn, 'wb') as f:
        pickle.dump(weights, f)
    print('Saving embedding into %s' % trg_emb_fn)


def load_emb_for_vocab(glove_fn, vocab_pt_fn, trg_emb_fn, emb_size=200):
    src_vocab = torch.load(vocab_pt_fn, 'rb')[0]  # (src_vocab, trg_vocab)
    vocab_size = len(src_vocab)
    weights = np.random.normal(0, scale=0.1, size=[vocab_size, emb_size]).astype(np.float32)
    hit_cnt = 0
    t0 = time.time()
    with open(glove_fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            word = line[0]
            if word in src_vocab.word2idx:
                id = src_vocab(word)
                try:
                    weights[id] = np.array(line[1:], dtype=np.float32)
                except:
                    print('word: %s (id: %d) has errors:\nLine length:%d\n' % (word, id, len(line)), line)
                hit_cnt += 1
    print('Find %d/%d from glove embedding, takes %.2f seconds' % (hit_cnt, vocab_size, time.time() - t0))
    with open(trg_emb_fn, 'wb') as f:
        pickle.dump(weights, f)
    print('Saving embedding into %s' % trg_emb_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data and path parameters
    parser.add_argument('-data_tag', type=str, default='tw_mm_s1')
    opt = parser.parse_args()
    vocab_pt_fn = '../processed_data/{}/vocab.pt'.format(opt.data_tag)

    glove_fn = '/research/lyu1/yuewang/workspace/embeddings/glove.twitter.27B.200d.txt'
    # glove_fn = '/research/lyu1/yuewang/workspace/embeddings/glove.6B.200d.txt'
    # glove_fn = '/research/lyu1/yuewang/workspace/embeddings/crawl-300d-2M-subword.vec'
    trg_emb_fn = '../processed_data/{}/glove_emb.pkl'.format(opt.data_tag)
    print('Start loading embedding for vocab: %s' % vocab_pt_fn)

    load_emb_for_vocab(glove_fn, vocab_pt_fn, trg_emb_fn, 200)
    # load_fasttext_emb_for_vocab(glove_fn, vocab_pt_fn, trg_emb_fn, 300)

# Start loading embedding for vocab: ../processed_data/tw_mm_s1/vocab.pt
# Find 36941/45006 from glove embedding, takes 32.77 seconds
# Saving embedding into ../processed_data/tw_mm_s1/glove_emb.pkl
