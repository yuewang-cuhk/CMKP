import numpy as np
from test_batch import Vocabulary
import pickle
import time


def prepare_glove_emb(src_vocab, emb_fn, glove_fn):
    vocab_size = len(src_vocab)
    weights = np.random.normal(0, scale=0.1, size=[vocab_size, 200]).astype(np.float32)
    hit_cnt = 0
    t0 = time.time()
    with open(glove_fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split()
            word = line[0]
            if word in src_vocab.word2idx:
                id = src_vocab(word)
                weights[id] = np.array(line[1:], dtype=np.float32)
                hit_cnt += 1
    print('Find %d/%d from glove embedding %s, takes %.2f seconds' % (hit_cnt, vocab_size, glove_fn, time.time() - t0))
    with open(emb_fn, 'wb') as f:
        pickle.dump(weights, f)
    print('Saving embedding into %s' % emb_fn)


def get_attribute_emb(src_fn, attr_fn, attr_vocab, emb_fn, trg_fn):
    print('Begin generating attributes for %s' % src_fn)
    t0 = time.time()
    attr_dict = dict()
    with open(attr_fn, 'r', encoding='utf-8') as f:
        for line in f:
            k = line.split(':')[0].strip()
            v = line.split(':')[1].strip().split(';')
            attr_dict[k] = v
    with open(emb_fn, 'rb') as f:
        emb = pickle.load(f)
    print('Read attributes from %s and all attribute emb from %s and takes %f' % (attr_fn, emb_fn, time.time() - t0))

    final_att_embs = []
    with open(src_fn, 'r', encoding='utf-8') as f:
        for line in f:
            cur_img = line.split('<sep>')[-1].split('/')[-1].strip()
            atts = attr_dict[cur_img]
            att_ids = [attr_vocab(att) for att in atts]
            att_embs = [emb[id] for id in att_ids]
            att_emb = np.stack(att_embs, axis=0)  # [5, emb_size=200]
            final_att_embs.append(att_emb)
        final_att_emb = np.stack(final_att_embs, axis=0)  # [file_len, 5, emb_size=200]

    with open(trg_fn, 'wb') as f:
        pickle.dump(final_att_emb, f)
    print('Saving %s of attribute emb into %s and takes %f' % (str(final_att_emb.shape), trg_fn, time.time() - t0))


if __name__ == '__main__':
    glove_fn = '/research/lyu1/yuewang/workspace/embeddings/glove.twitter.27B.200d.txt'
    emb_fn = 'attribute_glove_emb.pkl'
    vocab_path = 'vocab2017_1k_cleaned.pkl'
    data_tag = 'tw_mm_s1'
    with open(vocab_path, 'rb') as f:
        attr_vocab = pickle.load(f)
    vocab_size = len(attr_vocab)
    prepare_glove_emb(attr_vocab, emb_fn, glove_fn)

    attr_fn = 'CMKP_attributes.txt'
    for data_tag in ['tw_mm_s1']:
        for split in ['valid', 'train', 'test']:
            src_fn = '../data/{}/{}_src.txt'.format(data_tag, split)
            trg_fn = '../data/{}/{}_attribute.pt'.format(data_tag, split)
            get_attribute_emb(src_fn, attr_fn, attr_vocab, emb_fn, trg_fn)
