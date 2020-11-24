import numpy as np
import pickle
import time
import torch


# def get_glove_emb_by_vocab_pt(trg_emb_fn, vocab_pt_fn, glove_fn):
#     from classifier.my_io import Vocabulary
#     src_vocab = torch.load(vocab_pt_fn, 'rb')[0]
#     vocab_size = len(src_vocab)
#     weights = np.random.normal(0, scale=0.1, size=[vocab_size, 200]).astype(np.float32)
#     hit_cnt = 0
#     t0 = time.time()
#     with open(glove_fn, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.split()
#             word = line[0]
#             if word in src_vocab.word2idx:
#                 id = src_vocab(word)
#                 weights[id] = np.array(line[1:], dtype=np.float32)
#                 hit_cnt += 1
#     print('Find %d / %d from glove embedding, takes %.2f seconds' % (hit_cnt, vocab_size, time.time() - t0))
#     with open(trg_emb_fn, 'wb') as f:
#         pickle.dump(weights, f)
#     print('Saving embedding into %s' % trg_emb_fn)


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
    print('Find %d / %d from glove embedding, takes %.2f seconds' % (hit_cnt, vocab_size, time.time() - t0))
    with open(emb_fn, 'wb') as f:
        pickle.dump(weights, f)
    print('Saving embedding into %s' % emb_fn)
    # Find 1000 / 1000 from glove embedding, takes 18.53 seconds
    # Saving embedding into data/glove_emb.pkl


def get_attribute_emb(src_fn, src_vocab, emb_fn):
    for tag in ['train', 'valid', 'test']:
        cur_src_fn = src_fn.format(tag)
        cur_trg_fn = cur_src_fn.replace('_attribute_cleaned.txt', '_attribute.pt')
        print('Begin generating attributes for %s' % cur_src_fn)
        f = open(emb_fn, 'rb')
        emb = pickle.load(f)
        final_att_embs = []
        with open(cur_src_fn, 'r', encoding='utf-8') as f:
            for line in f:
                atts = line.strip().split()
                att_ids = [src_vocab(att) for att in atts]
                att_embs = [emb[id] for id in att_ids]
                att_emb = np.stack(att_embs, axis=0)  # [5, emb_size=200]
                final_att_embs.append(att_emb)
        final_att_emb = np.stack(final_att_embs, axis=0)  # [file_len, 5, emb_size=200]
        with open(cur_trg_fn, 'wb') as f:
            pickle.dump(final_att_emb, f)
        print('Saving %s of attribute embeddings into %s' % (str(final_att_emb.shape), cur_trg_fn))


# Begin generating attributes for ../TAKG/data/tw_mm_s1/train_attribute_cleaned.txt
# Saving (42959, 5, 200) of attribute embeddings into ../TAKG/data/tw_mm_s1/train_attribute.pt
# Begin generating attributes for ../TAKG/data/tw_mm_s1/valid_attribute_cleaned.txt
# Saving (5370, 5, 200) of attribute embeddings into ../TAKG/data/tw_mm_s1/valid_attribute.pt
# Begin generating attributes for ../TAKG/data/tw_mm_s1/test_attribute_cleaned.txt
# Saving (5372, 5, 200) of attribute embeddings into ../TAKG/data/tw_mm_s1/test_attribute.pt


if __name__ == '__main__':
    glove_fn = '/research/lyu1/yuewang/workspace/embeddings/glove.twitter.27B.200d.txt'
    # vocab_pt_fn = '../classifier/processed_data/tw_mm_s1/vocab.pt'
    # trg_emb_fn = '../classifier/processed_data/tw_mm_s1/glove_emb.pkl'
    # get_glove_emb_by_vocab_pt(trg_emb_fn, vocab_pt_fn, glove_fn)

    # emb_fn = 'data/glove_emb.pkl'
    # vocab_path = 'data/vocab2017_1k_cleaned.pkl'
    # with open(vocab_path, 'rb') as f:
    #     src_vocab = pickle.load(f)
    # vocab_size = len(src_vocab)
    # # prepare_glove_emb(src_vocab, emb_fn, glove_fn)
    #
    # # src_fn = '../TAKG/data/tw_mm_s1/{}_attribute_cleaned.txt'
    # # get_attribute_emb(src_fn, src_vocab, emb_fn)
