# coding=utf-8
# Yue Wang, 14/5/2020

import sys
import csv
import base64
import time
import numpy as np
import pickle

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


# do not read 'image_w' and 'image_h'


def get_img_ids(fname):
    with open(fname, 'r', encoding='utf-8') as fr:
        ids = []
        for line in fr:
            ids.append(line.split('/')[-1].strip())
    return ids


def convert_tsv_to_pkl_for_splits(butd_tsv_fn, src_fn, trg_fn):
    # Load tsv butd features into a dictionary
    start_time = time.time()
    print("\nStart to load Faster-RCNN detected objects from %s" % butd_tsv_fn)
    tsv_data_dict = dict()
    with open(butd_tsv_fn) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for idx, item in enumerate(reader):
            image_id = item['image_id']
            features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(36, 2048)
            tsv_data_dict[image_id] = features
            if idx % 1000 == 0:
                print("Loading %d images and takes %.2f seconds" % (idx + 1, time.time() - start_time))

    for split in ['train', 'valid', 'test']:
        t0 = time.time()
        cur_src_fn = src_fn.format(split)
        cur_trg_fn = trg_fn.format(split)
        cur_img_ids = get_img_ids(cur_src_fn)
        image_num = len(cur_img_ids)  # do not remove the duplicate ones
        print('\nConverting %d images for %s split' % (image_num, split))
        cur_feats = []
        for img_id in cur_img_ids:
            cur_feats.append(tsv_data_dict[img_id])

        cur_feats = np.stack(cur_feats, axis=0)
        print('The size of the final img features: %s' % str(cur_feats.shape))
        with open(cur_trg_fn, 'wb') as f:
            pickle.dump(cur_feats, f, protocol=4)
        print('Dump butd features into %s and takes %.2f' % (cur_trg_fn, time.time() - t0))
    return


if __name__ == '__main__':
    data_tag = 'tw_mm_s1'
    butd_fn = '/research/lyu1/yuewang/workspace/singularity/tmp/butd/{}_whole.tsv'.format(data_tag)
    src_fn = '../data/%s/{}_src.txt' % data_tag
    trg_fn = '../data/%s/{}_img_butd.pt' % data_tag

    convert_tsv_to_pkl_for_splits(butd_fn, src_fn, trg_fn)

# Converting 42959 images for train split
# The size of the final img features: (42959, 36, 2048)
# Dump butd features into ../data/tw_mm_s1/train_butd.pt and takes 362.14
#
# Converting 5370 images for valid split
# The size of the final img features: (5370, 36, 2048)
# Dump butd features into ../data/tw_mm_s1/valid_butd.pt and takes 43.94
#
# Converting 5372 images for test split
# The size of the final img features: (5372, 36, 2048)
# Dump butd features into ../data/tw_mm_s1/test_butd.pt and takes 43.80
