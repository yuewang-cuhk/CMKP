# coding=utf-8
# Yue Wang, 14/5/2020

import sys
import csv
import base64
import time
import numpy as np
import h5py

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


# do not read 'image_w' and 'image_h'


def get_img_ids(fname):
    with open(fname, 'r', encoding='utf-8') as fr:
        ids = []
        for line in fr:
            ids.append(line.split('/')[-1].strip())
    return ids


def convert_tsv_to_h5_for_splits(butd_tsv_fn, src_fn, trg_h5_fn):
    # Initialize h5py file for storing butd features
    h5_dict = dict()
    img_id_split = dict()
    split_tags = ['train', 'valid', 'test']
    for split in split_tags:
        cur_src_fn = src_fn.format(split)
        cur_img_ids = get_img_ids(cur_src_fn)
        image_num = len(set(cur_img_ids))
        print('Unique img num: %d for images in %s split' % (image_num, split))

        # map image id to split tag
        for img_id in cur_img_ids:
            img_id_split[img_id] = split

        cur_trg_fn_fn = trg_h5_fn.format(split)
        print('Construct %s for %s' % (cur_trg_fn_fn, cur_src_fn))

        save_h5 = h5py.File(cur_trg_fn_fn, "w")
        save_h5.attrs["image_num"] = image_num
        image_ids_h5d = save_h5.create_dataset("image_ids", (image_num,), dtype=h5py.string_dtype())
        boxes_h5d = save_h5.create_dataset("boxes", (image_num, 36, 4))
        features_h5d = save_h5.create_dataset("features", (image_num, 36, 2048))
        h5_dict[split] = {'h5_f': save_h5, 'image_ids': image_ids_h5d, 'boxes': boxes_h5d, 'features': features_h5d}

    start_time = time.time()
    print("\nStart to load Faster-RCNN detected objects from %s" % butd_tsv_fn)
    split_idx = dict([(tag, 0) for tag in split_tags])
    with open(butd_tsv_fn) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for idx, item in enumerate(reader):
            image_id = item['image_id']
            split_tag = img_id_split[image_id]

            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(36, 4)
            features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(36, 2048)

            h5_dict[split_tag]['image_ids'][split_idx[split_tag]] = image_id
            h5_dict[split_tag]['boxes'][split_idx[split_tag]] = boxes
            h5_dict[split_tag]['features'][split_idx[split_tag]] = features
            split_idx[split_tag] += 1

            if idx % 1000 == 0:
                print("Loading %d images and takes %.2f seconds" % (idx + 1, time.time() - start_time))

    assert idx == sum([len(img_id_split[split]) for split in split_tags])
    for split in split_tags:
        h5_dict[split]['h5_f'].attrs["split"] = split
        h5_dict[split]['h5_f'].close()

    print("Finish converting tsv butd features for %d images in %d seconds." % (idx, time.time() - start_time))
    return


if __name__ == '__main__':
    data_tag = 'tw_mm_s1'
    butd_fn = '/research/lyu1/yuewang/workspace/singularity/tmp/butd/{}_whole.tsv'.format(data_tag)
    src_fn = '../data/%s/{}_src.txt' % data_tag
    h5_fn = '../data/%s/{}.h5' % data_tag

    convert_tsv_to_h5_for_splits(butd_fn, src_fn, h5_fn)
