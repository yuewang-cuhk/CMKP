#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """

# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import caffe
import pprint
import time, os, sys
import random
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 36
MAX_BOXES = 36


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }


def generate_tsv(gpu_id, prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                found_ids.add(int(item['image_id']))
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print
        'GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids))
    else:
        print
        'GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            _t = {'misc': Timer()}
            count = 0
            for im_file, image_id in image_ids:
                if image_id in missing:
                    _t['misc'].tic()
                    writer.writerow(get_detections_from_im(net, im_file, image_id))
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print
                        'GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                            .format(gpu_id, count + 1, len(missing), _t['misc'].average_time,
                                    _t['misc'].average_time * (len(missing) - count) / 3600)
                    count += 1


def merge_tsvs():
    test = ['/work/data/tsv/test2015/resnet101_faster_rcnn_final_test.tsv.%d' % i for i in range(8)]
    outfile = '/work/data/tsv/merged.tsv'
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)

        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
                for item in reader:
                    try:
                        writer.writerow(item)
                    except Exception as e:
                        print  e

if __name__ == '__main__':
    gpu_id = '0' # '0,1,2,3'
    prototxt = 'models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt'
    caffemodel = 'data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel'
    cfg_file = 'experiments/cfgs/faster_rcnn_end2end_resnet.yml'
    data_tag = 'tw_mm_s1'
    outfile = '{}_whole.tsv'.format(data_tag)
    if cfg_file is not None:
        cfg_from_file(cfg_file)

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    img_dir = '/research/lyu1/yuewang/workspace/singularity/tmp/{}_images'.format(data_tag)

    image_ids = [(os.path.join(img_dir, img_fn), img_fn) for img_fn in os.listdir(img_dir)]

    if len(gpu_id) == 1:
        # Single GPUs
        generate_tsv(int(gpu_id), prototxt, caffemodel, image_ids, outfile)
    else:
        # Multiple GPUs
        random.seed(10)
        random.shuffle(image_ids)
        # Split image ids between gpus
        gpus = [int(i) for i in gpu_id.split(',')]
        image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]

        caffe.init_log()
        caffe.log('Using devices %s' % str(gpus))
        procs = []

        for i, gpu_id in enumerate(gpus):
            cur_outfile = '%s.%d' % (outfile, gpu_id)
            p = Process(target=generate_tsv,
                        args=(gpu_id, prototxt, caffemodel, image_ids[i], cur_outfile))
            p.daemon = True
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
