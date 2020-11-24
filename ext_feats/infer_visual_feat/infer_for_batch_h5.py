import torch
import numpy as np
import argparse
from infer_visual_feature.model import EncoderCNN
import time
from infer_visual_feature.data_loader import TweetImage
import h5py

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Build models
    encoder = EncoderCNN(args.img_ext_model).eval()  # eval mode (batchnorm uses moving mean/variance)
    encoder = encoder.to(device)
    print('Finish restore the model!')

    for split_tag in ['valid', 'train', 'test']:  # 'train', 'valid', 'test'
        t0 = time.time()
        img_fn = args.img_fn.format(split_tag)
        pred_fn = args.pred_fn.format(split_tag, args.img_ext_model)

        imgs = [line.strip().split('/')[-1] for line in open(img_fn, 'r')]
        image_num = len(imgs)
        tweet_imgs = TweetImage(args.img_dir, imgs)

        data_loader = torch.utils.data.DataLoader(dataset=tweet_imgs,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

        save_h5 = h5py.File(pred_fn, "w")
        save_h5.attrs['split'] = split_tag
        save_h5.attrs["image_num"] = image_num
        image_ids_h5d = save_h5.create_dataset("image_ids", (image_num,), dtype=h5py.string_dtype())
        features_h5d = save_h5.create_dataset("features", (image_num, 49, 512))

        print('\nPrepare data loader for %s' % img_fn)

        total_step = len(data_loader)
        print('\nBegin inferring for %d batches with batch_size: %d' % (total_step, args.batch_size))

        data_idx = 0
        for batch_id, (images, img_ids) in enumerate(data_loader):

            images = images.to(device)
            outputs = encoder(images.squeeze())
            outputs = outputs.detach().cpu().numpy()
            for idx, img_id in enumerate(img_ids):
                image_ids_h5d[data_idx] = img_id
                features_h5d[data_idx] = outputs[idx]
                data_idx += 1

            if batch_id == 1:
                break
            if (batch_id + 1) % max(1, (total_step // 100)) == 0:
                print('Processing %d / %d steps, takes %.2f seconds' % (batch_id + 1, total_step, time.time() - t0))

        print('Write %d features into %s for %s, take %.2f seconds' % (data_idx, pred_fn, img_fn, time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/research/lyu1/yuewang/workspace/MMKG_images', help='')
    # /research/lyu1/yuewang/workspace/MMKG_images
    parser.add_argument('--img_fn', type=str, default='../data/tw_mm_s3/{}_src.txt', help='')
    parser.add_argument('--pred_fn', type=str, default='../data/tw_mm_s3/{}_img_{}.h5', help='')
    parser.add_argument('--img_ext_model', type=str, choices=['resnet', 'vgg', 'complex_resnet'],
                        default='vgg', help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    args = parser.parse_args()
    main(args)
