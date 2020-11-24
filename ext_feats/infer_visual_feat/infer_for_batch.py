import torch
import numpy as np
import argparse
import pickle
from infer_visual_feature.model import EncoderCNN
import time
from infer_visual_feature.data_loader import get_tweet_img_loader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Build models
    encoder = EncoderCNN(args.img_ext_model).eval()  # eval mode (batchnorm uses moving mean/variance)
    encoder = encoder.to(device)
    print('Finish restore the model!')

    for type_tag in ['valid', 'train', 'test']:  # 'train', 'valid', 'test'
        t0 = time.time()
        img_fn = args.img_fn.format(type_tag)
        pred_fn = args.pred_fn.format(type_tag, args.img_ext_model)

        # Load the trained model parameters
        data_loader = get_tweet_img_loader(args.img_dir, img_fn, args.batch_size)
        print('\nPrepare data loader for %s' % img_fn)

        total_step = len(data_loader)
        print('\nBegin inferring for %d batches with batch_size: %d' % (total_step, args.batch_size))
        img_feats = []
        for batch_id, (images, img_id) in enumerate(data_loader):

            images = images.to(device)
            outputs = encoder(images.squeeze())
            outputs = outputs.detach().cpu().numpy()
            img_feats.append(outputs)
            if (batch_id + 1) % max(1, (total_step // 10)) == 0:
                print('Processing %d / %d steps, takes %.2f seconds' % (batch_id + 1, total_step, time.time() - t0))

        print('Finish inferring features for %s and take %.2f seconds' % (img_fn, time.time() - t0))

        img_feats = np.concatenate(img_feats, axis=0)
        print('The size of the final img features: %s' % str(img_feats.shape))
        with open(pred_fn, 'wb') as f:
            pickle.dump(img_feats, f, protocol=4)

        print('Write %s of image features into %s' % (str(img_feats.shape), pred_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='/research/lyu1/yuewang/workspace/MMKG_images', help='')
    # /research/lyu1/yuewang/workspace/MMKG_images
    parser.add_argument('--img_fn', type=str, default='../data/tw_mm_s4/{}_src.txt', help='')
    parser.add_argument('--pred_fn', type=str, default='../data/tw_mm_s4/{}_img_{}.pt', help='')
    parser.add_argument('--img_ext_model', type=str, choices=['resnet', 'vgg', 'complex_resnet'],
                        default='vgg', help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    args = parser.parse_args()
    main(args)
