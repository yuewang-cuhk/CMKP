import torch
import argparse
import pickle
import time
from torchvision import transforms
from infer_attribute.model import EncoderCNN
from infer_attribute.data_loader import get_tweet_img_loader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word_stem2word = dict()
        self.word2word_stem = dict()

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def main(args):
    t0 = time.time()
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    # Build models
    encoder = EncoderCNN(vocab_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    encoder = encoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location='cpu'))
    print('Finish restore the model for %s and take %.2f seconds' % (args.encoder_path, time.time() - t0))

    pred_fn = args.pred_fn
    print('Preparing data loader for %s folder with valid image flag file %s' % (args.img_dir, args.img_flag_fn))
    data_loader = get_tweet_img_loader(args.img_dir, args.pred_fn, args.batch_size, transform, args.img_flag_fn)

    total_step = len(data_loader)
    t1 = time.time()
    print('\nBegin inferring for %d batches with batch_size: %d' % (total_step, args.batch_size))
    fw = open(pred_fn, 'a', encoding='utf-8')
    idx = 0
    for batch_id, (images, img_fns) in enumerate(data_loader):

        images = images.to(device)

        # Forward, backward and optimize
        outputs = encoder(images)
        outputs = outputs.squeeze().detach().cpu().numpy()

        for i in range(outputs.shape[0]):
            img_fn = img_fns[i]
            output = outputs[i]
            pred_inds = output.argsort()[-args.top_k:][::-1]
            # Convert word_ids to words
            sampled_attribute = [vocab.idx2word[word_id] for word_id in pred_inds]
            sentence = ';'.join(sampled_attribute)
            fw.write(img_fn + ': ' + sentence + '\n')
            idx += 1

        if batch_id % 50 == 0:
            print('Processing %d / %d steps, writing %d lines,  takes %.2f seconds' % (
            batch_id + 1, total_step, idx, time.time() - t1))
    fw.close()
    # Print out the image and the generated caption
    print('Finish inferring for %s and take %.2f seconds' % (img_fn, time.time() - t1))
    print('Write %d lines of attribute into %s' % (idx, pred_fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str,
                        default='/research/lyu1/yuewang/workspace/MMKG_images', help='')
    parser.add_argument('--pred_fn', type=str, default='MMKG_attributes_new.txt', help='')
    parser.add_argument('--img_flag_fn', type=str, default='../processed_tweets/get_image_flags/image_flags.txt',
                        help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')

    parser.add_argument('--encoder_path', type=str, default='encoder-5-1000.ckpt',
                        help='path for trained encoder')

    parser.add_argument('--vocab_path', type=str, default='vocab2017_1k_cleaned.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--top_k', type=int, default=5, help='')

    args = parser.parse_args()
    main(args)
