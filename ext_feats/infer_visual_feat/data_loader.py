import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import time


class TweetImage(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, examples):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.examples = examples
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

    def __getitem__(self, index):
        example = self.examples[index]
        image = Image.open(os.path.join(self.root, example))
        img = image.resize([224, 224], Image.LANCZOS)
        if self.transform is not None:
            img = self.transform(img)
        return img, example

        # image = image.resize([896, 896], Image.LANCZOS)        #
        # if self.transform is not None:
        #     image = self.transform(image)
        # image = image.unfold(1, 224, 224).unfold(2, 224, 224).contiguous()
        # image = image.view([3, -1, 224, 224])
        # image = image.permute(1, 0, 2, 3)
        # return image

    def __len__(self):
        return len(self.examples)


def get_tweet_img_loader(root, img_fn, batch_size, shuffle=False, num_workers=4):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    imgs = []
    with open(img_fn, 'r', encoding='utf-8') as f:
        for line in f:
            img = line.strip().split('<sep>')[-1].split('/')[-1]
            imgs.append(img)
    tweet_imgs = TweetImage(root, imgs)

    # def collate_fn(images):
    #     # Merge images (from tuple of 3D tensor to 4D tensor).
    #     images = torch.stack(images, 0)
    #     return images

    data_loader = torch.utils.data.DataLoader(dataset=tweet_imgs,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)  # collate_fn=lambda x: x[0]

    return data_loader


if __name__ == '__main__':
    img_fn = '../TAKG/data/tw_mm_s1/train_src.txt'
    root = '../processed_tweets/images'
    with open(img_fn, 'r', encoding='utf-8') as f:
        imgs = []
        for line in f:
            img = line.strip().split('<sep>')[-1].split('/')[-1]
            imgs.append(img)
    tweet_imgs = TweetImage(root, imgs)
    from model import EncoderCNN

    encoder = EncoderCNN()
    img_num = len(tweet_imgs)
    for idx in range(1):
        if (idx + 1) % 100 == 0:
            print('Processing %d / %d lines' % (idx + 1, img_num))
        img = tweet_imgs[idx]
        output = encoder(img)
        print('id: %d, img: %s, output: %s' % (idx, str(img.shape), str(output.shape)))
