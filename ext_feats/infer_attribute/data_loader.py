import torch
import torch.utils.data as data
import os
from PIL import Image


class TweetImage(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, examples=None, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        if examples:
            self.examples = examples
        else:
            self.examples = os.listdir(root)
        print('Building TweetImage class with %d examples' % len(examples))
        self.transform = transform

    def __getitem__(self, index):
        example = self.examples[index]
        image = Image.open(os.path.join(self.root, example))
        image = image.resize([224, 224], Image.LANCZOS)

        if self.transform is not None:
            image = self.transform(image)

        return image, example

    def __len__(self):
        return len(self.examples)


def get_tweet_img_loader(root, pred_fn, batch_size, transform, valid_img_fn=None, shuffle=False, num_workers=4):
    used_imgs = set()
    if os.path.exists(pred_fn):
        imgs_list = []
        with open(pred_fn, 'r') as f:
            for line in f:
                imgs_list.append(line.split(':')[0].strip())
        used_imgs = set(imgs_list)
        print('There are %d images have been predicted' % len(used_imgs))

    if valid_img_fn:
        with open(valid_img_fn, 'r', encoding='utf-8') as fr:
            imgs = [line.split(':')[0].strip() for line in fr if int(line.split(':')[1].strip()) == 0]
    else:
        imgs = None

    if os.path.exists(pred_fn):
        imgs = [img for img in imgs if img not in used_imgs]

    tweet_imgs = TweetImage(root, imgs, transform)

    def collate_fn(batch):
        # Merge images (from tuple of 3D tensor to 4D tensor).
        images = [b[0] for b in batch]
        images = torch.stack(images, 0)
        img_fns = [b[1] for b in batch]
        return images, img_fns

    data_loader = torch.utils.data.DataLoader(dataset=tweet_imgs,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
