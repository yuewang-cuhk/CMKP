import torch
import torch.nn as nn
import torchvision.models as models


# model saved in /uac/gds/yuewang/.cache/torch/checkpoints/vgg16-397923af.pth
class EncoderCNN(nn.Module):
    def __init__(self, img_ext_model):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        self.img_ext_model = img_ext_model
        if img_ext_model == 'resnet':
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-1]  # delete the last fc layer.
            self.img_ext = nn.Sequential(*modules)
        elif self.img_ext_model == 'complex_resnet':
            resnet = models.resnet101(pretrained=True)
            modules = list(resnet.children())[:-2]  # delete the last fc layer.
            self.img_ext = nn.Sequential(*modules)
        elif img_ext_model == 'vgg':
            vgg = models.vgg16(pretrained=True)
            modules = list(vgg.children())[:-1]  # delete the last fc layer.
            self.img_ext = nn.Sequential(*modules)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.img_ext(images)

        if self.img_ext_model == 'resnet':
            features = features.reshape(features.size(0), -1)
        elif self.img_ext_model == 'complex_resnet':
            features = features.view([features.size(0), 2048, -1]).permute([0, 2, 1])  # [batch_size, 49, 2048]
        elif self.img_ext_model == 'vgg':
            features = features.view([features.size(0), 512, -1]).permute([0, 2, 1])  # [batch_size, 49, 512]
        return features
