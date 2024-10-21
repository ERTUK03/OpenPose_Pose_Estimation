from torchvision import models
from torchvision.models import VGG19_Weights
import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
        vgg19_features = vgg19.features

        self.feature_extractor = torch.nn.Sequential(*list(vgg19_features.children())[:24])

    def forward(self, x):
        return self.feature_extractor(x)
