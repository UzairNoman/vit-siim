import torch
import torch.nn as nn
import torchvision.models as models

class CNNEmbedder(nn.Module):

    def __init__(self, num_classes=2, hparams=None, freeze=False):
        super().__init__()

        self.hparams=hparams
        self.num_classes = num_classes
        self.freeze = freeze
        
        self.feature_extractor = nn.Sequential(*(list(models.resnet18(pretrained=False).children())[:-2]))
        self.AdAvgP = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.FC = nn.Sequential(nn.Linear(in_features=512, out_features=self.num_classes, bias=True))

        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.
        Inputs:
        - x: PyTorch input Variable
        """
        x = self.feature_extractor(x)
        #torch.Size([18, 512, 1, 1])
        #print(x.shape)
        """
        POTENTIAL TO DO: we need to make sure that output from the feature extractor can be passed to _to_words function. maybe requires playing with the layers (cutting off more of them from feature extractor?)
        """
        if not self.freeze:
            x = self.AdAvgP(x)
            x = x.squeeze()
            x = self.FC(x)
        
        return x