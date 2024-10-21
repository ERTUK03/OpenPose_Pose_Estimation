from model.convolution_block import ConvolutionBlock
import torch

class Stage(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Stage, self).__init__()
        mid_channels = input_channels // 2
        self.conv_block1 = ConvolutionBlock(input_channels, mid_channels)
        self.conv_block2 = ConvolutionBlock(mid_channels*3, mid_channels)
        self.conv_block3 = ConvolutionBlock(mid_channels*3, mid_channels)
        self.conv_block4 = ConvolutionBlock(mid_channels*3, mid_channels)
        self.conv_block5 = ConvolutionBlock(mid_channels*3, mid_channels)
        self.conv1 = torch.nn.Conv2d(mid_channels*3, mid_channels, kernel_size=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(mid_channels)
        self.conv2 = torch.nn.Conv2d(mid_channels, output_channels, kernel_size=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(output_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x
