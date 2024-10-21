import torch

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.batch_norm1(x_1)
        x_1 = self.relu(x_1)
        x_2 = self.conv2(x_1)
        x_2 = self.batch_norm2(x_2)
        x_2 = self.relu(x_2)
        x_3 = self.conv3(x_2)
        x_3 = self.batch_norm3(x_3)
        x_3 = self.relu(x_3)
        return torch.cat((x_1, x_2, x_3), dim=1)
