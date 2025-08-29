import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(BasicBlock, self).__init__()
        if kernel_size == 5: padding = 2
        elif kernel_size == 3: padding = 1
        stride = 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn4 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out2 = self.conv4(x)
        out2 = self.bn4(out2)

        out = out + out2
        out = self.relu(out)
        return out


class resCNN(nn.Module):
    def __init__(self, block, input_channel = 2, inner_channels = [16, 64, 64, 16], output_channel = 2, kernel_size=5):
        super(resCNN, self).__init__()
        self.output_channel = output_channel
        if kernel_size == 5: padding = 2
        elif kernel_size == 3: padding = 1
        stride = 1

        layers = [nn.Sequential(nn.Conv2d(input_channel, inner_channels[0], kernel_size, stride, padding),
                                nn.BatchNorm2d(inner_channels[0]),
                                nn.ReLU())]
        
        for i in range(1, len(inner_channels)):
            layers.append(block(inner_channels[i - 1], inner_channels[i], kernel_size))

        layers.append(nn.Conv2d(inner_channels[-1], output_channel, kernel_size, stride, padding))
        layers.append(nn.BatchNorm2d(output_channel))

        self.layers = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.layers(x)
        x = self.sig(x)
        return x