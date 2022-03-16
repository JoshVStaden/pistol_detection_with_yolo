import torch.nn as nn
import torch

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (3, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)   
]

position_config = [
    32,
    64,
    256,
    512
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakurelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakurelu(self.batchnorm(self.conv(x)))


class NNBlock(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super(CNNBlock, self).__init__()
        self.dense = nn.Conv2d(in_features, out_features, bias=False,**kwargs)
        self.leakurelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakurelu(self.dense(x))


class Yolov1(nn.module):
    def __init__(self,in_channels=3, **kwargs):
        super(Yolov1,self).__init__()
        self.architecture = architecture_config
        self.position = position_config
        self.position_features = position_config[-1]
        self.in_channels =in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.position_net = self._create_position_net(self.position)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x, x_pos = x
        x = self.darknet(x)
        x_pos = self.position_net(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_position_net(self, config):
        layers = []
        in_features = 2
        for x in config:
            layers += [NNBlock(
                in_features,
                x
            )]
            in_features = x
        return nn.Sequential(*layers)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(
                    in_channels,
                    out_channels=x[1],
                    kernel_size=x[2],
                    padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.maxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats=x[2]
                for _ in range(num_repeats):
                    layers += [CNNBlock(
                        in_channels,
                        out_channels=conv1[1],
                        kernel_size = conv1[0],
                        stride=conv1[2],
                        padding=conv1[3]
                    )]
                    
                    layers += [CNNBlock(
                        conv1[1],
                        out_channels=conv2[1],
                        kernel_size = conv2[0],
                        stride=conv2[2],
                        padding=conv2[3]
                    )]
                    in_channels=conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )

def test(split_size = 7, num_boxes=2, num_classes=20):
    model = Yolov1(split_size, num_boxes, num_classes)
    X = torch.randn((2, 3, 448, 448))
    print(model(X).shape) # Should be [2, 1470] 