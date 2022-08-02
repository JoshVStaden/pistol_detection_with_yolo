"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms.functional import affine

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    "D5",
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    "D5",    
    (3, 256, 1, 1),
    
    (1, 256, 1, 0),
    
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    
    
    (1, 512, 1, 0),
    
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    
    (3, 1024, 1, 1),
    
    (3, 1024, 2, 1),
    
    (3, 1024, 1, 1),
    
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def _center_image(self, image, center):
        batch_size = image.size()[0]

        img_shape_0 = image.size()[-2]
        img_shape_1 = image.size()[-1]

        img_center_0 = img_shape_0 // 2
        img_center_1 = img_shape_1 // 2

        x_shift = torch.sub(img_center_1, center[...,1], alpha=img_shape_1).type(torch.long)
        y_shift = torch.sub(img_center_0, center[...,0], alpha=img_shape_0).type(torch.long)
        
        mod_im = torch.zeros(image.size(), device= DEVICE)
        for b in range(batch_size):
            mod_im[b, ...] = affine(
                image[b,...], 0,
                [-x_shift[b], -y_shift[b]], 1, 0
            )


        return mod_im
        

        # inds_x = torch.arange(img_shape_1, dtype=torch.long).to(DEVICE).repeat((batch_size, image.size()[1], 1))
        # inds_y = torch.arange(img_shape_0, dtype=torch.long).to(DEVICE).repeat((batch_size, image.size()[1], 1))

        # inds_x = torch.sub(inds_x, x_shift)
        # inds_y = torch.sub(inds_y, y_shift)

        # inds_x = torch.remainder(inds_x, img_shape_1)
        # inds_y = torch.remainder(inds_x, img_shape_0)

        # img_clone = torch.clone(image)



 
        # img_clone[] = image[inds_x]
        # return img_clone




    def forward(self, x):
        x, x_pos = x
        # print(x_pos.size())
        l_pos, r_pos = x_pos[...,:2], x_pos[...,2:]
        l = self._center_image(x, l_pos)
        r = self._center_image(x, r_pos)
        # plt.figure()
        # plt.imshow(x[0,...].cpu()[0,:,:])
        # plt.savefig("transformed.png")
        # plt.close()
        # quit()
        x1 = self.darknet(l)
        pred1 = self.fcs(torch.flatten(x1, start_dim=1)) 

        
        x2 = self.darknet(r)
        pred2 = self.fcs(torch.flatten(x2, start_dim=1)) 

        pred = torch.cat((
            pred1[...,:1], 
            pred2[...,:1],
            pred1[...,1:],
            pred2[...,1:]), axis=1)
        return torch.sigmoid(pred)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                if x[0] == 'D':
                    layers += [nn.Dropout(int(x[1]) / 10)]
                else:
                    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, (B * 3))
            
        )
