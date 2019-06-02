import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


class Network(nn.Module):
    def __init__(self, init_weights=True):
        super(Network, self).__init__()
        resnet_gray_model = models.resnet18(num_classes=365)
        resnet_gray_model.conv1.weight = nn.Parameter(resnet_gray_model.conv1.weight.sum(dim=1).unsqueeze(1).data)
        self.global_resnet = nn.Sequential(*list(resnet_gray_model.children())[0:9])
        self.lowlevel = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.afterfusion = nn.Conv2d(256, 256, kernel_size=1)
        self.color = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.UpsamplingNearest2d(scale_factor=2),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x1 = self.lowlevel(x)
        x2 = self.global_resnet(x)
        print(x1.shape)
        print(x2.shape)
        x = x2
        for i in range(28 // 1 -1):
            x = torch.cat((x, x2), 2)
        x2 = x
        for i in range(28 // 1 - 1):
            x = torch.cat((x, x2), 3)
        x = torch.cat((x, x1), 1)
        x = self.afterfusion(x)
        x = self.color(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    rand_image = torch.randn(1,1,224,224)
    model = Network()
    print(model)
    output = model(rand_image)
    print(output)
    print(output.shape)

