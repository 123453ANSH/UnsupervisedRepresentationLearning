
import torch.nn as nn
#add imports as necessary

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.conv1 = nn.Conv2d(3, 64, 7, padding = 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, padding = 1)

        self.layer1_1 = self.new_block(64, 64, 1)
        self.layer1_2 = self.new_block(64, 64, 1)
        #last conv in the list is to use to increase the feature number on the
        #    input x to the blocks where in_planes != out_planes
        self.layer2_1 = self.new_block(64, 128, 2)
        self.layer2_2 = self.new_block(128, 128, 1)
        self.layer2_conv = nn.Conv2d(64, 128, 1, stride = 2)

        self.layer3_1 = self.new_block(128, 256, 2)
        self.layer3_2 = self.new_block(256, 256, 1)
        self.layer3_conv = nn.Conv2d(128, 256, 1, stride = 2)

        self.layer4_1 = self.new_block(256, 512, 2)
        self.layer4_2 = self.new_block(512, 512, 1)
        self.layer4_conv = nn.Conv2d(256, 512, 1, stride = 2)

        self.avgpool = None #doesnt matter, since final conv outputs Nx512x1x1 output
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        #TODO: implement the forward function for resnet,
        #use all the functions you've made
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #layer1
        x = self.relu(self.layer1_1(x) + x)
        x = self.relu(self.layer1_2(x) + x)

        #layer2
        x = self.relu(self.layer2_1(x) + self.layer2_conv(x))
        x = self.relu(self.layer2_2(x) + x)

        #layer3
        x = self.relu(self.layer3_1(x) + self.layer3_conv(x))
        x = self.relu(self.layer3_2(x) + x)

        #layer4
        x = self.relu(self.layer4_1(x) + self.layer4_conv(x))
        x = self.relu(self.layer4_2(x) + x)

        x = x.view(-1, 512)

        x = self.fc(x)

        return x


    def new_block(self, in_planes, out_planes, stride):
        layers = [nn.Conv2d(in_planes, out_planes, 3, padding = 1, stride = stride),
                  nn.BatchNorm2d(out_planes),
                  nn.ReLU(),
                  nn.Conv2d(out_planes, out_planes, 3, padding = 1),
                  nn.BatchNorm2d(out_planes)]

        #TODO: make a convolution with the above params
        return nn.Sequential(*layers)
