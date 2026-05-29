import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


#VGG-16 model architecture
class VGG16(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        #Block 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)

        #Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, return_indices=True)

        #Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, return_indices=True)

        #Block 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, return_indices=True)

        #Block 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, return_indices=True)

        #Classifier
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if pretrained:
            self._load_imagenet_conv_weights()

    #load pretrained weights from ImageNet
    def _load_imagenet_conv_weights(self):

        #code to fix cerficate problem on Mac
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        pretrained_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        #Mapping: torchvision features index to our layer name
        mapping = {
            '0': 'conv1_1', '2': 'conv1_2',
            '5': 'conv2_1', '7': 'conv2_2',
            '10': 'conv3_1', '12': 'conv3_2', '14': 'conv3_3',
            '17': 'conv4_1', '19': 'conv4_2', '21': 'conv4_3',
            '24': 'conv5_1', '26': 'conv5_2', '28': 'conv5_3',
        }
        pretrained_sd = pretrained_vgg.state_dict()
        for idx, our_name in mapping.items():
            our_layer = getattr(self, our_name)
            our_layer.weight.data.copy_(pretrained_sd[f'features.{idx}.weight'])
            our_layer.bias.data.copy_(pretrained_sd[f'features.{idx}.bias'])
        print("  Loaded pretrained ImageNet conv weights into VGG-16")
        del pretrained_vgg

    def forward(self, x):
        #Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x, _ = self.pool1(x)

        #Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x, _ = self.pool2(x)

        #Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x, _ = self.pool3(x)

        #Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x, _ = self.pool4(x)

        #Block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x, _ = self.pool5(x)

        #Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)
