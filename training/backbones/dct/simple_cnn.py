import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) 

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  

        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.num_classes = num_classes

        input_h, input_w = input_size
        flattened_size = 32 * (input_h // 4) * (input_w // 4)

        self.fc = nn.Linear(flattened_size, num_classes)

    def extract_feature(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = F.relu(self.conv4(x))

        x = self.flatten(x)
    
        return x
    
    def classify(self,feature):
        logits = self.fc(feature)
        # if self.num_classes == 1:
        #     out = torch.sigmoid(logits)
        # else:
        #     out = torch.softmax(logits, dim=1)

        return logits
    def forward(self, x):
        feat = self.extract_feature(x)
        out = self.classify(feat)
        return out
        

        