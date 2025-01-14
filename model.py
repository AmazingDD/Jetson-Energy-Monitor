import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional

class SVGG9(nn.Module):
    def __init__(self, num_classes=10, C=3, H=32, W=32, T=4):
        super().__init__()

        self.features = nn.Sequential(
            layer.Conv2d(C, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            layer.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
            layer.MaxPool2d(kernel_size=2, stride=2, padding=0),
            layer.Flatten(start_dim=1, end_dim=-1),
            layer.Linear(in_features=256 * (H // 8) * (W // 8), out_features=1024, bias=False),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0),
        )
        
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=False)

        self.T = T
        functional.set_step_mode(self, "m") 

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) # (T, B, C, H, W)

        functional.reset_net(self)

        out = self.features(x)
        out = self.classifier(out)

        return out
    
class VGG9(nn.Module):
    def __init__(self, num_classes=10, C=3, H=32, W=32, T=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=256 * (H // 8) * (W // 8), out_features=1024, bias=False),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes, bias=False)

        self.T = T

    def forward(self, x):
        out = 0.
        for _ in range(self.T):
            _out = self.features(x)
            _out = self.classifier(_out)
            out += _out

        out /= self.T

        return out