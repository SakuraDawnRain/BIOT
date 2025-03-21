import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer


# Simple 3DCNN provided by deepseek
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple3DCNN, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
        #     nn.ReLU(),
        # )
        
        # self.conv = nn.Sequential(
        #     nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        # )
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class EEG2vidEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, time]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :]).unsqueeze(2)
            emb_seq.append(channel_spec_emb)
        # (batch_size, feature, time, emb_size, channel)
        emb = torch.cat(emb_seq, dim=2).permute(0, 3, 1, 2).unsqueeze(1)
        return emb
    
    
# supervised classifier module
class EEG2vidClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()
        # 1. Use STFT to transform EEG into "video" of energy function
        self.eeg_encoder = EEG2vidEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        # 2. Channel embedding such as a linear layer
        self.channel_emb = nn.Linear(23, 64)
        # 3. Use video encoder
        self.vid_encoder = Simple3DCNN()
        # 4. Use classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        z = self.eeg_encoder(x)
        z = self.channel_emb(z)
        z = self.vid_encoder(z)
        y = self.classifier(z)
        return y


if __name__ == "__main__":
    x = torch.randn(16, 23, 2000)
    # model = BIOTClassifier(n_fft=200, hop_length=200, depth=4, heads=8)
    # out = model(x)
    # print(out.shape)

    # model = UnsupervisedPretrain(n_fft=200, hop_length=200, depth=4, heads=8)
    # out1, out2 = model(x)
    # print(out1.shape, out2.shape)
    
    model = EEG2vidClassifier(n_fft=50, hop_length=25, depth=4, heads=8)
    out = model(x)
    print(out.shape)
