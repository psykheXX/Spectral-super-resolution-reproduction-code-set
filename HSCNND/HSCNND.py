import torch
from torch import nn
import torch.nn.functional as F

class FEN(nn.Module):
    def __init__(self, input_channels):
        super(FEN, self).__init__()

        self.conv31 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = x
        x2 = x

        x1 = F.relu(self.conv31(x1))
        x2 = F.relu(self.conv32(x2))

        x1_branch = x1
        x2_branch = x2

        x1 = F.relu(self.conv11(x1))
        x2 = F.relu(self.conv12(x2))

        FEN_output = torch.cat((x1, x1_branch, x2_branch, x2), dim=1)

        return FEN_output

class FMN(nn.Module):
    def __init__(self, input_channels):
        super(FMN, self).__init__()

        self.conv1in = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=1, stride=1)
        self.conv31 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1)
        self.conv12 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1)
        self.conv1out = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = x

        x = F.relu(self.conv1in(x))

        x1 = F.relu(self.conv31(x))
        x2 = F.relu(self.conv32(x))

        x1_branch = x1
        x2_branch = x2

        x1 = F.relu(self.conv11(x1))
        x2 = F.relu(self.conv12(x2))

        x_cat = torch.cat((x1, x1_branch, x2_branch, x2), dim=1)

        x_conv = F.relu(self.conv1out(x_cat))

        FMB_out = torch.cat((x_in, x_conv), dim=1)

        return FMB_out

class HSCNND(nn.Module):
    def __init__(self, in_channels, FMN_num):
        super(HSCNND, self).__init__()

        self.FMN_num = FMN_num
        self.FMNChannel = 64 + 16 * FMN_num

        self.FEN = FEN(in_channels)
        self.FMN = nn.ModuleList([
            FMN(64+i*16)
            for i in range(self.FMN_num)])
        self.conv1R = nn.Conv2d(in_channels=self.FMNChannel, out_channels=31, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.FEN(x)
        for fmn in self.FMN:
            x = fmn(x)
        x = self.conv1R(x)

        return x







