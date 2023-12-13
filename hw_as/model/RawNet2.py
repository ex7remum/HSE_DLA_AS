import torch
import torch.nn as nn
import torch.nn.functional as F
from hw_as.model.modules import SincConvFast, ResidualBlock
from hw_as.base import BaseModel


class RawNet2(BaseModel):
    def __init__(self, sinc_out_channels=20, sinc_kernel_size=1024, res1_out_channels=20,
                 res2_out_channels=128, min_low_hz=0, min_band_hz=0, gru_hidden=1024, fc_hidden=1024,
                 num_gru_layers=3, leaky_relu_slope=0.3, **kwargs):
        super(RawNet2, self).__init__()
        self.sinc_layer = SincConvFast(out_channels=sinc_out_channels,
                                       kernel_size=sinc_kernel_size,
                                       min_low_hz=min_low_hz,
                                       min_band_hz=min_band_hz)

        self.bn1 = nn.BatchNorm1d(sinc_out_channels)
        self.bn2 = nn.BatchNorm1d(res2_out_channels)
        self.lrelu = nn.LeakyReLU(leaky_relu_slope)

        resblocks = list()
        resblocks.append(ResidualBlock(in_channels=sinc_out_channels,
                                       out_channels=res1_out_channels))
        resblocks.append(ResidualBlock(in_channels=res1_out_channels,
                                       out_channels=res1_out_channels))
        resblocks.append(ResidualBlock(in_channels=res1_out_channels,
                                       out_channels=res2_out_channels))

        for i in range(3):
            resblocks.append(ResidualBlock(in_channels=res2_out_channels,
                                           out_channels=res2_out_channels))

        self.resblocks = nn.ModuleList(resblocks)
        self.gru = nn.GRU(input_size=res2_out_channels,
                          hidden_size=gru_hidden,
                          num_layers=num_gru_layers,
                          batch_first=True)
        self.fc = nn.Linear(in_features=gru_hidden,
                            out_features=fc_hidden)

        self.out = nn.Linear(in_features=fc_hidden,
                             out_features=2)

    def forward(self, x):
        x = self.sinc_layer(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.bn1(x)
        x = self.lrelu(x)

        for block in self.resblocks:
            x = block(x)

        x = self.bn2(x)
        x = self.lrelu(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.out(x)

        return {
            "logits": x
        }
